package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/mosajjal/frugalai/internal/model"
	"github.com/mosajjal/frugalai/internal/openrouter"
)

// Handler handles OpenAI-compatible API requests
type Handler struct {
	selector     *model.Selector
	client       *openrouter.Client
	modelManager *openrouter.ModelManager
	mu           sync.RWMutex
}

// NewHandler creates a new OpenAI-compatible handler (legacy, for compatibility)
func NewHandler(selector *model.Selector, client *openrouter.Client) *Handler {
	return &Handler{
		selector: selector,
		client:   client,
	}
}

// NewHandlerWithManager creates a new OpenAI-compatible handler with model manager
func NewHandlerWithManager(selector *model.Selector, client *openrouter.Client, mgr *openrouter.ModelManager) *Handler {
	return &Handler{
		selector:     selector,
		client:       client,
		modelManager: mgr,
	}
}

// RegisterRoutes registers the OpenAI-compatible routes
func (h *Handler) RegisterRoutes(mux *http.ServeMux, path string) {
	mux.HandleFunc(path+"/chat/completions", h.handleChatCompletions)
	mux.HandleFunc(path+"/models", h.handleModels)
}

// handleChatCompletions handles chat completion requests with error handling and retry
func (h *Handler) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		h.writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	// Parse request
	var req openrouter.ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		h.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid request body: %v", err))
		return
	}

	// Get current model ID
	modelID := h.getCurrentModelID()
	if modelID == "" {
		h.writeError(w, http.StatusServiceUnavailable, "no available model")
		return
	}

	// Always replace incoming model with current model (this is a proxy)
	req.Model = modelID

	// Handle streaming vs non-streaming
	if req.Stream {
		h.handleStream(w, r, &req)
		return
	}

	// Try request with retry on error
	maxRetries := 3
	var lastErr error
	var resp *openrouter.ChatResponse

	for attempt := 0; attempt < maxRetries; attempt++ {
		// Update model for this attempt
		req.Model = h.getCurrentModelID()
		if req.Model == "" {
			h.writeError(w, http.StatusServiceUnavailable, "no available model")
			return
		}

		resp, lastErr = h.client.ChatCompletion(&req)

		if lastErr == nil {
			// Success - write response
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Model-Used", req.Model)
			if err := json.NewEncoder(w).Encode(resp); err != nil {
				log.Printf("failed to encode response: %v", err)
			}
			return
		}

		if h.recoverModel(req.Model, lastErr, attempt, maxRetries) {
			continue
		}
		break
	}

	// All retries exhausted
	h.writeError(w, http.StatusInternalServerError, fmt.Sprintf("chat completion failed after %d attempts: %v", maxRetries, lastErr))
}

// handleStream handles streaming chat completion requests with retry on failure
func (h *Handler) handleStream(w http.ResponseWriter, r *http.Request, req *openrouter.ChatRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		h.writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	maxRetries := 3
	for attempt := 0; attempt < maxRetries; attempt++ {
		req.Model = h.getCurrentModelID()
		if req.Model == "" {
			h.writeError(w, http.StatusServiceUnavailable, "no available model")
			return
		}

		// Create a cancellable context for this attempt
		ctx, cancel := context.WithCancel(r.Context())
		chunkChan, errChan := h.client.StreamChatCompletionWithContext(ctx, req)

		// Wait for first chunk or error with a first-byte timeout
		firstByteTimer := time.NewTimer(15 * time.Second)

		select {
		case chunk, ok := <-chunkChan:
			firstByteTimer.Stop()
			if !ok {
				// Stream closed without data — check for error
				cancel()
				if err := <-errChan; err != nil {
					if h.handleStreamRetry(req.Model, err, attempt, maxRetries) {
						continue
					}
					h.writeError(w, http.StatusBadGateway, err.Error())
					return
				}
				// Empty successful response
				h.commitStreamHeaders(w, req.Model)
				h.writeServerEvent(w, "done", nil)
				flusher.Flush()
				return
			}

			// Got first chunk — commit headers and forward the stream
			h.commitStreamHeaders(w, req.Model)
			h.writeServerEvent(w, "chunk", chunk)
			flusher.Flush()
			h.forwardStream(w, flusher, r, chunkChan, errChan, req.Model, cancel)
			return

		case err := <-errChan:
			firstByteTimer.Stop()
			cancel()
			if err != nil {
				if h.handleStreamRetry(req.Model, err, attempt, maxRetries) {
					continue
				}
				h.writeError(w, http.StatusBadGateway, err.Error())
				return
			}
			return

		case <-firstByteTimer.C:
			// First-byte timeout — treat as model timeout
			cancel()
			if h.recordTimeout(req.Model) {
				log.Printf("[INFO] Model %s stream timed out, switching (attempt %d/%d)", req.Model, attempt+1, maxRetries)
				continue
			}
			h.writeError(w, http.StatusGatewayTimeout, "stream timed out")
			return

		case <-r.Context().Done():
			firstByteTimer.Stop()
			cancel()
			return
		}
	}

	h.writeError(w, http.StatusBadGateway, fmt.Sprintf("streaming failed after %d attempts", maxRetries))
}

// commitStreamHeaders writes the SSE headers once we're committed to a stream
func (h *Handler) commitStreamHeaders(w http.ResponseWriter, modelID string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("X-Model-Used", modelID)
}

// forwardStream forwards remaining chunks from an established stream to the client
func (h *Handler) forwardStream(w http.ResponseWriter, flusher http.Flusher, r *http.Request, chunkChan <-chan openrouter.StreamChunk, errChan <-chan error, modelID string, cancel context.CancelFunc) {
	defer cancel()
	for {
		select {
		case chunk, ok := <-chunkChan:
			if !ok {
				// Stream ended — check for trailing error
				select {
				case err := <-errChan:
					if err != nil {
						if apiErr := h.tryParseAPIError(err); apiErr != nil {
							h.recordFailure(modelID, apiErr.Code)
						}
						h.writeServerEvent(w, "error", map[string]string{"error": err.Error()})
						flusher.Flush()
						return
					}
				default:
				}
				h.writeServerEvent(w, "done", nil)
				flusher.Flush()
				return
			}
			h.writeServerEvent(w, "chunk", chunk)
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

// handleStreamRetry checks if a stream error is retryable and switches models if so
func (h *Handler) handleStreamRetry(modelID string, err error, attempt, maxRetries int) bool {
	return h.recoverModel(modelID, err, attempt, maxRetries)
}

// getCurrentModelID gets the current model ID from model manager
func (h *Handler) getCurrentModelID() string {
	if h.modelManager != nil {
		h.mu.RLock()
		current := h.modelManager.Current
		h.mu.RUnlock()

		if current != nil {
			available, err := h.selector.IsModelAvailable(current.ID)
			if err != nil {
				log.Printf("[WARN] Could not verify model availability for %s: %v", current.ID, err)
				return current.ID
			}
			if available {
				return current.ID
			}

			log.Printf("[WARN] Current model %s is no longer in the live model list; refreshing candidates", current.ID)
		}

		if h.refreshCandidates() {
			h.mu.RLock()
			defer h.mu.RUnlock()
			if h.modelManager.Current != nil {
				return h.modelManager.Current.ID
			}
		}
	}

	if id, err := h.selector.GetBestModelID(); err == nil {
		return id
	}
	return ""
}

func (h *Handler) candidateCount() int {
	if h.modelManager != nil && len(h.modelManager.Candidates) > 0 {
		return len(h.modelManager.Candidates)
	}
	return 10
}

func (h *Handler) refreshCandidates() bool {
	if h.modelManager == nil {
		return false
	}

	candidates, err := h.selector.GetTopCandidates(h.candidateCount())
	if err != nil {
		log.Printf("[WARN] Failed to refresh model candidates: %v", err)
		return false
	}
	if len(candidates) == 0 {
		log.Printf("[WARN] Refresh returned no candidates")
		return false
	}

	currentID := ""
	h.mu.RLock()
	if h.modelManager.Current != nil {
		currentID = h.modelManager.Current.ID
	}
	h.mu.RUnlock()

	selected, currentIdx, probe, err := h.selector.SelectWorkingCandidate(candidates, currentID)
	if err != nil {
		log.Printf("[WARN] Candidate refresh probe failed: %v", err)
		return false
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	h.modelManager.Candidates = candidates
	h.modelManager.Current = selected
	h.modelManager.CurrentIdx = currentIdx
	h.modelManager.Failures = make(map[string]int)
	h.modelManager.LastFailure = make(map[string]time.Time)
	h.modelManager.Timeouts = make(map[string]int)
	h.modelManager.Burned = make(map[string]bool)

	log.Printf("[INFO] Refreshed %d live model candidates; current model: %s",
		len(h.modelManager.Candidates), h.modelManager.Current.ID)
	log.Printf("[INFO] Probe succeeded for %s in %dms with reply %q",
		probe.ModelID, probe.Duration.Milliseconds(), probe.Reply)

	return true
}

func (h *Handler) recoverModel(modelID string, err error, attempt, maxRetries int) bool {
	var timeoutErr *openrouter.TimeoutError
	if errors.As(err, &timeoutErr) {
		if h.recordTimeout(modelID) {
			log.Printf("[INFO] Model %s timed out, switching (attempt %d/%d)", modelID, attempt+1, maxRetries)
			return true
		}
		if h.refreshCandidates() {
			log.Printf("[INFO] Refreshed live model list after timeout on %s (attempt %d/%d)", modelID, attempt+1, maxRetries)
			return true
		}
		return false
	}

	apiErr := h.tryParseAPIError(err)
	if apiErr == nil {
		return false
	}

	if h.recordFailure(modelID, apiErr.Code) {
		log.Printf("[INFO] Retrying with new model after upstream status %d (attempt %d/%d)", apiErr.Code, attempt+1, maxRetries)
		return true
	}

	retryable := apiErr.Code == http.StatusBadRequest ||
		apiErr.Code == http.StatusNotFound ||
		apiErr.Code == http.StatusUnprocessableEntity ||
		apiErr.Code == http.StatusTooManyRequests ||
		apiErr.Code >= http.StatusInternalServerError

	if retryable && h.refreshCandidates() {
		log.Printf("[INFO] Refreshed live model list after upstream status %d (attempt %d/%d)", apiErr.Code, attempt+1, maxRetries)
		return true
	}

	return false
}

// tryParseAPIError attempts to parse an error as an API error
func (h *Handler) tryParseAPIError(err error) *openrouter.APIError {
	// Check if it's an HTTP error with status code
	type httpError interface {
		StatusCode() int
	}

	if he, ok := err.(httpError); ok {
		return &openrouter.APIError{
			Code:    he.StatusCode(),
			Message: err.Error(),
		}
	}

	// Try to parse from error string
	errStr := err.Error()
	if strings.Contains(errStr, "status ") {
		// Extract status code
		parts := strings.Split(errStr, "status ")
		if len(parts) > 1 {
			var code int
			fmt.Sscanf(parts[1], "%d", &code)
			if code > 0 {
				return &openrouter.APIError{
					Code:    code,
					Message: errStr,
				}
			}
		}
	}

	return nil
}

// writeServerEvent writes a server-sent event
func (h *Handler) writeServerEvent(w http.ResponseWriter, eventType string, data interface{}) {
	var jsonData string
	if data != nil {
		bytes, err := json.Marshal(data)
		if err != nil {
			return
		}
		jsonData = string(bytes)
	}

	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, jsonData)
}

// handleModels handles model list requests
func (h *Handler) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var models []openrouter.Model
	var err error

	// Use model manager candidates if available
	if h.modelManager != nil && len(h.modelManager.Candidates) > 0 {
		models = h.modelManager.Candidates
	} else {
		models, err = h.client.GetFreeModels()
		if err != nil {
			h.writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to get models: %v", err))
			return
		}
	}

	// Convert to OpenAI format
	type OpenAIModel struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}

	openaiModels := []OpenAIModel{}
	for _, model := range models {
		openaiModels = append(openaiModels, OpenAIModel{
			ID:      model.ID,
			Object:  "model",
			Created: 0,
			OwnedBy: "openrouter",
		})
	}

	response := map[string]interface{}{
		"object": "list",
		"data":   openaiModels,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// InvalidateCache invalidates the cached model ID
func (h *Handler) InvalidateCache() {
	h.mu.Lock()
	defer h.mu.Unlock()
}

// writeError writes an error response
func (h *Handler) writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	errorResp := map[string]interface{}{
		"error": map[string]string{
			"message": message,
			"type":    "invalid_request_error",
			"code":    fmt.Sprintf("%d", status),
		},
	}
	json.NewEncoder(w).Encode(errorResp)
}

// ConvertAnthropicToOpenAI converts Anthropic format to OpenAI format
func ConvertAnthropicToOpenAI(anthropicReq map[string]interface{}) (*openrouter.ChatRequest, error) {
	req := &openrouter.ChatRequest{
		Temperature: 0.7,
		MaxTokens:   4096,
	}

	// Get model
	if model, ok := anthropicReq["model"].(string); ok {
		req.Model = model
	}

	// Get max tokens
	if maxTokens, ok := anthropicReq["max_tokens"].(float64); ok {
		req.MaxTokens = int(maxTokens)
	}

	// Get temperature
	if temp, ok := anthropicReq["temperature"].(float64); ok {
		req.Temperature = temp
	}

	// Convert messages
	messages, ok := anthropicReq["messages"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid messages format")
	}

	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}

		role, _ := msgMap["role"].(string)

		// Handle content (can be string or array of blocks)
		var content string
		switch c := msgMap["content"].(type) {
		case string:
			content = c
		case []interface{}:
			var textParts []string
			for _, block := range c {
				blockMap, ok := block.(map[string]interface{})
				if !ok {
					continue
				}
				if blockType, ok := blockMap["type"].(string); ok && blockType == "text" {
					if text, ok := blockMap["text"].(string); ok {
						textParts = append(textParts, text)
					}
				}
			}
			content = strings.Join(textParts, "\n")
		}

		// Handle system prompt
		if role == "system" {
			// Add as user message for now (OpenRouter will handle it)
			req.Messages = append(req.Messages, openrouter.ChatMessage{
				Role:    "user",
				Content: content,
			})
		} else {
			req.Messages = append(req.Messages, openrouter.ChatMessage{
				Role:    role,
				Content: content,
			})
		}
	}

	return req, nil
}

// recordFailure records a model failure and potentially switches models
func (h *Handler) recordFailure(modelID string, statusCode int) bool {
	if h.modelManager == nil {
		return false
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	h.modelManager.Failures[modelID]++
	h.modelManager.LastFailure[modelID] = time.Now()

	log.Printf("[WARN] Model %s failed (status %d), failure count: %d",
		modelID, statusCode, h.modelManager.Failures[modelID])

	// Switch on rate limit, server error, or 3+ failures
	shouldSwitch := statusCode == 429 || statusCode >= 500 || h.modelManager.Failures[modelID] >= 3

	if shouldSwitch && len(h.modelManager.Candidates) > 1 {
		return h.switchToNextModel()
	}

	return false
}

// recordTimeout records a model timeout and potentially burns/switches it
func (h *Handler) recordTimeout(modelID string) bool {
	if h.modelManager == nil {
		return false
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	h.modelManager.Timeouts[modelID]++

	log.Printf("[WARN] Model %s timed out, timeout count: %d",
		modelID, h.modelManager.Timeouts[modelID])

	// Burn model on first timeout
	h.modelManager.Burned[modelID] = true
	log.Printf("[WARN] Model %s burned after %d timeouts",
		modelID, h.modelManager.Timeouts[modelID])

	if len(h.modelManager.Candidates) > 1 {
		return h.switchToNextModel()
	}

	return false
}

// switchToNextModel switches to the next available non-burned model
func (h *Handler) switchToNextModel() bool {
	for i := 1; i < len(h.modelManager.Candidates); i++ {
		nextIdx := (h.modelManager.CurrentIdx + i) % len(h.modelManager.Candidates)
		nextModel := &h.modelManager.Candidates[nextIdx]

		// Skip burned models
		if h.modelManager.Burned[nextModel.ID] {
			continue
		}

		// Skip models with 3+ recent failures
		if h.modelManager.Failures[nextModel.ID] >= 3 {
			continue
		}

		probe, err := h.selector.ProbeModel(nextModel.ID)
		if err != nil {
			log.Printf("[WARN] Candidate %s failed probe during failover: %v", nextModel.ID, err)
			continue
		}

		log.Printf("[INFO] Switching from %s to %s",
			h.modelManager.Current.ID, nextModel.ID)
		log.Printf("[INFO] Probe succeeded for %s in %dms with reply %q",
			probe.ModelID, probe.Duration.Milliseconds(), probe.Reply)

		h.modelManager.Current = nextModel
		h.modelManager.CurrentIdx = nextIdx
		return true
	}

	log.Printf("[WARN] No alternative models available")
	return false
}
