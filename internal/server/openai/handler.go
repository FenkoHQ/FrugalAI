package openai

import (
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

		// Check if it's a timeout error
		var timeoutErr *openrouter.TimeoutError
		if errors.As(lastErr, &timeoutErr) {
			if h.recordTimeout(req.Model) {
				log.Printf("[INFO] Model %s timed out, switching (attempt %d/%d)", req.Model, attempt+1, maxRetries)
				continue
			}
		}

		// Check if error is from API response
		if apiErr := h.tryParseAPIError(lastErr); apiErr != nil {
			// Record failure and try switching
			if h.recordFailure(req.Model, apiErr.Code) {
				log.Printf("[INFO] Retrying with new model (attempt %d/%d)", attempt+1, maxRetries)
				continue
			}
		}

		// For other errors, break
		break
	}

	// All retries exhausted
	h.writeError(w, http.StatusInternalServerError, fmt.Sprintf("chat completion failed after %d attempts: %v", maxRetries, lastErr))
}

// handleStream handles streaming chat completion requests
func (h *Handler) handleStream(w http.ResponseWriter, r *http.Request, req *openrouter.ChatRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		h.writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Update model before starting stream
	req.Model = h.getCurrentModelID()

	chunkChan, errChan := h.client.StreamChatCompletion(req)

	for {
		select {
		case chunk, ok := <-chunkChan:
			if !ok {
				h.writeServerEvent(w, "done", nil)
				flusher.Flush()
				return
			}
			h.writeServerEvent(w, "chunk", chunk)
			flusher.Flush()
		case err := <-errChan:
			if err != nil {
				// Try to parse as API error
				if apiErr := h.tryParseAPIError(err); apiErr != nil {
					h.recordFailure(req.Model, apiErr.Code)
				}
				h.writeServerEvent(w, "error", map[string]string{"error": err.Error()})
				flusher.Flush()
				return
			}
		case <-r.Context().Done():
			return
		}
	}
}

// getCurrentModelID gets the current model ID from model manager
func (h *Handler) getCurrentModelID() string {
	if h.modelManager != nil && h.modelManager.Current != nil {
		return h.modelManager.Current.ID
	}
	// Fallback to selector
	if id, err := h.selector.GetBestModelID(); err == nil {
		return id
	}
	return ""
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
		nextModel := h.modelManager.Candidates[nextIdx]

		// Skip burned models
		if h.modelManager.Burned[nextModel.ID] {
			continue
		}

		// Skip models with 3+ recent failures
		if h.modelManager.Failures[nextModel.ID] >= 3 {
			continue
		}

		log.Printf("[INFO] Switching from %s to %s",
			h.modelManager.Current.ID, nextModel.ID)

		h.modelManager.Current = &nextModel
		h.modelManager.CurrentIdx = nextIdx
		return true
	}

	log.Printf("[WARN] No alternative models available")
	return false
}
