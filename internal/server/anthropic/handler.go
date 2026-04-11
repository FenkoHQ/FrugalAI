package anthropic

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
	"github.com/mosajjal/frugalai/internal/server/openai"
)

// Handler handles Anthropic-compatible API requests
type Handler struct {
	selector     *model.Selector
	client       *openrouter.Client
	modelManager *openrouter.ModelManager
	mu           sync.RWMutex
}

// NewHandler creates a new Anthropic-compatible handler (legacy)
func NewHandler(selector *model.Selector, client *openrouter.Client) *Handler {
	return &Handler{
		selector: selector,
		client:   client,
	}
}

// NewHandlerWithManager creates a new Anthropic-compatible handler with model manager
func NewHandlerWithManager(selector *model.Selector, client *openrouter.Client, mgr *openrouter.ModelManager) *Handler {
	return &Handler{
		selector:     selector,
		client:       client,
		modelManager: mgr,
	}
}

// RegisterRoutes registers the Anthropic-compatible routes
func (h *Handler) RegisterRoutes(mux *http.ServeMux, path string) {
	mux.HandleFunc(path+"/messages", h.handleMessages)
}

// handleMessages handles message requests with retry on error
func (h *Handler) handleMessages(w http.ResponseWriter, r *http.Request) {
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

	// Parse Anthropic request
	var anthropicReq map[string]interface{}
	if err := json.Unmarshal(body, &anthropicReq); err != nil {
		h.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid request body: %v", err))
		return
	}

	// Check if streaming
	stream := false
	if s, ok := anthropicReq["stream"].(bool); ok {
		stream = s
	}

	if stream {
		h.handleStream(w, r, anthropicReq)
		return
	}

	// Try non-streaming request with retry
	maxRetries := 3
	var lastErr error
	var resp *openrouter.ChatResponse

	for attempt := 0; attempt < maxRetries; attempt++ {
		// Convert to OpenAI format
		openaiReq, err := openai.ConvertAnthropicToOpenAI(anthropicReq)
		if err != nil {
			h.writeError(w, http.StatusBadRequest, fmt.Sprintf("failed to convert request: %v", err))
			return
		}

		// Get current model - always replace with proxy model
		modelID := h.getCurrentModelID()
		if modelID == "" {
			h.writeError(w, http.StatusServiceUnavailable, "no available model")
			return
		}
		openaiReq.Model = modelID

		resp, lastErr = h.client.ChatCompletion(openaiReq)

		if lastErr == nil {
			// Success - convert and write response
			anthropicResp := h.convertToAnthropic(resp)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Model-Used", openaiReq.Model)
			if err := json.NewEncoder(w).Encode(anthropicResp); err != nil {
				log.Printf("failed to encode response: %v", err)
			}
			return
		}

		if h.recoverModel(openaiReq.Model, lastErr, attempt, maxRetries) {
			continue
		}
		break
	}

	// All retries exhausted
	h.writeError(w, http.StatusInternalServerError, fmt.Sprintf("chat completion failed after %d attempts: %v", maxRetries, lastErr))
}

// handleStream handles streaming requests with retry on failure
func (h *Handler) handleStream(w http.ResponseWriter, r *http.Request, anthropicReq map[string]interface{}) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		h.writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	// Convert to OpenAI format once — only model changes per attempt
	openaiReq, err := openai.ConvertAnthropicToOpenAI(anthropicReq)
	if err != nil {
		h.writeError(w, http.StatusBadRequest, fmt.Sprintf("failed to convert request: %v", err))
		return
	}

	maxRetries := 3
	for attempt := 0; attempt < maxRetries; attempt++ {
		openaiReq.Model = h.getCurrentModelID()
		if openaiReq.Model == "" {
			h.writeError(w, http.StatusServiceUnavailable, "no available model")
			return
		}

		ctx, cancel := context.WithCancel(r.Context())
		chunkChan, errChan := h.client.StreamChatCompletionWithContext(ctx, openaiReq)

		// Wait for first chunk or error with a first-byte timeout
		firstByteTimer := time.NewTimer(15 * time.Second)

		select {
		case chunk, ok := <-chunkChan:
			firstByteTimer.Stop()
			if !ok {
				// Stream closed without data — check for error
				cancel()
				if err := <-errChan; err != nil {
					if h.handleStreamRetry(openaiReq.Model, err, attempt, maxRetries) {
						continue
					}
					h.writeError(w, http.StatusBadGateway, err.Error())
					return
				}
				h.commitStreamHeaders(w)
				h.writeAnthropicEvent(w, "message_stop", map[string]interface{}{"type": "message_stop"})
				flusher.Flush()
				return
			}

			// Got first chunk — commit headers and forward the stream
			h.commitStreamHeaders(w)
			eventIndex := 0
			if len(chunk.Choices) > 0 {
				h.writeAnthropicStreamEvent(w, chunk.Choices[0].Message.Content, eventIndex)
				eventIndex++
			}
			flusher.Flush()
			h.forwardAnthropicStream(w, flusher, r, chunkChan, errChan, openaiReq.Model, cancel, eventIndex)
			return

		case err := <-errChan:
			firstByteTimer.Stop()
			cancel()
			if err != nil {
				if h.handleStreamRetry(openaiReq.Model, err, attempt, maxRetries) {
					continue
				}
				h.writeError(w, http.StatusBadGateway, err.Error())
				return
			}
			return

		case <-firstByteTimer.C:
			cancel()
			if h.recordTimeout(openaiReq.Model) {
				log.Printf("[INFO] Model %s stream timed out, switching (attempt %d/%d)", openaiReq.Model, attempt+1, maxRetries)
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
func (h *Handler) commitStreamHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
}

// forwardAnthropicStream forwards remaining chunks from an established stream to the client
func (h *Handler) forwardAnthropicStream(w http.ResponseWriter, flusher http.Flusher, r *http.Request, chunkChan <-chan openrouter.StreamChunk, errChan <-chan error, modelID string, cancel context.CancelFunc, eventIndex int) {
	defer cancel()
	for {
		select {
		case chunk, ok := <-chunkChan:
			if !ok {
				select {
				case err := <-errChan:
					if err != nil {
						if apiErr := h.tryParseAPIError(err); apiErr != nil {
							h.recordFailure(modelID, apiErr.Code)
						}
						h.writeAnthropicEvent(w, "error", map[string]string{"error": err.Error()})
						flusher.Flush()
						return
					}
				default:
				}
				h.writeAnthropicEvent(w, "message_stop", map[string]interface{}{"type": "message_stop"})
				flusher.Flush()
				return
			}
			if len(chunk.Choices) > 0 {
				h.writeAnthropicStreamEvent(w, chunk.Choices[0].Message.Content, eventIndex)
				eventIndex++
			}
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

	h.mu.Lock()
	defer h.mu.Unlock()

	currentID := ""
	if h.modelManager.Current != nil {
		currentID = h.modelManager.Current.ID
	}

	currentIdx := 0
	for i := range candidates {
		if candidates[i].ID == currentID {
			currentIdx = i
			break
		}
	}

	h.modelManager.Candidates = candidates
	h.modelManager.Current = &h.modelManager.Candidates[currentIdx]
	h.modelManager.CurrentIdx = currentIdx
	h.modelManager.Failures = make(map[string]int)
	h.modelManager.LastFailure = make(map[string]time.Time)
	h.modelManager.Timeouts = make(map[string]int)
	h.modelManager.Burned = make(map[string]bool)

	log.Printf("[INFO] Refreshed %d live model candidates; current model: %s",
		len(h.modelManager.Candidates), h.modelManager.Current.ID)

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
	type httpError interface {
		StatusCode() int
	}

	if he, ok := err.(httpError); ok {
		return &openrouter.APIError{
			Code:    he.StatusCode(),
			Message: err.Error(),
		}
	}

	errStr := err.Error()
	if strings.Contains(errStr, "status ") {
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

// writeAnthropicEvent writes an Anthropic-style server-sent event
func (h *Handler) writeAnthropicEvent(w http.ResponseWriter, eventType string, data interface{}) {
	jsonData, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, jsonData)
}

// writeAnthropicStreamEvent writes a content block delta event
func (h *Handler) writeAnthropicStreamEvent(w http.ResponseWriter, content string, index int) {
	delta := map[string]interface{}{
		"type":  "content_block_delta",
		"index": index,
		"delta": map[string]string{
			"type": "text_delta",
			"text": content,
		},
	}
	h.writeAnthropicEvent(w, "content_block_delta", delta)
}

// convertToAnthropic converts OpenRouter response to Anthropic format
func (h *Handler) convertToAnthropic(resp *openrouter.ChatResponse) openrouter.AnthropicResponse {
	content := ""
	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
	}

	return openrouter.AnthropicResponse{
		ID:   resp.ID,
		Type: "message",
		Role: "assistant",
		Content: []openrouter.ContentBlock{
			{
				Type: "text",
				Text: content,
			},
		},
		StopReason: "end_turn",
		Model:      resp.Model,
		Usage: openrouter.AnthropicUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		},
	}
}

// InvalidateCache invalidates the cached model ID
func (h *Handler) InvalidateCache() {
	h.mu.Lock()
	defer h.mu.Unlock()
}

// writeError writes an Anthropic-style error response
func (h *Handler) writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	errorResp := map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    "invalid_request_error",
			"message": message,
		},
	}
	json.NewEncoder(w).Encode(errorResp)
}

// ConvertOpenAIToAnthropic converts OpenAI format to Anthropic format
func ConvertOpenAIToAnthropic(openaiReq *openrouter.ChatRequest) *openrouter.AnthropicRequest {
	req := &openrouter.AnthropicRequest{
		Model:       openaiReq.Model,
		MaxTokens:   openaiReq.MaxTokens,
		Temperature: openaiReq.Temperature,
		TopP:        openaiReq.TopP,
		Stream:      openaiReq.Stream,
	}

	for _, msg := range openaiReq.Messages {
		anthropicMsg := openrouter.AnthropicMessage{
			Role: msg.Role,
			Content: []openrouter.ContentBlock{
				{
					Type: "text",
					Text: msg.Content,
				},
			},
		}
		req.Messages = append(req.Messages, anthropicMsg)
	}

	return req
}

// IsAnthropicRequest checks if the request is in Anthropic format
func IsAnthropicRequest(r *http.Request) bool {
	if r.Header.Get("anthropic-version") != "" || r.Header.Get("anthropic-api-version") != "" {
		return true
	}

	if strings.Contains(r.URL.Path, "/v1/messages") {
		return true
	}

	if r.Method == http.MethodPost {
		body, _ := io.ReadAll(r.Body)
		defer r.Body.Close()

		var req map[string]interface{}
		if json.Unmarshal(body, &req) == nil {
			if _, hasMaxTokens := req["max_tokens"]; hasMaxTokens {
				if _, hasMessages := req["messages"]; hasMessages {
					return true
				}
			}
		}
		r.Body = io.NopCloser(strings.NewReader(string(body)))
	}

	return false
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

		log.Printf("[INFO] Switching from %s to %s",
			h.modelManager.Current.ID, nextModel.ID)

		h.modelManager.Current = nextModel
		h.modelManager.CurrentIdx = nextIdx
		return true
	}

	log.Printf("[WARN] No alternative models available")
	return false
}
