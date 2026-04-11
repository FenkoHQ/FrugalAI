package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	baseURL        = "https://openrouter.ai/api"
	modelsEndpoint = "/v1/models"
	chatEndpoint   = "/v1/chat/completions"
	userAgent      = "curl/8.7.1"
	probePrompt    = "ping"
)

// HTTPError represents an HTTP error with status code
type HTTPError struct {
	Code    int
	Message string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("API returned status %d: %s", e.Code, e.Message)
}

func (e *HTTPError) StatusCode() int {
	return e.Code
}

// TimeoutError represents a request timeout
type TimeoutError struct {
	Duration time.Duration
}

func (e *TimeoutError) Error() string {
	return fmt.Sprintf("request timed out after %v", e.Duration)
}

func (e *TimeoutError) IsTimeout() bool {
	return true
}

// Client represents an OpenRouter API client
type Client struct {
	apiKey     string
	httpClient *http.Client
	cache      *CachedModels
	cacheMutex sync.RWMutex
	cacheTTL   time.Duration
}

// NewClient creates a new OpenRouter client
func NewClient(apiKey string, cacheTTL int) *Client {
	return &Client{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		cacheTTL: time.Duration(cacheTTL) * time.Second,
	}
}

func (c *Client) setCommonHeaders(req *http.Request) {
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	req.Header.Set("User-Agent", userAgent)
}

func (c *Client) setJSONHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	c.setCommonHeaders(req)
}

// GetModels fetches available models from OpenRouter
func (c *Client) GetModels() ([]Model, error) {
	// Check cache first
	c.cacheMutex.RLock()
	if c.cache != nil && time.Since(c.cache.Timestamp) < c.cacheTTL {
		models := c.cache.Models
		c.cacheMutex.RUnlock()
		return models, nil
	}
	c.cacheMutex.RUnlock()

	// Fetch from API
	req, err := http.NewRequest("GET", baseURL+modelsEndpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	c.setCommonHeaders(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &HTTPError{
			Code:    resp.StatusCode,
			Message: string(body),
		}
	}

	var modelsResp ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Update cache
	c.cacheMutex.Lock()
	c.cache = &CachedModels{
		Models:    modelsResp.Data,
		Timestamp: time.Now(),
	}
	c.cacheMutex.Unlock()

	return modelsResp.Data, nil
}

// GetFreeModels returns only free models
func (c *Client) GetFreeModels() ([]Model, error) {
	models, err := c.GetModels()
	if err != nil {
		return nil, err
	}

	freeModels := []Model{}
	for _, model := range models {
		if c.isFreeModel(model) {
			freeModels = append(freeModels, model)
		}
	}

	return freeModels, nil
}

// isFreeModel checks if a model is free
func (c *Client) isFreeModel(model Model) bool {
	// Parse pricing as float to handle "0", "0.00", "0.000000", etc.
	prompt, err1 := strconv.ParseFloat(model.Pricing.Prompt, 64)
	completion, err2 := strconv.ParseFloat(model.Pricing.Completion, 64)
	if err1 != nil || err2 != nil {
		return false
	}
	return prompt == 0 && completion == 0
}

// ChatCompletion sends a chat completion request with timeout tracking
func (c *Client) ChatCompletion(req *ChatRequest) (*ChatResponse, error) {
	return c.ChatCompletionWithTimeout(req, 10*time.Second)
}

func (c *Client) ProbeModel(modelID string) (*ProbeResult, error) {
	start := time.Now()
	resp, err := c.ChatCompletionWithTimeout(&ChatRequest{
		Model:       modelID,
		Messages:    []ChatMessage{{Role: "user", Content: probePrompt}},
		Temperature: 0,
		MaxTokens:   8,
	}, 12*time.Second)
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("empty probe response")
	}

	reply := strings.TrimSpace(resp.Choices[0].Message.Content)
	if !isHealthyProbeReply(reply) {
		return nil, fmt.Errorf("unexpected probe reply: %q", reply)
	}

	return &ProbeResult{
		ModelID:  modelID,
		Prompt:   probePrompt,
		Reply:    reply,
		Duration: time.Since(start),
	}, nil
}

func isHealthyProbeReply(reply string) bool {
	normalized := strings.ToLower(strings.TrimSpace(reply))
	return normalized != "" && strings.Contains(normalized, "pong")
}

// ChatCompletionWithTimeout sends a chat completion request with custom timeout
func (c *Client) ChatCompletionWithTimeout(req *ChatRequest, timeout time.Duration) (*ChatResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, "POST", baseURL+chatEndpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	c.setJSONHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		// Check if it's a timeout (context deadline or wrapped timeout)
		if ctx.Err() == context.DeadlineExceeded || errors.Is(err, context.DeadlineExceeded) {
			return nil, &TimeoutError{Duration: timeout}
		}
		// Also check for url.Error with timeout
		if urlErr, ok := err.(*url.Error); ok && urlErr.Timeout() {
			return nil, &TimeoutError{Duration: timeout}
		}
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &HTTPError{
			Code:    resp.StatusCode,
			Message: string(body),
		}
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &chatResp, nil
}

// StreamChatCompletion sends a streaming chat completion request
func (c *Client) StreamChatCompletion(req *ChatRequest) (<-chan StreamChunk, <-chan error) {
	return c.StreamChatCompletionWithContext(context.Background(), req)
}

// StreamChatCompletionWithContext sends a streaming chat completion request with context support.
// The context controls the HTTP request lifecycle — cancelling it aborts the connection.
// Timeouts are wrapped as TimeoutError for consistent handling by callers.
func (c *Client) StreamChatCompletionWithContext(ctx context.Context, req *ChatRequest) (<-chan StreamChunk, <-chan error) {
	chunkChan := make(chan StreamChunk, 10)
	errChan := make(chan error, 1)

	go func() {
		defer close(chunkChan)
		defer close(errChan)

		req.Stream = true
		body, err := json.Marshal(req)
		if err != nil {
			errChan <- fmt.Errorf("failed to marshal request: %w", err)
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", baseURL+chatEndpoint, bytes.NewReader(body))
		if err != nil {
			errChan <- fmt.Errorf("failed to create request: %w", err)
			return
		}

		c.setJSONHeaders(httpReq)

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			if ctx.Err() == context.DeadlineExceeded || errors.Is(err, context.DeadlineExceeded) {
				errChan <- &TimeoutError{Duration: 0}
				return
			}
			if urlErr, ok := err.(*url.Error); ok && urlErr.Timeout() {
				errChan <- &TimeoutError{Duration: 0}
				return
			}
			errChan <- fmt.Errorf("failed to send request: %w", err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			errChan <- &HTTPError{
				Code:    resp.StatusCode,
				Message: string(body),
			}
			return
		}

		// Handle SSE stream
		decoder := json.NewDecoder(resp.Body)
		for {
			var chunk StreamChunk
			if err := decoder.Decode(&chunk); err != nil {
				if err == io.EOF {
					break
				}
				if ctx.Err() != nil {
					errChan <- &TimeoutError{Duration: 0}
					return
				}
				errChan <- fmt.Errorf("failed to decode chunk: %w", err)
				return
			}
			select {
			case chunkChan <- chunk:
			case <-ctx.Done():
				return
			}
		}
	}()

	return chunkChan, errChan
}

// InvalidateCache invalidates the model cache
func (c *Client) InvalidateCache() {
	c.cacheMutex.Lock()
	c.cache = nil
	c.cacheMutex.Unlock()
}
