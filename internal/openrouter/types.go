package openrouter

import "time"

// Architecture represents model architecture information
type Architecture struct {
	Modality          string   `json:"modality"`
	InputModalities   []string `json:"input_modalities"`
	OutputModalities  []string `json:"output_modalities"`
	Tokenizer         string   `json:"tokenizer"`
	InstructType      *string  `json:"instruct_type"`
}

// Model represents an OpenRouter model
type Model struct {
	ID            string       `json:"id"`
	Name          string       `json:"name"`
	Description   string       `json:"description"`
	Pricing       Pricing      `json:"pricing"`
	Architecture  Architecture `json:"architecture"`
	ContextLength int          `json:"context_length"`
	Popularity    int          `json:"popularity,omitempty"`
	Params        int          `json:"params,omitempty"`
}

// Pricing represents model pricing
type Pricing struct {
	Prompt     string `json:"prompt"`
	Completion string `json:"completion"`
}

// ModelsResponse is the response from the models endpoint
type ModelsResponse struct {
	Data []Model `json:"data"`
}

// ChatMessage represents a chat message
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents a chat completion request
type ChatRequest struct {
	Model            string          `json:"model"`
	Messages         []ChatMessage   `json:"messages"`
	Temperature      float64         `json:"temperature,omitempty"`
	MaxTokens        int             `json:"max_tokens,omitempty"`
	TopP             float64         `json:"top_p,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	FrequencyPenalty float64         `json:"frequency_penalty,omitempty"`
	PresencePenalty  float64         `json:"presence_penalty,omitempty"`
	Seed             *int            `json:"seed,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
}

// ResponseFormat specifies the format of the response
type ResponseFormat struct {
	Type string `json:"type"`
}

// ChatChoice represents a choice in the chat response
type ChatChoice struct {
	Index        int          `json:"index"`
	Message      ChatMessage  `json:"message"`
	FinishReason string       `json:"finish_reason"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatResponse represents a chat completion response
type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   Usage        `json:"usage"`
}

// StreamChunk represents a streaming chunk
type StreamChunk struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
}

// AnthropicMessage represents an Anthropic-style message
type AnthropicMessage struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

// ContentBlock represents a content block in Anthropic format
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// AnthropicRequest represents an Anthropic-style request
type AnthropicRequest struct {
	Model     string              `json:"model"`
	MaxTokens int                 `json:"max_tokens"`
	Messages  []AnthropicMessage  `json:"messages"`
	System    string              `json:"system,omitempty"`
	Temperature float64           `json:"temperature,omitempty"`
	TopP     float64              `json:"top_p,omitempty"`
	Stream   bool                 `json:"stream,omitempty"`
}

// AnthropicResponse represents an Anthropic-style response
type AnthropicResponse struct {
	ID      string           `json:"id"`
	Type    string           `json:"type"`
	Role    string           `json:"role"`
	Content []ContentBlock   `json:"content"`
	StopReason string        `json:"stop_reason"`
	Model      string        `json:"model"`
	Usage      AnthropicUsage `json:"usage"`
}

// AnthropicUsage represents Anthropic token usage
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ModelScore represents a model with its score
type ModelScore struct {
	Model Model
	Score float64
}

// CachedModels holds cached model data
type CachedModels struct {
	Models    []Model
	Timestamp time.Time
}

// ModelManager manages model selection and failover
type ModelManager struct {
	Candidates  []Model
	Current     *Model
	CurrentIdx  int
	Failures    map[string]int
	LastFailure map[string]time.Time
	Timeouts    map[string]int
	Burned      map[string]bool
}

// APIError represents an error response from the API
type APIError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Type    string `json:"type"`
}

// HealthStatus represents the health status of the service
type HealthStatus struct {
	Status     string  `json:"status"`
	Model      string  `json:"model,omitempty"`
	ModelName  string  `json:"model_name,omitempty"`
	Candidates int     `json:"candidates"`
	Uptime     float64 `json:"uptime_seconds"`
}
