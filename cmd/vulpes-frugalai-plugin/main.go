package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/FenkoHQ/vulpes-core-plugins/sdk"
	"github.com/mosajjal/frugalai/internal/config"
	"github.com/mosajjal/frugalai/internal/model"
	"github.com/mosajjal/frugalai/internal/openrouter"
)

const (
	defaultBaseURL        = "https://openrouter.ai/api/v1"
	defaultMaxRetries     = 3
	defaultRequestTimeout = 120 * time.Second
)

type plugin struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client

	selector *model.Selector
	client   *openrouter.Client
	cfg      *config.Config

	maxRetries   int
	probeOnStart bool

	mu      sync.RWMutex
	manager *openrouter.ModelManager
}

func (p *plugin) Configure(ctx context.Context, cfg map[string]any, secrets map[string]string) error {
	apiKey := firstString(stringValue(cfg["api_key"]), secrets["OPENROUTER_API_KEY"], secrets["FRUGALAI_API_KEY"])
	if apiKey == "" {
		return fmt.Errorf("api_key is required")
	}

	frugalCfg := config.LoadFromEnv()
	frugalCfg.APIKey = apiKey
	frugalCfg.MinParams = intValue(cfg["min_params"], frugalCfg.MinParams)
	frugalCfg.MinPopularity = intValue(cfg["min_popularity"], frugalCfg.MinPopularity)
	frugalCfg.CacheTTL = intValue(cfg["cache_ttl_seconds"], frugalCfg.CacheTTL)
	frugalCfg.PreferredArchitectures = stringSliceValue(cfg["preferred_architectures"], frugalCfg.PreferredArchitectures)
	frugalCfg.TopWeeklyModels = stringSliceValue(cfg["top_weekly_models"], frugalCfg.TopWeeklyModels)
	frugalCfg.ModelIndex = intValue(cfg["model_index"], frugalCfg.ModelIndex)
	frugalCfg.NumCandidates = intValue(cfg["num_candidates"], frugalCfg.NumCandidates)
	if frugalCfg.NumCandidates <= 0 {
		frugalCfg.NumCandidates = 10
	}

	p.apiKey = apiKey
	p.baseURL = strings.TrimRight(firstString(stringValue(cfg["base_url"]), defaultBaseURL), "/")
	p.maxRetries = intValue(cfg["max_retries"], defaultMaxRetries)
	if p.maxRetries <= 0 {
		p.maxRetries = defaultMaxRetries
	}
	p.probeOnStart = boolValue(cfg["probe_on_start"], true)

	timeout := durationSeconds(cfg["timeout_seconds"], defaultRequestTimeout)
	p.httpClient = &http.Client{Timeout: timeout}
	p.client = openrouter.NewClient(apiKey, frugalCfg.CacheTTL)
	p.selector = model.NewSelector(p.client, frugalCfg)
	p.cfg = frugalCfg

	if p.probeOnStart {
		if err := p.refresh(ctx, ""); err != nil {
			log.Printf("[WARN] frugalai initial model selection failed: %v", err)
			p.setEmptyManager(nil)
		}
		return nil
	}

	candidates, err := p.selector.GetTopCandidates(frugalCfg.NumCandidates)
	if err != nil {
		log.Printf("[WARN] frugalai candidate fetch failed: %v", err)
		p.setEmptyManager(nil)
		return nil
	}
	p.setEmptyManager(candidates)
	return nil
}

// Invoke streams chunks for an upstream call. Retries are only safe before
// any chunk has been emitted to the gateway — invokeModel signals that by
// returning a non-nil error iff it has not written anything to out. Once a
// chunk crosses the channel boundary the caller has committed; any later
// upstream error is delivered as an error chunk and Invoke returns nil.
func (p *plugin) Invoke(ctx context.Context, req sdk.InvokeRequest, out chan<- sdk.ResponseChunk) error {
	if err := p.ensureManager(ctx); err != nil {
		out <- sdk.ResponseChunk{Error: upstreamErr("no_model_available", err.Error(), http.StatusServiceUnavailable, true)}
		return nil
	}

	var lastErr error
	for attempt := 0; attempt < p.maxRetries; attempt++ {
		modelID := p.modelForRequest(req)
		if modelID == "" {
			out <- sdk.ResponseChunk{Error: upstreamErr("no_model_available", "no OpenRouter free model selected", http.StatusServiceUnavailable, true)}
			return nil
		}

		err := p.invokeModel(ctx, req, modelID, out)
		if err == nil {
			return nil
		}
		lastErr = err
		if !isRetryable(err) || attempt == p.maxRetries-1 {
			break
		}
		p.markFailure(modelID)
		if switchErr := p.switchModel(ctx, modelID); switchErr != nil {
			lastErr = fmt.Errorf("%w; failed to switch model: %v", err, switchErr)
			break
		}
	}

	status := statusCode(lastErr)
	out <- sdk.ResponseChunk{Error: upstreamErr(upstreamCode(status), lastErr.Error(), status, isRetryable(lastErr))}
	return nil
}

func (p *plugin) ListModels(ctx context.Context) ([]sdk.ModelInfo, error) {
	if p.selector == nil {
		return nil, nil
	}
	candidates, err := p.selector.GetTopCandidates(p.cfg.NumCandidates)
	if err != nil {
		return nil, err
	}
	currentID := p.currentModelID()
	models := make([]sdk.ModelInfo, 0, len(candidates)+1)
	models = append(models, sdk.ModelInfo{ID: "auto", Object: "model", OwnedBy: "frugalai", ProviderInstance: "frugalai", ProviderModel: "auto", SupportsStreaming: true, SupportsJSONMode: true, Healthy: currentID != "", Properties: map[string]string{"current_model": currentID}})
	for _, m := range candidates {
		models = append(models, sdk.ModelInfo{ID: m.ID, Object: "model", OwnedBy: "openrouter", ProviderInstance: "frugalai", ProviderModel: m.ID, ContextWindow: int64(m.ContextLength), SupportsStreaming: true, SupportsJSONMode: true, Healthy: true, Properties: modelProperties(m, currentID == m.ID)})
	}
	return models, nil
}

// invokeModel returns a non-nil error only if no chunks were written to out,
// so the caller can safely retry against the next model. After the first
// chunk crosses out it has committed; any later failure surfaces as an error
// chunk and invokeModel returns nil.
func (p *plugin) invokeModel(ctx context.Context, req sdk.InvokeRequest, modelID string, out chan<- sdk.ResponseChunk) error {
	body := p.chatRequest(req, modelID)
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return err
	}
	p.addHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return &temporaryError{err: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		msg := readLimit(resp.Body, 4096)
		return &httpError{status: resp.StatusCode, message: msg}
	}
	if req.Request.Stream {
		return p.streamChunks(ctx, resp.Body, req, modelID, out)
	}

	var decoded openAIChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return &temporaryError{err: fmt.Errorf("decode upstream response: %w", err)}
	}
	responseChunks(req, modelID, decoded, out)
	return nil
}

func (p *plugin) chatRequest(req sdk.InvokeRequest, modelID string) map[string]any {
	body := map[string]any{
		"model":    modelID,
		"messages": convertMessages(req.Request.Messages),
		"stream":   req.Request.Stream,
	}
	if req.Request.Temperature != nil {
		body["temperature"] = *req.Request.Temperature
	}
	if req.Request.MaxTokens != nil {
		body["max_tokens"] = *req.Request.MaxTokens
	}
	if req.Request.Tools != nil {
		body["tools"] = req.Request.Tools
	}
	if req.Request.ToolChoice != nil {
		body["tool_choice"] = req.Request.ToolChoice
	}
	if req.Request.Metadata != nil {
		body["metadata"] = req.Request.Metadata
	}
	for k, v := range req.Request.Extra {
		if _, ok := body[k]; !ok {
			body[k] = v
		}
	}
	return body
}

func (p *plugin) streamChunks(ctx context.Context, r io.Reader, req sdk.InvokeRequest, modelID string, out chan<- sdk.ResponseChunk) error {
	send := func(c sdk.ResponseChunk) bool {
		select {
		case out <- c:
			return true
		case <-ctx.Done():
			return false
		}
	}
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			if data == "[DONE]" {
				break
			}
			continue
		}
		var ev openAIStreamChunk
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		for _, choice := range ev.Choices {
			toolCalls := toSDKToolCalls(choice.Delta.ToolCalls)
			content := choice.Delta.Content
			// Skip reasoning fallback when tool_calls are streaming through —
			// see the upstream-openai plugin for the same trap.
			if content == "" && len(toolCalls) == 0 {
				content = choice.Delta.ReasoningContent
			}
			if !send(sdk.ResponseChunk{Chunk: &sdk.ChatCompletionChunk{ID: ev.ID, Object: "chat.completion.chunk", Created: ev.Created, Model: firstString(ev.Model, modelID), Choices: []sdk.ChatChoice{{Index: choice.Index, Delta: sdk.ChatMessage{Role: firstString(choice.Delta.Role, "assistant"), Content: content, ToolCalls: toolCalls}, FinishReason: choice.FinishReason}}}}) {
				return nil
			}
		}
		if ev.Usage.TotalTokens > 0 {
			usage := sdk.Usage{ProviderInstance: req.Context.PluginInstance, ProviderModel: modelID, InputTokens: ev.Usage.PromptTokens, OutputTokens: ev.Usage.CompletionTokens, TotalTokens: ev.Usage.TotalTokens}
			if !send(sdk.ResponseChunk{Usage: &usage}) {
				return nil
			}
		}
	}
	if err := s.Err(); err != nil {
		send(sdk.ResponseChunk{Error: upstreamErr("upstream_stream_failed", err.Error(), http.StatusBadGateway, true)})
	}
	return nil
}

func responseChunks(req sdk.InvokeRequest, modelID string, decoded openAIChatResponse, out chan<- sdk.ResponseChunk) {
	for _, choice := range decoded.Choices {
		msg := sdk.ChatMessage{Role: firstString(choice.Message.Role, "assistant"), Content: choice.Message.Content, ToolCalls: toSDKToolCalls(choice.Message.ToolCalls)}
		finish := choice.FinishReason
		if len(msg.ToolCalls) > 0 && (finish == "stop" || finish == "") {
			finish = "tool_calls"
		}
		out <- sdk.ResponseChunk{Chunk: &sdk.ChatCompletionChunk{ID: decoded.ID, Object: "chat.completion.chunk", Created: decoded.Created, Model: firstString(decoded.Model, modelID), Choices: []sdk.ChatChoice{{Index: choice.Index, Message: msg, FinishReason: finish}}}}
	}
	usage := sdk.Usage{ProviderInstance: req.Context.PluginInstance, ProviderModel: modelID, InputTokens: decoded.Usage.PromptTokens, OutputTokens: decoded.Usage.CompletionTokens, TotalTokens: decoded.Usage.TotalTokens}
	out <- sdk.ResponseChunk{Usage: &usage}
}

func (p *plugin) ensureManager(ctx context.Context) error {
	p.mu.RLock()
	ready := p.manager != nil && p.manager.Current != nil
	p.mu.RUnlock()
	if ready {
		return nil
	}
	return p.refresh(ctx, "")
}

func (p *plugin) refresh(ctx context.Context, preferredID string) error {
	candidates, err := p.selector.GetTopCandidates(p.cfg.NumCandidates)
	if err != nil {
		return err
	}
	if len(candidates) == 0 {
		return fmt.Errorf("no OpenRouter free model candidates")
	}

	selectedIdx := 0
	if preferredID == "" && p.cfg.ModelIndex >= 0 && p.cfg.ModelIndex < len(candidates) {
		selectedIdx = p.cfg.ModelIndex
		preferredID = candidates[selectedIdx].ID
	}
	if preferredID == "" {
		preferredID = candidates[selectedIdx].ID
	}

	selected, idx, probe, err := p.selector.SelectWorkingCandidate(candidates, preferredID)
	if err != nil {
		p.setEmptyManager(candidates)
		return err
	}
	_ = ctx
	log.Printf("[INFO] frugalai selected OpenRouter model %s (%s), probe=%q", selected.ID, selected.Name, probe.Reply)

	p.mu.Lock()
	p.manager = &openrouter.ModelManager{Candidates: candidates, Current: selected, CurrentIdx: idx, Failures: map[string]int{}, LastFailure: map[string]time.Time{}, Timeouts: map[string]int{}, Burned: map[string]bool{}}
	p.mu.Unlock()
	return nil
}

func (p *plugin) switchModel(ctx context.Context, failedID string) error {
	p.mu.RLock()
	if p.manager == nil || len(p.manager.Candidates) == 0 {
		p.mu.RUnlock()
		return p.refresh(ctx, "")
	}
	candidates := append([]openrouter.Model(nil), p.manager.Candidates...)
	startIdx := (p.manager.CurrentIdx + 1) % len(candidates)
	p.mu.RUnlock()

	rotated := rotate(candidates, startIdx)
	selected, _, probe, err := p.selector.SelectWorkingCandidate(rotated, "")
	if err != nil {
		return err
	}
	selectedIdx := findCandidateIndex(candidates, selected.ID)
	if selectedIdx < 0 {
		return fmt.Errorf("selected model %s is no longer a candidate", selected.ID)
	}

	p.mu.Lock()
	if p.manager == nil {
		p.manager = &openrouter.ModelManager{Failures: map[string]int{}, LastFailure: map[string]time.Time{}, Timeouts: map[string]int{}, Burned: map[string]bool{}}
	}
	p.manager.Candidates = candidates
	p.manager.Current = &p.manager.Candidates[selectedIdx]
	p.manager.CurrentIdx = selectedIdx
	p.mu.Unlock()
	log.Printf("[INFO] frugalai switched from %s to %s, probe=%q", failedID, selected.ID, probe.Reply)
	return nil
}

func (p *plugin) modelForRequest(req sdk.InvokeRequest) string {
	if forced := strings.TrimSpace(req.Properties["force_model"]); forced != "" {
		return forced
	}
	if isConcreteModel(req.ProviderModel) {
		return req.ProviderModel
	}
	return p.currentModelID()
}

func (p *plugin) currentModelID() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.manager == nil || p.manager.Current == nil {
		return ""
	}
	return p.manager.Current.ID
}

func (p *plugin) markFailure(modelID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.manager == nil {
		return
	}
	if p.manager.Failures == nil {
		p.manager.Failures = map[string]int{}
	}
	if p.manager.LastFailure == nil {
		p.manager.LastFailure = map[string]time.Time{}
	}
	p.manager.Failures[modelID]++
	p.manager.LastFailure[modelID] = time.Now()
}

func (p *plugin) setEmptyManager(candidates []openrouter.Model) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.manager = &openrouter.ModelManager{Candidates: candidates, Current: nil, CurrentIdx: 0, Failures: map[string]int{}, LastFailure: map[string]time.Time{}, Timeouts: map[string]int{}, Burned: map[string]bool{}}
}

func (p *plugin) addHeaders(req *http.Request) {
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://fenko.dev")
	req.Header.Set("X-Title", "Vulpes FrugalAI")
}

func convertMessages(messages []sdk.ChatMessage) []map[string]any {
	out := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		role := msg.Role
		if role == "" {
			role = "user"
		}
		out = append(out, map[string]any{"role": role, "content": contentText(msg.Content)})
	}
	return out
}

func contentText(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case []any:
		parts := make([]string, 0, len(x))
		for _, item := range x {
			if m, ok := item.(map[string]any); ok {
				if s, ok := m["text"].(string); ok {
					parts = append(parts, s)
				}
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "\n")
		}
	}
	b, _ := json.Marshal(v)
	return string(b)
}

func modelProperties(m openrouter.Model, current bool) map[string]string {
	props := map[string]string{
		"name":           m.Name,
		"pricing_prompt": m.Pricing.Prompt,
		"pricing_output": m.Pricing.Completion,
		"modality":       m.Architecture.Modality,
		"tokenizer":      m.Architecture.Tokenizer,
		"params":         strconv.Itoa(m.Params),
		"popularity":     strconv.Itoa(m.Popularity),
	}
	if current {
		props["current"] = "true"
	}
	return props
}

func isConcreteModel(model string) bool {
	model = strings.TrimSpace(strings.ToLower(model))
	return model != "" && model != "auto" && model != "frugal" && model != "frugalai"
}

func rotate[T any](items []T, start int) []T {
	if len(items) == 0 {
		return nil
	}
	start = start % len(items)
	out := make([]T, 0, len(items))
	out = append(out, items[start:]...)
	out = append(out, items[:start]...)
	return out
}

func findCandidateIndex(candidates []openrouter.Model, id string) int {
	for i := range candidates {
		if candidates[i].ID == id {
			return i
		}
	}
	return -1
}

type openAIChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role      string           `json:"role"`
			Content   string           `json:"content"`
			ToolCalls []openAIToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

type openAIStreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role             string           `json:"role"`
			Content          string           `json:"content"`
			ReasoningContent string           `json:"reasoning_content"`
			ToolCalls        []openAIToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

type openAIToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func toSDKToolCalls(in []openAIToolCall) []sdk.ToolCall {
	if len(in) == 0 {
		return nil
	}
	out := make([]sdk.ToolCall, len(in))
	for i, tc := range in {
		out[i] = sdk.ToolCall{
			Index:    tc.Index,
			ID:       tc.ID,
			Type:     tc.Type,
			Function: sdk.ToolCallFunction{Name: tc.Function.Name, Arguments: tc.Function.Arguments},
		}
	}
	return out
}

type httpError struct {
	status  int
	message string
}

func (e *httpError) Error() string { return strings.TrimSpace(e.message) }

type temporaryError struct{ err error }

func (e *temporaryError) Error() string { return e.err.Error() }
func (e *temporaryError) Unwrap() error { return e.err }

func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	if _, ok := err.(*temporaryError); ok {
		return true
	}
	if e, ok := err.(*httpError); ok {
		return e.status == http.StatusTooManyRequests || e.status >= 500
	}
	return false
}

func statusCode(err error) int {
	if e, ok := err.(*httpError); ok {
		return e.status
	}
	return http.StatusBadGateway
}

func upstreamErr(code, message string, status int, retryable bool) *sdk.UpstreamError {
	return &sdk.UpstreamError{Code: code, Message: message, HTTPStatus: status, Retryable: retryable, RateLimited: status == http.StatusTooManyRequests}
}

func upstreamCode(status int) string {
	switch {
	case status == http.StatusTooManyRequests:
		return "upstream_rate_limited"
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		return "auth_error"
	case status >= 500:
		return "upstream_5xx"
	case status >= 400:
		return "invalid_request"
	default:
		return "upstream_error"
	}
}

func readLimit(r io.Reader, n int64) string {
	b, _ := io.ReadAll(io.LimitReader(r, n))
	return string(b)
}

func stringValue(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func intValue(v any, def int) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	case string:
		if i, err := strconv.Atoi(strings.TrimSpace(x)); err == nil {
			return i
		}
	}
	return def
}

func boolValue(v any, def bool) bool {
	switch x := v.(type) {
	case bool:
		return x
	case string:
		if b, err := strconv.ParseBool(strings.TrimSpace(x)); err == nil {
			return b
		}
	}
	return def
}

func durationSeconds(v any, def time.Duration) time.Duration {
	seconds := intValue(v, 0)
	if seconds <= 0 {
		return def
	}
	return time.Duration(seconds) * time.Second
}

func stringSliceValue(v any, def []string) []string {
	switch x := v.(type) {
	case []string:
		return x
	case []any:
		out := make([]string, 0, len(x))
		for _, item := range x {
			if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
				out = append(out, strings.TrimSpace(s))
			}
		}
		return out
	case string:
		if x == "" {
			return def
		}
		parts := strings.Split(x, ",")
		out := make([]string, 0, len(parts))
		for _, part := range parts {
			if s := strings.TrimSpace(part); s != "" {
				out = append(out, s)
			}
		}
		return out
	default:
		return def
	}
}

func firstString(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func main() {
	p := &plugin{baseURL: defaultBaseURL, httpClient: &http.Client{Timeout: defaultRequestTimeout}, maxRetries: defaultMaxRetries, probeOnStart: true}
	s := &sdk.Service{
		Metadata: sdk.Metadata{
			Name:         "frugalai-upstream",
			Version:      "0.1.0",
			Homepage:     "https://github.com/mosajjal/frugalai",
			Capabilities: []sdk.CapabilityDescriptor{{Type: sdk.CapabilityUpstreamProvider, Name: "frugalai-openrouter", Version: "0.1.0"}},
			Permissions: sdk.Permissions{
				OutboundHosts: []string{"openrouter.ai:443"},
				SecretNames:   []string{"OPENROUTER_API_KEY", "FRUGALAI_API_KEY"},
				Data:          sdk.DataPermissions{ReadPrompt: true, ReadResponse: true},
			},
		},
		Schema:           `{"type":"object","required":["api_key"],"properties":{"api_key":{"type":"string","secret":true},"base_url":{"type":"string","default":"https://openrouter.ai/api/v1"},"min_params":{"type":"integer","minimum":0},"min_popularity":{"type":"integer","minimum":0},"preferred_architectures":{"type":["array","string"],"items":{"type":"string"}},"top_weekly_models":{"type":["array","string"],"items":{"type":"string"}},"model_index":{"type":"integer","default":-1},"num_candidates":{"type":"integer","default":10},"cache_ttl_seconds":{"type":"integer","default":300},"probe_on_start":{"type":"boolean","default":true},"max_retries":{"type":"integer","default":3},"timeout_seconds":{"type":"integer","default":120}}}`,
		Configurer:       p,
		UpstreamProvider: p,
	}
	if err := sdk.ServeFromEnv(s); err != nil {
		panic(err)
	}
}
