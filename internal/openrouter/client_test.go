package openrouter

import (
	"bytes"
	"io"
	"net/http"
	"testing"
)

func TestSetCommonHeadersUsesGenericUserAgent(t *testing.T) {
	client := NewClient("test-key", 300)
	req, err := http.NewRequest(http.MethodGet, "https://example.com", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}

	client.setCommonHeaders(req)

	if got := req.Header.Get("User-Agent"); got != userAgent {
		t.Fatalf("expected user agent %q, got %q", userAgent, got)
	}

	if got := req.Header.Get("Authorization"); got != "Bearer test-key" {
		t.Fatalf("expected bearer auth header, got %q", got)
	}

	if got := req.Header.Get("HTTP-Referer"); got != "" {
		t.Fatalf("expected no referer header, got %q", got)
	}
}

func TestSetCommonHeadersSkipsEmptyAuthorization(t *testing.T) {
	client := NewClient("", 300)
	req, err := http.NewRequest(http.MethodGet, "https://example.com", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}

	client.setCommonHeaders(req)

	if got := req.Header.Get("Authorization"); got != "" {
		t.Fatalf("expected empty authorization header, got %q", got)
	}
}

func TestInferParamCount(t *testing.T) {
	tests := []struct {
		name string
		text string
		want int
		ok   bool
	}{
		{name: "id 120b", text: "nvidia/nemotron-3-super-120b-a12b:free", want: 120_000_000_000, ok: true},
		{name: "name 26b", text: "Google: Gemma 4 26B A4B (free)", want: 26_000_000_000, ok: true},
		{name: "description 1.5t", text: "A 1.5T parameter model for testing", want: 1_500_000_000_000, ok: true},
		{name: "no params", text: "model-without-size", want: 0, ok: false},
	}

	for _, tc := range tests {
		got, ok := inferParamCount(tc.text)
		if ok != tc.ok {
			t.Fatalf("%s: inferParamCount(%q) ok=%v, want %v", tc.name, tc.text, ok, tc.ok)
		}
		if got != tc.want {
			t.Fatalf("%s: inferParamCount(%q)=%d, want %d", tc.name, tc.text, got, tc.want)
		}
	}
}

func TestGetModelsInfersMissingParams(t *testing.T) {
	client := NewClient("", 300)
	client.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(http.StatusOK, `{"data":[{"id":"nvidia/nemotron-3-super-120b-a12b:free","name":"NVIDIA: Nemotron 3 Super (free)","description":"NVIDIA Nemotron 3 Super is a 120B-parameter open hybrid MoE model.","pricing":{"prompt":"0","completion":"0"},"architecture":{"modality":"text->text","input_modalities":["text"],"output_modalities":["text"],"tokenizer":"Other"},"context_length":262144},{"id":"google/gemma-4-26b-a4b-it:free","name":"Google: Gemma 4 26B A4B (free)","pricing":{"prompt":"0","completion":"0"},"architecture":{"modality":"text->text","input_modalities":["text"],"output_modalities":["text"],"tokenizer":"Other"},"context_length":262144}]}`), nil
		}),
	}

	models, err := client.GetModels()
	if err != nil {
		t.Fatalf("GetModels returned error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].Params != 120_000_000_000 {
		t.Fatalf("expected super params to be inferred, got %d", models[0].Params)
	}
	if models[1].Params != 26_000_000_000 {
		t.Fatalf("expected gemma params to be inferred, got %d", models[1].Params)
	}
}

func TestProbeModelReturnsSuccessOnAnyNonErrorReply(t *testing.T) {
	client := NewClient("test-key", 300)
	client.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.Header.Get("Authorization") != "Bearer test-key" {
				t.Fatalf("expected auth header, got %q", req.Header.Get("Authorization"))
			}
			if req.Header.Get("User-Agent") != userAgent {
				t.Fatalf("expected user agent %q, got %q", userAgent, req.Header.Get("User-Agent"))
			}

			body, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatalf("failed to read request body: %v", err)
			}
			if !bytes.Contains(body, []byte(`"model":"qwen/test:free"`)) {
				t.Fatalf("expected probe to target requested model, got %s", string(body))
			}
			if !bytes.Contains(body, []byte(`"content":"ping"`)) {
				t.Fatalf("expected ping probe body, got %s", string(body))
			}

			return jsonResponse(http.StatusOK, `{"id":"1","object":"chat.completion","created":1,"model":"qwen/test:free","choices":[{"index":0,"message":{"role":"assistant","content":"hello there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`), nil
		}),
	}

	probe, err := client.ProbeModel("qwen/test:free")
	if err != nil {
		t.Fatalf("ProbeModel returned error: %v", err)
	}
	if probe.ModelID != "qwen/test:free" {
		t.Fatalf("expected model id to round-trip, got %s", probe.ModelID)
	}
	if probe.Reply != "hello there" {
		t.Fatalf("expected assistant reply to round-trip, got %q", probe.Reply)
	}
}

func TestProbeModelRejectsEmptyChoices(t *testing.T) {
	client := NewClient("test-key", 300)
	client.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(http.StatusOK, `{"id":"1","object":"chat.completion","created":1,"model":"qwen/test:free","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}`), nil
		}),
	}

	if _, err := client.ProbeModel("qwen/test:free"); err == nil {
		t.Fatal("expected ProbeModel to fail on empty choices")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func jsonResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(bytes.NewBufferString(body)),
	}
}
