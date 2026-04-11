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

func TestIsHealthyProbeReply(t *testing.T) {
	tests := []struct {
		reply string
		want  bool
	}{
		{reply: "pong", want: true},
		{reply: "Pong!", want: true},
		{reply: "\"pong\"", want: true},
		{reply: "ping", want: false},
		{reply: "", want: false},
	}

	for _, tc := range tests {
		if got := isHealthyProbeReply(tc.reply); got != tc.want {
			t.Fatalf("isHealthyProbeReply(%q) = %v, want %v", tc.reply, got, tc.want)
		}
	}
}

func TestProbeModelReturnsSuccessOnHealthyPong(t *testing.T) {
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

			return jsonResponse(http.StatusOK, `{"id":"1","object":"chat.completion","created":1,"model":"qwen/test:free","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`), nil
		}),
	}

	probe, err := client.ProbeModel("qwen/test:free")
	if err != nil {
		t.Fatalf("ProbeModel returned error: %v", err)
	}
	if probe.ModelID != "qwen/test:free" {
		t.Fatalf("expected model id to round-trip, got %s", probe.ModelID)
	}
	if probe.Reply != "pong" {
		t.Fatalf("expected pong reply, got %q", probe.Reply)
	}
}

func TestProbeModelRejectsUnexpectedReply(t *testing.T) {
	client := NewClient("test-key", 300)
	client.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(http.StatusOK, `{"id":"1","object":"chat.completion","created":1,"model":"qwen/test:free","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`), nil
		}),
	}

	if _, err := client.ProbeModel("qwen/test:free"); err == nil {
		t.Fatal("expected ProbeModel to fail on non-pong reply")
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
