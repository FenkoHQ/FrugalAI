package openrouter

import (
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
