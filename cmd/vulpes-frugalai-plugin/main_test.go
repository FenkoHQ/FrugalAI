package main

import (
	"testing"

	"github.com/FenkoHQ/vulpes-core-plugins/sdk"
)

func TestConvertMessages(t *testing.T) {
	got := convertMessages([]sdk.ChatMessage{{Role: "user", Content: "hello"}})
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	if got[0]["role"] != "user" || got[0]["content"] != "hello" {
		t.Fatalf("message = %#v", got[0])
	}
}

func TestRotate(t *testing.T) {
	got := rotate([]int{1, 2, 3, 4}, 2)
	want := []int{3, 4, 1, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rotate = %#v, want %#v", got, want)
		}
	}
}

func TestIsConcreteModel(t *testing.T) {
	for _, model := range []string{"", "auto", "frugal", "frugalai"} {
		if isConcreteModel(model) {
			t.Fatalf("%q should not be concrete", model)
		}
	}
	if !isConcreteModel("qwen/qwen3-next-80b-a3b-instruct:free") {
		t.Fatal("real model should be concrete")
	}
}

func TestResponseChunksUseActualProviderModel(t *testing.T) {
	out := openAIChatResponse{ID: "chatcmpl_1", Created: 123, Model: "qwen/free"}
	out.Choices = append(out.Choices, struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	}{Index: 0, FinishReason: "stop"})
	out.Choices[0].Message.Role = "assistant"
	out.Choices[0].Message.Content = "ok"
	out.Usage.PromptTokens = 2
	out.Usage.CompletionTokens = 3
	out.Usage.TotalTokens = 5

	chunks := responseChunks(sdk.InvokeRequest{}, "qwen/free", out)
	if len(chunks) != 2 {
		t.Fatalf("len = %d, want 2", len(chunks))
	}
	if chunks[1].Usage == nil || chunks[1].Usage.ProviderModel != "qwen/free" {
		t.Fatalf("usage = %#v", chunks[1].Usage)
	}
}
