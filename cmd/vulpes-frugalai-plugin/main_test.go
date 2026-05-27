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
	decoded := openAIChatResponse{ID: "chatcmpl_1", Created: 123, Model: "qwen/free"}
	decoded.Choices = append(decoded.Choices, struct {
		Index   int `json:"index"`
		Message struct {
			Role      string           `json:"role"`
			Content   string           `json:"content"`
			ToolCalls []openAIToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	}{Index: 0, FinishReason: "stop"})
	decoded.Choices[0].Message.Role = "assistant"
	decoded.Choices[0].Message.Content = "ok"
	decoded.Usage.PromptTokens = 2
	decoded.Usage.CompletionTokens = 3
	decoded.Usage.TotalTokens = 5

	out := make(chan sdk.ResponseChunk, 8)
	responseChunks(sdk.InvokeRequest{}, "qwen/free", decoded, out)
	close(out)
	var chunks []sdk.ResponseChunk
	for c := range out {
		chunks = append(chunks, c)
	}
	if len(chunks) != 2 {
		t.Fatalf("len = %d, want 2", len(chunks))
	}
	if chunks[1].Usage == nil || chunks[1].Usage.ProviderModel != "qwen/free" {
		t.Fatalf("usage = %#v", chunks[1].Usage)
	}
}
