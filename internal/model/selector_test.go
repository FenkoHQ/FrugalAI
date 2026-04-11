package model

import (
	"testing"

	"github.com/mosajjal/frugalai/internal/config"
	"github.com/mosajjal/frugalai/internal/openrouter"
)

func TestRankCandidatesAppendsRouterFallback(t *testing.T) {
	selector := &Selector{config: config.LoadFromEnv()}
	models := []openrouter.Model{
		{
			ID:            "google/gemma-3-27b-it:free",
			Name:          "Google: Gemma 3 27B (free)",
			Pricing:       openrouter.Pricing{Prompt: "0", Completion: "0"},
			ContextLength: 131072,
		},
		{
			ID:            "qwen/qwen3-next-80b-a3b-instruct:free",
			Name:          "Qwen: Qwen3 Next 80B A3B Instruct (free)",
			Pricing:       openrouter.Pricing{Prompt: "0", Completion: "0"},
			ContextLength: 262144,
		},
		{
			ID:            fallbackRouterID,
			Name:          "Free Models Router",
			Pricing:       openrouter.Pricing{Prompt: "0", Completion: "0"},
			ContextLength: 200000,
		},
	}

	candidates, err := selector.rankCandidates(models, 2)
	if err != nil {
		t.Fatalf("rankCandidates returned error: %v", err)
	}

	if len(candidates) != 3 {
		t.Fatalf("expected 3 candidates including router fallback, got %d", len(candidates))
	}

	if candidates[0].ID != "qwen/qwen3-next-80b-a3b-instruct:free" {
		t.Fatalf("expected top weekly qwen model first, got %s", candidates[0].ID)
	}

	if candidates[len(candidates)-1].ID != fallbackRouterID {
		t.Fatalf("expected router fallback last, got %s", candidates[len(candidates)-1].ID)
	}
}

func TestRankCandidatesFallsBackToRouterWhenNoDedicatedFreeModels(t *testing.T) {
	selector := &Selector{config: config.LoadFromEnv()}
	models := []openrouter.Model{
		{
			ID:            "qwen/qwen3.6-plus",
			Name:          "Qwen3.6 Plus",
			Pricing:       openrouter.Pricing{Prompt: "0.000000325", Completion: "0.00000195"},
			ContextLength: 1000000,
		},
		{
			ID:            fallbackRouterID,
			Name:          "Free Models Router",
			Pricing:       openrouter.Pricing{Prompt: "0", Completion: "0"},
			ContextLength: 200000,
		},
	}

	candidates, err := selector.rankCandidates(models, 5)
	if err != nil {
		t.Fatalf("rankCandidates returned error: %v", err)
	}

	if len(candidates) != 1 || candidates[0].ID != fallbackRouterID {
		t.Fatalf("expected only router fallback, got %+v", candidates)
	}
}

func TestRankCandidatesErrorsWithoutFreeFallback(t *testing.T) {
	selector := &Selector{config: config.LoadFromEnv()}
	models := []openrouter.Model{
		{
			ID:            "qwen/qwen3.6-plus",
			Name:          "Qwen3.6 Plus",
			Pricing:       openrouter.Pricing{Prompt: "0.000000325", Completion: "0.00000195"},
			ContextLength: 1000000,
		},
	}

	if _, err := selector.rankCandidates(models, 5); err == nil {
		t.Fatal("expected an error when neither free models nor router fallback exist")
	}
}

func TestCandidateOrderPrefersRequestedModelFirst(t *testing.T) {
	candidates := []openrouter.Model{
		{ID: "a"},
		{ID: "b"},
		{ID: "c"},
	}

	order := candidateOrder(candidates, "b")
	want := []int{1, 0, 2}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("candidateOrder()[%d] = %d, want %d", i, order[i], want[i])
		}
	}
}
