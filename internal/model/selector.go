package model

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	"github.com/mosajjal/frugalai/internal/config"
	"github.com/mosajjal/frugalai/internal/openrouter"
)

// Selector selects the best model based on configuration
type Selector struct {
	client *openrouter.Client
	config *config.Config
	mu     sync.RWMutex
}

// NewSelector creates a new model selector
func NewSelector(client *openrouter.Client, cfg *config.Config) *Selector {
	return &Selector{
		client: client,
		config: cfg,
	}
}

// SelectBest selects the best free model based on configuration
func (s *Selector) SelectBest() (*openrouter.Model, error) {
	models, err := s.client.GetFreeModels()
	if err != nil {
		return nil, fmt.Errorf("failed to get free models: %w", err)
	}

	if len(models) == 0 {
		return nil, fmt.Errorf("no free models available")
	}

	// Filter by constraints
	filtered := s.filterModels(models)
	if len(filtered) == 0 {
		return nil, fmt.Errorf("no models match the constraints")
	}

	// Score models
	scored := s.scoreModels(filtered)

	// Sort by score (descending)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	return &scored[0].Model, nil
}

// filterModels filters models based on configuration constraints
func (s *Selector) filterModels(models []openrouter.Model) []openrouter.Model {
	filtered := []openrouter.Model{}

	for _, model := range models {
		// Check minimum parameter count
		if s.config.MinParams > 0 && model.Params < s.config.MinParams {
			continue
		}

		// Check minimum popularity
		if s.config.MinPopularity > 0 && model.Popularity < s.config.MinPopularity {
			continue
		}

		filtered = append(filtered, model)
	}

	return filtered
}

// scoreModels scores models based on various factors
func (s *Selector) scoreModels(models []openrouter.Model) []openrouter.ModelScore {
	scored := make([]openrouter.ModelScore, len(models))

	for i, model := range models {
		scored[i] = openrouter.ModelScore{
			Model: model,
			Score: s.calculateScore(model),
		}
	}

	return scored
}

// calculateScore calculates a score for a single model
func (s *Selector) calculateScore(model openrouter.Model) float64 {
	score := 0.0

	// Popularity score (normalized to 0-1, weight: 0.3)
	popularityScore := s.normalizePopularity(model.Popularity)
	score += popularityScore * 0.3

	// Parameter count score (normalized to 0-1, weight: 0.4)
	paramScore := s.normalizeParams(model.Params)
	score += paramScore * 0.4

	// Context length score (normalized to 0-1, weight: 0.2)
	contextScore := s.normalizeContextLength(model.ContextLength)
	score += contextScore * 0.2

	// Preferred architecture bonus (weight: 0.1)
	if s.isPreferredArchitecture(model.Architecture.Modality, model.Architecture.Tokenizer) {
		score += 0.1
	}

	// Quality bonus based on known good model names
	score += s.getModelQualityBonus(model.Name, model.ID)

	return score
}

// normalizePopularity normalizes popularity to 0-1 range
func (s *Selector) normalizePopularity(popularity int) float64 {
	if popularity <= 0 {
		return 0.1
	}
	// Logarithmic scale: log(1) = 0, log(1000000) ≈ 13.8
	normalized := math.Log(float64(popularity)) / math.Log(1000000)
	return math.Min(normalized, 1.0)
}

// normalizeParams normalizes parameter count to 0-1 range
func (s *Selector) normalizeParams(params int) float64 {
	if params <= 0 {
		return 0.1
	}
	// Linear scale: 0 = 0, 70B+ = 1
	normalized := float64(params) / 70_000_000_000
	return math.Min(normalized, 1.0)
}

// normalizeContextLength normalizes context length to 0-1 range
func (s *Selector) normalizeContextLength(length int) float64 {
	if length <= 0 {
		return 0.1
	}
	// Linear scale: 0 = 0, 200k+ = 1
	normalized := float64(length) / 200_000
	return math.Min(normalized, 1.0)
}

// isPreferredArchitecture checks if the model architecture is preferred
func (s *Selector) isPreferredArchitecture(modality, tokenizer string) bool {
	if len(s.config.PreferredArchitectures) == 0 {
		return false
	}

	// Check modality and tokenizer against preferred list
	combined := strings.ToLower(modality) + " " + strings.ToLower(tokenizer)
	for _, preferred := range s.config.PreferredArchitectures {
		if strings.Contains(combined, strings.ToLower(preferred)) {
			return true
		}
	}
	return false
}

// getModelQualityBonus adds a bonus for known high-quality models
func (s *Selector) getModelQualityBonus(name, id string) float64 {
	bonus := 0.0

	nameLower := strings.ToLower(name)
	idLower := strings.ToLower(id)

	// Known high-quality model families
	qualityIndicators := []struct {
		patterns []string
		bonus    float64
	}{
		{[]string{"claude", "anthropic"}, 0.15},
		{[]string{"gpt-", "openai"}, 0.12},
		{[]string{"gemini", "google"}, 0.10},
		{[]string{"mistral", "mixtral"}, 0.08},
		{[]string{"llama", "meta"}, 0.08},
		{[]string{"qwen"}, 0.07},
		{[]string{"deepseek"}, 0.07},
		{[]string{"command", "cohere"}, 0.06},
		{[]string{"xiaomi", "mimo"}, 0.08},
		{[]string{"kwaipilot", "kat-coder"}, 0.08},
		{[]string{"nvidia", "nemotron"}, 0.07},
		{[]string{"olmo", "allenai"}, 0.06},
		{[]string{"trinity", "arcee"}, 0.06},
	}

	for _, indicator := range qualityIndicators {
		for _, pattern := range indicator.patterns {
			if strings.Contains(idLower, pattern) || strings.Contains(nameLower, pattern) {
				bonus += indicator.bonus
				break
			}
		}
	}

	// Bonus for "flash" or "pro" models (usually newer/better variants)
	if strings.Contains(idLower, "flash") || strings.Contains(nameLower, "flash") {
		bonus += 0.03
	}
	if strings.Contains(idLower, "pro") || strings.Contains(nameLower, "pro") {
		bonus += 0.02
	}

	// Penalize very old or tiny models
	weakIndicators := []string{"tiny", "mini", "nano", "micro"}
	for _, indicator := range weakIndicators {
		if strings.Contains(idLower, indicator) {
			bonus -= 0.05
		}
	}

	return bonus
}

// SelectModelByID selects a specific model by ID
func (s *Selector) SelectModelByID(id string) (*openrouter.Model, error) {
	models, err := s.client.GetModels()
	if err != nil {
		return nil, fmt.Errorf("failed to get models: %w", err)
	}

	for _, model := range models {
		if model.ID == id {
			return &model, nil
		}
	}

	return nil, fmt.Errorf("model not found: %s", id)
}

// GetBestModelID returns the ID of the best model
func (s *Selector) GetBestModelID() (string, error) {
	model, err := s.SelectBest()
	if err != nil {
		return "", err
	}
	return model.ID, nil
}

// GetTopCandidates returns the top N candidates, sorted by score
func (s *Selector) GetTopCandidates(n int) ([]openrouter.Model, error) {
	models, err := s.client.GetFreeModels()
	if err != nil {
		return nil, fmt.Errorf("failed to get free models: %w", err)
	}

	if len(models) == 0 {
		return nil, fmt.Errorf("no free models available")
	}

	// Filter by constraints
	filtered := s.filterModels(models)
	if len(filtered) == 0 {
		return nil, fmt.Errorf("no models match the constraints")
	}

	// Score models
	scored := s.scoreModels(filtered)

	// Sort by score (descending)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	// Return top N
	result := []openrouter.Model{}
	for i := 0; i < n && i < len(scored); i++ {
		result = append(result, scored[i].Model)
	}

	return result, nil
}

// GetCandidateByIndex gets a candidate by its index (0-based) from the top candidates
func (s *Selector) GetCandidateByIndex(n, idx int) (*openrouter.Model, error) {
	candidates, err := s.GetTopCandidates(n)
	if err != nil {
		return nil, err
	}

	if idx < 0 || idx >= len(candidates) {
		return nil, fmt.Errorf("index %d out of range (0-%d)", idx, len(candidates)-1)
	}

	return &candidates[idx], nil
}
