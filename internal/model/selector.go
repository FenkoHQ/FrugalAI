package model

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mosajjal/frugalai/internal/config"
	"github.com/mosajjal/frugalai/internal/openrouter"
)

// Selector selects the best model based on configuration
type Selector struct {
	client *openrouter.Client
	config *config.Config
	mu     sync.RWMutex
}

const fallbackRouterID = "openrouter/free"

// NewSelector creates a new model selector
func NewSelector(client *openrouter.Client, cfg *config.Config) *Selector {
	return &Selector{
		client: client,
		config: cfg,
	}
}

// SelectBest selects the best free model based on configuration
func (s *Selector) SelectBest() (*openrouter.Model, error) {
	candidates, err := s.GetTopCandidates(1)
	if err != nil {
		return nil, err
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("no models available")
	}

	return &candidates[0], nil
}

func (s *Selector) IsModelAvailable(id string) (bool, error) {
	models, err := s.client.GetModels()
	if err != nil {
		return false, fmt.Errorf("failed to get models: %w", err)
	}

	_, found := findModelByID(models, id)
	return found, nil
}

func (s *Selector) ProbeModel(id string) (*openrouter.ProbeResult, error) {
	return s.client.ProbeModel(id)
}

func (s *Selector) SelectWorkingCandidate(candidates []openrouter.Model, preferredID string) (*openrouter.Model, int, *openrouter.ProbeResult, error) {
	if len(candidates) == 0 {
		return nil, -1, nil, fmt.Errorf("no candidates available")
	}

	order := candidateOrder(candidates, preferredID)
	failures := []string{}

	for _, idx := range order {
		probe, err := s.client.ProbeModel(candidates[idx].ID)
		if err == nil {
			return &candidates[idx], idx, probe, nil
		}
		failures = append(failures, fmt.Sprintf("%s: %v", candidates[idx].ID, err))
	}

	return nil, -1, nil, fmt.Errorf("no working model candidates: %s", strings.Join(failures, "; "))
}

func (s *Selector) rankCandidates(models []openrouter.Model, n int) ([]openrouter.Model, error) {
	freeModels := []openrouter.Model{}
	for _, model := range models {
		if isFreeModel(model) {
			freeModels = append(freeModels, model)
		}
	}

	filtered := s.filterModels(freeModels)
	if len(filtered) == 0 {
		return s.fallbackCandidates(models)
	}

	scored := s.scoreModels(filtered)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	result := []openrouter.Model{}
	for i := 0; i < n && i < len(scored); i++ {
		result = append(result, scored[i].Model)
	}

	return s.appendFallbackRouter(result, models), nil
}

func (s *Selector) fallbackCandidates(models []openrouter.Model) ([]openrouter.Model, error) {
	router, found := findModelByID(models, fallbackRouterID)
	if !found {
		return nil, fmt.Errorf("no free models available")
	}

	return []openrouter.Model{router}, nil
}

func (s *Selector) appendFallbackRouter(candidates, models []openrouter.Model) []openrouter.Model {
	router, found := findModelByID(models, fallbackRouterID)
	if !found || containsModel(candidates, router.ID) {
		return candidates
	}

	return append(candidates, router)
}

// filterModels filters models based on configuration constraints
func (s *Selector) filterModels(models []openrouter.Model) []openrouter.Model {
	filtered := []openrouter.Model{}

	for _, model := range models {
		// Skip meta-routers — these aren't real models, they randomly dispatch
		// to other free models (e.g. openrouter/free)
		if strings.HasSuffix(model.ID, "/free") && !strings.Contains(model.ID, ":free") {
			continue
		}

		// Check minimum parameter count
		if s.config.MinParams > 0 && model.Params < s.config.MinParams {
			continue
		}

		// Check minimum popularity — exempt stealth models (recently published from known quality providers)
		if s.config.MinPopularity > 0 && model.Popularity < s.config.MinPopularity {
			if !s.isStealthModel(model) {
				continue
			}
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

	// Top Weekly bonus (weight: 0.5 - highly prioritized)
	if s.isTopWeekly(model.ID) {
		score += 0.5
	}

	// Quality bonus based on known good model names
	score += s.getModelQualityBonus(model.Name, model.ID)

	// Stealth model bonus: free models with very low popularity from known providers
	// These are usually quiet launches of excellent models
	if s.isStealthModel(model) {
		score += 0.4
	}

	return score
}

// normalizePopularity normalizes popularity to 0-1 range
func (s *Selector) normalizePopularity(popularity int) float64 {
	if popularity <= 0 {
		// Unknown/new model — don't penalize, assume mid-range
		return 0.5
	}
	// Logarithmic scale: log(1) = 0, log(1000000) ≈ 13.8
	normalized := math.Log(float64(popularity)) / math.Log(1000000)
	return math.Min(normalized, 1.0)
}

// normalizeParams normalizes parameter count to 0-1 range
func (s *Selector) normalizeParams(params int) float64 {
	if params <= 0 {
		// Unknown params — don't penalize, assume mid-range
		return 0.5
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

// isTopWeekly checks if the model is in the top weekly list
func (s *Selector) isTopWeekly(id string) bool {
	if len(s.config.TopWeeklyModels) == 0 {
		return false
	}

	idLower := strings.ToLower(id)
	for _, top := range s.config.TopWeeklyModels {
		if idLower == strings.ToLower(top) {
			return true
		}
	}
	return false
}

// stealthMaxAge is the maximum age for a model to be considered a "stealth" launch.
// Models published within this window from a known provider get boosted.
const stealthMaxAge = 7 * 24 * time.Hour

// knownQualityProviders are model ID prefixes for reputable providers whose
// stealth (recently published) free models should be prioritized.
var knownQualityProviders = []string{
	"google/", "anthropic/", "openai/", "meta-llama/", "mistralai/",
	"deepseek/", "qwen/", "stepfun/", "nvidia/", "cohere/",
	"microsoft/", "xiaomi/", "allenai/", "openrouter/",
}

// isStealthModel returns true if the model is a recent launch (within stealthMaxAge)
// from a known quality provider. These are usually quiet drops of excellent models.
func (s *Selector) isStealthModel(model openrouter.Model) bool {
	if model.Created == 0 {
		return false
	}
	age := time.Since(time.Unix(model.Created, 0))
	if age > stealthMaxAge {
		return false
	}
	idLower := strings.ToLower(model.ID)
	for _, prefix := range knownQualityProviders {
		if strings.HasPrefix(idLower, prefix) {
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
		{[]string{"stepfun"}, 0.15},
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
		{[]string{"openrouter"}, 0.10},
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
	models, err := s.client.GetModels()
	if err != nil {
		return nil, fmt.Errorf("failed to get models: %w", err)
	}

	return s.rankCandidates(models, n)
}

func isFreeModel(model openrouter.Model) bool {
	prompt, err1 := strconv.ParseFloat(model.Pricing.Prompt, 64)
	completion, err2 := strconv.ParseFloat(model.Pricing.Completion, 64)
	if err1 != nil || err2 != nil {
		return false
	}

	return prompt == 0 && completion == 0
}

func findModelByID(models []openrouter.Model, id string) (openrouter.Model, bool) {
	for _, model := range models {
		if model.ID == id {
			return model, true
		}
	}

	return openrouter.Model{}, false
}

func containsModel(models []openrouter.Model, id string) bool {
	_, found := findModelByID(models, id)
	return found
}

func candidateOrder(candidates []openrouter.Model, preferredID string) []int {
	order := make([]int, 0, len(candidates))
	seen := map[int]bool{}

	if preferredID != "" {
		for i := range candidates {
			if candidates[i].ID == preferredID {
				order = append(order, i)
				seen[i] = true
				break
			}
		}
	}

	for i := range candidates {
		if seen[i] {
			continue
		}
		order = append(order, i)
	}

	return order
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
