package config

import (
	"os"
	"strconv"
)

// Config holds the configuration for the frugalai proxy
type Config struct {
	// OpenRouter API key (required)
	APIKey string

	// Server port (default: 8080)
	Port int

	// Minimum parameter count for model selection (default: 0)
	MinParams int

	// Minimum popularity score for model selection (default: 0)
	MinPopularity int

	// Enable OpenAI-compatible API (default: true)
	EnableOpenAI bool

	// Enable Anthropic-compatible API (default: true)
	EnableAnthropic bool

	// OpenAI endpoint path (default: /v1)
	OpenAIPath string

	// Anthropic endpoint path (default: /v1)
	AnthropicPath string

	// Log level (debug, info, warn, error)
	LogLevel string

	// Cache TTL for models in seconds (default: 300)
	CacheTTL int

	// Prefer specific model architectures
	PreferredArchitectures []string

	// Model index to use from top candidates (0-based, -1 for auto/interactive)
	ModelIndex int

	// Number of candidates to show
	NumCandidates int
}

// LoadFromEnv loads configuration from environment variables (used when CLI is not available)
func LoadFromEnv() *Config {
	cfg := &Config{
		Port:                8080,
		MinParams:           0,
		MinPopularity:       0,
		EnableOpenAI:        true,
		EnableAnthropic:     true,
		OpenAIPath:          "/v1",
		AnthropicPath:       "/v1",
		LogLevel:            "info",
		CacheTTL:            300,
		PreferredArchitectures: []string{},
		ModelIndex:          -1,
		NumCandidates:       10,
	}

	// Environment variables
	if v := os.Getenv("OPENROUTER_API_KEY"); v != "" {
		cfg.APIKey = v
	}
	if v := os.Getenv("FRUGALAI_API_KEY"); v != "" {
		cfg.APIKey = v
	}
	if v := os.Getenv("FRUGALAI_PORT"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Port = i
		}
	}
	if v := os.Getenv("FRUGALAI_MIN_PARAMS"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.MinParams = i
		}
	}
	if v := os.Getenv("FRUGALAI_MIN_POPULARITY"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.MinPopularity = i
		}
	}
	if v := os.Getenv("FRUGALAI_LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}
	if v := os.Getenv("FRUGALAI_CACHE_TTL"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.CacheTTL = i
		}
	}
	if v := os.Getenv("FRUGALAI_PREFERRED_ARCH"); v != "" {
		cfg.PreferredArchitectures = splitAndTrim(v)
	}
	if v := os.Getenv("FRUGALAI_MODEL_INDEX"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.ModelIndex = i
		}
	}
	if v := os.Getenv("FRUGALAI_NUM_CANDIDATES"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.NumCandidates = i
		}
	}

	return cfg
}

func splitAndTrim(s string) []string {
	parts := []string{}
	for _, p := range splitComma(s) {
		if trimmed := trimSpace(p); trimmed != "" {
			parts = append(parts, trimmed)
		}
	}
	return parts
}

func splitComma(s string) []string {
	result := []string{}
	current := ""
	for _, c := range s {
		if c == ',' {
			result = append(result, current)
			current = ""
		} else {
			current += string(c)
		}
	}
	if current != "" {
		result = append(result, current)
	}
	return result
}

func trimSpace(s string) string {
	start := 0
	end := len(s)
	for i, c := range s {
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			start = i
			break
		}
	}
	for i := len(s) - 1; i >= start; i-- {
		c := s[i]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			end = i + 1
			break
		}
	}
	if start >= end {
		return ""
	}
	return s[start:end]
}
