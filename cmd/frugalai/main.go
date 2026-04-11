package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/mosajjal/frugalai/internal/config"
	"github.com/mosajjal/frugalai/internal/model"
	"github.com/mosajjal/frugalai/internal/openrouter"
	"github.com/mosajjal/frugalai/internal/server/anthropic"
	"github.com/mosajjal/frugalai/internal/server/openai"
	"github.com/urfave/cli/v2"
)

var (
	startTime      time.Time
	modelManager   *openrouter.ModelManager
	modelManagerMu sync.RWMutex
)

const defaultProbeCandidateCount = 10

func main() {
	app := &cli.App{
		Name:    "frugalai",
		Usage:   "Intelligent LLM proxy that routes to the best free model on OpenRouter",
		Version: "1.0.0",
		Before:  setupLogging,
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:     "api-key",
				Aliases:  []string{"k"},
				Usage:    "OpenRouter API key (can also set OPENROUTER_API_KEY env var)",
				EnvVars:  []string{"OPENROUTER_API_KEY", "FRUGALAI_API_KEY"},
				Required: true,
			},
			&cli.IntFlag{
				Name:    "port",
				Aliases: []string{"p"},
				Usage:   "Server port (default: 8080)",
				Value:   8080,
				EnvVars: []string{"FRUGALAI_PORT"},
			},
			&cli.IntFlag{
				Name:    "min-params",
				Usage:   "Minimum parameter count for model selection (default: 0)",
				Value:   0,
				EnvVars: []string{"FRUGALAI_MIN_PARAMS"},
			},
			&cli.IntFlag{
				Name:    "min-popularity",
				Usage:   "Minimum popularity score for model selection (default: 0)",
				Value:   0,
				EnvVars: []string{"FRUGALAI_MIN_POPULARITY"},
			},
			&cli.BoolFlag{
				Name:  "enable-openai",
				Usage: "Enable OpenAI-compatible API (default: true)",
				Value: true,
			},
			&cli.BoolFlag{
				Name:  "enable-anthropic",
				Usage: "Enable Anthropic-compatible API (default: true)",
				Value: true,
			},
			&cli.StringFlag{
				Name:  "openai-path",
				Usage: "OpenAI endpoint path (default: /v1)",
				Value: "/v1",
			},
			&cli.StringFlag{
				Name:  "anthropic-path",
				Usage: "Anthropic endpoint path (default: /v1)",
				Value: "/v1",
			},
			&cli.StringFlag{
				Name:    "log-level",
				Usage:   "Log level: debug, info, warn, error (default: info)",
				Value:   "info",
				EnvVars: []string{"FRUGALAI_LOG_LEVEL"},
			},
			&cli.IntFlag{
				Name:    "cache-ttl",
				Usage:   "Model cache TTL in seconds (default: 300)",
				Value:   300,
				EnvVars: []string{"FRUGALAI_CACHE_TTL"},
			},
			&cli.StringFlag{
				Name:    "preferred-arch",
				Usage:   "Comma-separated list of preferred architectures (e.g., transformer,llama)",
				EnvVars: []string{"FRUGALAI_PREFERRED_ARCH"},
			},
			&cli.IntFlag{
				Name:    "model-index",
				Usage:   "Model index to use from candidates (default: 0)",
				Value:   0,
				EnvVars: []string{"FRUGALAI_MODEL_INDEX"},
			},
			&cli.IntFlag{
				Name:    "num-candidates",
				Usage:   "Number of model candidates to show (default: 10)",
				Value:   10,
				EnvVars: []string{"FRUGALAI_NUM_CANDIDATES"},
			},
		},
		Action: run,
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatalf("[ERROR] %v", err)
	}
}

func setupLogging(c *cli.Context) error {
	level := c.String("log-level")
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.SetOutput(os.Stdout)

	switch level {
	case "debug":
		log.SetPrefix("[DEBUG] ")
	case "info":
		log.SetPrefix("[INFO] ")
	case "warn":
		log.SetPrefix("[WARN] ")
	case "error":
		log.SetPrefix("[ERROR] ")
	default:
		log.SetPrefix("[INFO] ")
	}
	return nil
}

func run(c *cli.Context) error {
	return runHeadless(c)
}

// runHeadless runs the server without TUI
func runHeadless(c *cli.Context) error {
	setupLogging(c)
	log.Printf("[INFO] Starting FrugalAI...")

	startTime = time.Now()

	// Start from environment-backed defaults so runtime ranking matches selector defaults.
	cfg := config.LoadFromEnv()
	cfg.APIKey = c.String("api-key")
	cfg.Port = c.Int("port")
	cfg.MinParams = c.Int("min-params")
	cfg.MinPopularity = c.Int("min-popularity")
	cfg.EnableOpenAI = c.Bool("enable-openai")
	cfg.EnableAnthropic = c.Bool("enable-anthropic")
	cfg.OpenAIPath = c.String("openai-path")
	cfg.AnthropicPath = c.String("anthropic-path")
	cfg.LogLevel = c.String("log-level")
	cfg.CacheTTL = c.Int("cache-ttl")
	cfg.PreferredArchitectures = splitAndTrim(c.String("preferred-arch"))
	cfg.ModelIndex = c.Int("model-index")
	cfg.NumCandidates = c.Int("num-candidates")

	// Create OpenRouter client
	client := openrouter.NewClient(cfg.APIKey, cfg.CacheTTL)

	// Create model selector
	selector := model.NewSelector(client, cfg)

	// Initialize model manager and select initial model
	initializeModelManager(selector, cfg)

	// Create handlers with model manager
	openaiHandler := openai.NewHandlerWithManager(selector, client, modelManager)
	anthropicHandler := anthropic.NewHandlerWithManager(selector, client, modelManager)

	// Setup HTTP server
	mux := http.NewServeMux()

	// Register OpenAI-compatible routes
	if cfg.EnableOpenAI {
		openaiHandler.RegisterRoutes(mux, cfg.OpenAIPath)
		log.Printf("[INFO] OpenAI-compatible API enabled at: http://localhost:%d%s", cfg.Port, cfg.OpenAIPath)
	}

	// Register Anthropic-compatible routes
	if cfg.EnableAnthropic {
		anthropicHandler.RegisterRoutes(mux, cfg.AnthropicPath)
		log.Printf("[INFO] Anthropic-compatible API enabled at: http://localhost:%d%s", cfg.Port, cfg.AnthropicPath)
	}

	// Health check endpoint with model info
	mux.HandleFunc("/health", healthHandler)

	// Model info endpoint
	mux.HandleFunc("/model", modelInfoHandler)

	// Model switch endpoint (for manual switching)
	mux.HandleFunc("/model/switch", modelSwitchHandler(selector))

	// Force a live candidate refresh and probe-validated reselection.
	mux.HandleFunc("/model/refresh", modelRefreshHandler(selector))

	// Candidates endpoint
	mux.HandleFunc("/candidates", candidatesHandler(selector))

	// Lightweight upstream health probe for the current or requested model.
	mux.HandleFunc("/probe", probeHandler(selector))

	// Create server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine with restart capability
	go runServer(server, cfg)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Println("[INFO] Shutting down server...")
	if err := server.Close(); err != nil {
		log.Printf("[ERROR] Error closing server: %v", err)
	}
	log.Println("[INFO] Server stopped")
	return nil
}

func runServer(server *http.Server, cfg *config.Config) {
	for {
		log.Printf("[INFO] FrugalAI proxy listening on port %d", cfg.Port)
		log.Printf("[INFO] Min params: %d, Min popularity: %d", cfg.MinParams, cfg.MinPopularity)
		if len(cfg.PreferredArchitectures) > 0 {
			log.Printf("[INFO] Preferred architectures: %v", cfg.PreferredArchitectures)
		}

		if err := server.ListenAndServe(); err != nil {
			if err == http.ErrServerClosed {
				return
			}
			log.Printf("[ERROR] Server error: %v", err)
			log.Printf("[INFO] Restarting server in 5 seconds...")
			time.Sleep(5 * time.Second)
		}
	}
}

func initializeModelManager(selector *model.Selector, cfg *config.Config) {
	log.Println("[INFO] Fetching available free models from OpenRouter...")

	candidates, err := selector.GetTopCandidates(cfg.NumCandidates)
	if err != nil {
		log.Printf("[WARN] Could not get model candidates: %v", err)
		log.Println("[INFO] Will retry on first request")
		// Create empty manager - will be populated later
		modelManager = &openrouter.ModelManager{
			Candidates:  []openrouter.Model{},
			Current:     nil,
			CurrentIdx:  0,
			Failures:    make(map[string]int),
			LastFailure: make(map[string]time.Time),
			Timeouts:    make(map[string]int),
			Burned:      make(map[string]bool),
		}
		return
	}

	// Show candidates
	log.Printf("[INFO] Found %d free model candidates:", len(candidates))
	for i, m := range candidates {
		log.Printf("  [%d] %s", i, m.Name)
		log.Printf("      ID: %s", m.ID)
		log.Printf("      Modality: %s, Tokenizer: %s", m.Architecture.Modality, m.Architecture.Tokenizer)
	}

	// Select model
	selectedIdx := 0
	if cfg.ModelIndex >= 0 && cfg.ModelIndex < len(candidates) {
		selectedIdx = cfg.ModelIndex
		log.Printf("[INFO] Using model index from CLI: %d", cfg.ModelIndex)
	} else if cfg.ModelIndex == -1 {
		selectedIdx = 0
		log.Printf("[INFO] Using best model (index 0)")
	} else {
		selectedIdx = 0
		log.Printf("[INFO] Using best model (index 0)")
	}

	preferredModelID := candidates[selectedIdx].ID
	selectedModel, selectedIdx, probe, err := selector.SelectWorkingCandidate(candidates, preferredModelID)
	if err != nil {
		log.Printf("[WARN] Could not probe any candidate successfully: %v", err)
		modelManager = &openrouter.ModelManager{
			Candidates:  candidates,
			Current:     nil,
			CurrentIdx:  0,
			Failures:    make(map[string]int),
			LastFailure: make(map[string]time.Time),
			Timeouts:    make(map[string]int),
			Burned:      make(map[string]bool),
		}
		return
	}

	log.Printf("[INFO] Selected model: %s", selectedModel.Name)
	log.Printf("[INFO]   ID: %s", selectedModel.ID)
	log.Printf("[INFO]   Modality: %s", selectedModel.Architecture.Modality)
	log.Printf("[INFO]   Tokenizer: %s", selectedModel.Architecture.Tokenizer)
	log.Printf("[INFO]   Context Length: %d", selectedModel.ContextLength)
	if selectedModel.ID != preferredModelID {
		log.Printf("[WARN] Preferred candidate %s failed probe; promoted %s instead", preferredModelID, selectedModel.ID)
	}
	logProbeSuccess(probe)

	// Create model manager
	modelManager = &openrouter.ModelManager{
		Candidates:  candidates,
		Current:     selectedModel,
		CurrentIdx:  selectedIdx,
		Failures:    make(map[string]int),
		LastFailure: make(map[string]time.Time),
		Timeouts:    make(map[string]int),
		Burned:      make(map[string]bool),
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	modelManagerMu.RLock()
	defer modelManagerMu.RUnlock()

	status := openrouter.HealthStatus{
		Status: "ok",
		Uptime: time.Since(startTime).Seconds(),
	}

	if modelManager != nil {
		status.Candidates = len(modelManager.Candidates)
		if modelManager.Current != nil {
			status.Model = modelManager.Current.ID
			status.ModelName = modelManager.Current.Name
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func modelInfoHandler(w http.ResponseWriter, r *http.Request) {
	modelManagerMu.RLock()
	defer modelManagerMu.RUnlock()

	if modelManager.Current == nil {
		http.Error(w, "No model selected", http.StatusServiceUnavailable)
		return
	}

	m := modelManager.Current
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"model_id":"%s","name":"%s","modality":"%s","tokenizer":"%s","context_length":%d,"params":%d,"popularity":%d}`,
		m.ID, m.Name, m.Architecture.Modality, m.Architecture.Tokenizer, m.ContextLength, m.Params, m.Popularity)
}

func modelSwitchHandler(selector *model.Selector) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		modelManagerMu.RLock()
		if modelManager == nil || len(modelManager.Candidates) == 0 {
			modelManagerMu.RUnlock()
			http.Error(w, "No candidates available", http.StatusServiceUnavailable)
			return
		}
		candidates := append([]openrouter.Model(nil), modelManager.Candidates...)
		startIdx := (modelManager.CurrentIdx + 1) % len(candidates)
		modelManagerMu.RUnlock()

		rotated := rotateCandidates(candidates, startIdx)
		selected, _, probe, err := selector.SelectWorkingCandidate(rotated, "")
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}

		modelManagerMu.Lock()
		defer modelManagerMu.Unlock()
		selectedIdx := findCandidateIndexByID(modelManager.Candidates, selected.ID)
		if selectedIdx < 0 {
			http.Error(w, "Selected model is no longer available", http.StatusServiceUnavailable)
			return
		}
		modelManager.Current = &modelManager.Candidates[selectedIdx]
		modelManager.CurrentIdx = selectedIdx
		log.Printf("[INFO] Switched to model: %s (index %d)", modelManager.Current.Name, selectedIdx)
		logProbeSuccess(probe)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":      "switched",
			"model_id":    modelManager.Current.ID,
			"model_name":  modelManager.Current.Name,
			"index":       selectedIdx,
			"probe_reply": probe.Reply,
			"latency_ms":  probe.Duration.Milliseconds(),
		})
	}
}

func modelRefreshHandler(selector *model.Selector) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		modelManagerMu.RLock()
		preferredID := ""
		if modelManager != nil && modelManager.Current != nil {
			preferredID = modelManager.Current.ID
		}
		modelManagerMu.RUnlock()

		selected, probe, err := refreshModelManager(selector, preferredID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":      "refreshed",
			"model_id":    selected.ID,
			"model_name":  selected.Name,
			"probe_reply": probe.Reply,
			"latency_ms":  probe.Duration.Milliseconds(),
		})
	}
}

func probeHandler(selector *model.Selector) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		modelID := r.URL.Query().Get("model")
		if modelID == "" {
			modelManagerMu.RLock()
			if modelManager != nil && modelManager.Current != nil {
				modelID = modelManager.Current.ID
			}
			modelManagerMu.RUnlock()
		}
		if modelID == "" {
			http.Error(w, "No model selected", http.StatusServiceUnavailable)
			return
		}

		probe, err := selector.ProbeModel(modelID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		logProbeSuccess(probe)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":     "ok",
			"model_id":   probe.ModelID,
			"prompt":     probe.Prompt,
			"reply":      probe.Reply,
			"latency_ms": probe.Duration.Milliseconds(),
		})
	}
}

func candidatesHandler(selector *model.Selector) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelManagerMu.RLock()
		defer modelManagerMu.RUnlock()

		if len(modelManager.Candidates) == 0 {
			// Try to refresh candidates
			candidates, err := selector.GetTopCandidates(10)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			modelManager.Candidates = candidates
		}

		type Candidate struct {
			Index      int    `json:"index"`
			ID         string `json:"id"`
			Name       string `json:"name"`
			Modality   string `json:"modality"`
			Tokenizer  string `json:"tokenizer"`
			ContextLen int    `json:"context_length"`
			Params     int    `json:"params"`
			Popularity int    `json:"popularity"`
			IsCurrent  bool   `json:"is_current"`
			Failures   int    `json:"failures"`
		}

		result := []Candidate{}
		for i, m := range modelManager.Candidates {
			result = append(result, Candidate{
				Index:      i,
				ID:         m.ID,
				Name:       m.Name,
				Modality:   m.Architecture.Modality,
				Tokenizer:  m.Architecture.Tokenizer,
				ContextLen: m.ContextLength,
				Params:     m.Params,
				Popularity: m.Popularity,
				IsCurrent:  modelManager.Current != nil && m.ID == modelManager.Current.ID,
				Failures:   modelManager.Failures[m.ID],
			})
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	}
}

func refreshModelManager(selector *model.Selector, preferredID string) (*openrouter.Model, *openrouter.ProbeResult, error) {
	modelManagerMu.RLock()
	candidateCount := defaultProbeCandidateCount
	if modelManager != nil && len(modelManager.Candidates) > 0 {
		candidateCount = len(modelManager.Candidates)
	}
	modelManagerMu.RUnlock()

	candidates, err := selector.GetTopCandidates(candidateCount)
	if err != nil {
		return nil, nil, err
	}

	selected, selectedIdx, probe, err := selector.SelectWorkingCandidate(candidates, preferredID)
	if err != nil {
		return nil, nil, err
	}

	modelManagerMu.Lock()
	defer modelManagerMu.Unlock()

	modelManager = &openrouter.ModelManager{
		Candidates:  candidates,
		Current:     selected,
		CurrentIdx:  selectedIdx,
		Failures:    make(map[string]int),
		LastFailure: make(map[string]time.Time),
		Timeouts:    make(map[string]int),
		Burned:      make(map[string]bool),
	}

	log.Printf("[INFO] Refreshed candidates; selected working model %s", selected.ID)
	logProbeSuccess(probe)
	return selected, probe, nil
}

func findCandidateIndexByID(candidates []openrouter.Model, modelID string) int {
	for i := range candidates {
		if candidates[i].ID == modelID {
			return i
		}
	}
	return -1
}

func rotateCandidates(candidates []openrouter.Model, startIdx int) []openrouter.Model {
	if len(candidates) == 0 {
		return nil
	}

	rotated := make([]openrouter.Model, 0, len(candidates))
	for i := 0; i < len(candidates); i++ {
		idx := (startIdx + i) % len(candidates)
		rotated = append(rotated, candidates[idx])
	}
	return rotated
}

func logProbeSuccess(probe *openrouter.ProbeResult) {
	if probe == nil {
		return
	}
	log.Printf("[INFO] Probe succeeded for %s in %dms with reply %q",
		probe.ModelID, probe.Duration.Milliseconds(), probe.Reply)
}

// GetCurrentModelID returns the current model ID (thread-safe)
func GetCurrentModelID() string {
	modelManagerMu.RLock()
	defer modelManagerMu.RUnlock()

	if modelManager == nil || modelManager.Current == nil {
		return ""
	}
	return modelManager.Current.ID
}

// RecordModelFailure records a failure for the current model and potentially switches
func RecordModelFailure(modelID string, statusCode int) bool {
	modelManagerMu.Lock()
	defer modelManagerMu.Unlock()

	if modelManager == nil {
		return false
	}

	// Increment failure count
	modelManager.Failures[modelID]++
	modelManager.LastFailure[modelID] = time.Now()

	log.Printf("[WARN] Model %s failed (status %d), failure count: %d",
		modelID, statusCode, modelManager.Failures[modelID])

	// Check if we should switch models
	// Switch on: 429 (rate limit), 500+, or 3+ failures in quick succession
	shouldSwitch := false

	if statusCode == 429 {
		shouldSwitch = true
		log.Printf("[INFO] Rate limit hit, switching model")
	} else if statusCode >= 500 {
		shouldSwitch = true
		log.Printf("[INFO] Server error, switching model")
	} else if modelManager.Failures[modelID] >= 3 {
		// Check if failures happened recently (within 2 minutes)
		recentFailures := 0
		cutoff := time.Now().Add(-2 * time.Minute)
		if modelManager.LastFailure[modelID].After(cutoff) {
			recentFailures = modelManager.Failures[modelID]
		}
		if recentFailures >= 3 {
			shouldSwitch = true
			log.Printf("[INFO] Multiple recent failures, switching model")
		}
	}

	if shouldSwitch && len(modelManager.Candidates) > 1 {
		// Find next candidate that isn't the current one
		for i := 1; i < len(modelManager.Candidates); i++ {
			nextIdx := (modelManager.CurrentIdx + i) % len(modelManager.Candidates)
			nextModel := &modelManager.Candidates[nextIdx]

			// Skip if this model also has recent failures
			if modelManager.Failures[nextModel.ID] >= 3 {
				continue
			}

			log.Printf("[INFO] Switching from %s to %s",
				modelManager.Current.ID, nextModel.ID)

			modelManager.Current = nextModel
			modelManager.CurrentIdx = nextIdx
			return true
		}

		log.Printf("[WARN] No alternative models available, keeping current model")
	}

	return false
}

// RecordModelTimeout records a timeout for a model and potentially burns/switches it
func RecordModelTimeout(modelID string) bool {
	modelManagerMu.Lock()
	defer modelManagerMu.Unlock()

	if modelManager == nil {
		return false
	}

	// Increment timeout count
	modelManager.Timeouts[modelID]++

	log.Printf("[WARN] Model %s timed out, timeout count: %d",
		modelID, modelManager.Timeouts[modelID])

	// Burn model on first timeout
	modelManager.Burned[modelID] = true
	log.Printf("[WARN] Model %s burned after %d timeouts",
		modelID, modelManager.Timeouts[modelID])

	// Switch to next non-burned model
	if len(modelManager.Candidates) > 1 {
		for i := 1; i < len(modelManager.Candidates); i++ {
			nextIdx := (modelManager.CurrentIdx + i) % len(modelManager.Candidates)
			nextModel := &modelManager.Candidates[nextIdx]

			// Skip burned models
			if modelManager.Burned[nextModel.ID] {
				continue
			}

			log.Printf("[INFO] Switching from %s to %s due to timeout",
				modelManager.Current.ID, nextModel.ID)

			modelManager.Current = nextModel
			modelManager.CurrentIdx = nextIdx
			return true
		}

		log.Printf("[WARN] All models burned or unavailable, keeping current model")
	}

	return false
}

// GetCandidates returns the list of candidate models (thread-safe)
func GetCandidates() []openrouter.Model {
	modelManagerMu.RLock()
	defer modelManagerMu.RUnlock()

	if modelManager == nil {
		return []openrouter.Model{}
	}
	return modelManager.Candidates
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

// Atomic model ID accessor for handlers
var currentModelID atomic.Value

func init() {
	currentModelID.Store("")
}

func SetCurrentModelID(id string) {
	currentModelID.Store(id)
}

func LoadCurrentModelID() string {
	v := currentModelID.Load()
	if v == nil {
		return ""
	}
	return v.(string)
}
