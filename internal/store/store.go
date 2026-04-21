package store

import (
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const DefaultSize = 1000

type Level string

const (
	LevelDebug Level = "DEBUG"
	LevelInfo  Level = "INFO"
	LevelWarn  Level = "WARN"
	LevelError Level = "ERROR"
)

type LogEntry struct {
	Time    time.Time `json:"time"`
	Level   Level     `json:"level"`
	Message string    `json:"message"`
}

type ModelStats struct {
	Requests  int64
	TokensIn  int64
	TokensOut int64
	Failures  int64
}

type Snapshot struct {
	TotalRequests  int64
	TotalTokensIn  int64
	TotalTokensOut int64
	TotalFailures  int64
	TotalLogs      int
	Models         map[string]ModelStats
}

type Store struct {
	mu      sync.RWMutex
	entries []LogEntry
	head    int
	count   int
	size    int

	requests      atomic.Int64
	totalTokensIn  atomic.Int64
	totalTokensOut atomic.Int64
	totalFailures  atomic.Int64

	modelMu sync.Mutex
	models  map[string]*ModelStats
}

func New(size int) *Store {
	if size <= 0 {
		size = DefaultSize
	}
	return &Store{
		size:    size,
		entries: make([]LogEntry, size),
		models:  make(map[string]*ModelStats),
	}
}

// Write implements io.Writer — captures log output into the ring buffer.
func (s *Store) Write(p []byte) (n int, err error) {
	msg := strings.TrimSpace(string(p))
	if msg == "" {
		return len(p), nil
	}

	level := LevelInfo
	switch {
	case strings.Contains(msg, "[DEBUG]"):
		level = LevelDebug
	case strings.Contains(msg, "[WARN]"):
		level = LevelWarn
	case strings.Contains(msg, "[ERROR]"):
		level = LevelError
	}

	s.mu.Lock()
	s.entries[s.head] = LogEntry{Time: time.Now(), Level: level, Message: msg}
	s.head = (s.head + 1) % s.size
	if s.count < s.size {
		s.count++
	}
	s.mu.Unlock()

	return len(p), nil
}

// Logs returns entries newest-first, starting at offset, up to limit entries.
func (s *Store) Logs(offset, limit int) []LogEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.count == 0 || offset >= s.count {
		return nil
	}

	end := s.count - offset
	start := max(end-limit, 0)

	result := make([]LogEntry, 0, end-start)
	for i := end - 1; i >= start; i-- {
		idx := ((s.head - 1 - i) % s.size + s.size) % s.size
		result = append(result, s.entries[idx])
	}
	return result
}

func (s *Store) TotalLogs() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.count
}

func (s *Store) RecordRequest(modelID string, tokensIn, tokensOut int) {
	s.requests.Add(1)
	s.totalTokensIn.Add(int64(tokensIn))
	s.totalTokensOut.Add(int64(tokensOut))

	s.modelMu.Lock()
	m := s.models[modelID]
	if m == nil {
		m = &ModelStats{}
		s.models[modelID] = m
	}
	m.Requests++
	m.TokensIn += int64(tokensIn)
	m.TokensOut += int64(tokensOut)
	s.modelMu.Unlock()
}

func (s *Store) RecordFailure(modelID string) {
	s.totalFailures.Add(1)

	s.modelMu.Lock()
	m := s.models[modelID]
	if m == nil {
		m = &ModelStats{}
		s.models[modelID] = m
	}
	m.Failures++
	s.modelMu.Unlock()
}

func (s *Store) Snapshot() Snapshot {
	s.modelMu.Lock()
	models := make(map[string]ModelStats, len(s.models))
	for k, v := range s.models {
		models[k] = *v
	}
	s.modelMu.Unlock()

	return Snapshot{
		TotalRequests:  s.requests.Load(),
		TotalTokensIn:  s.totalTokensIn.Load(),
		TotalTokensOut: s.totalTokensOut.Load(),
		TotalFailures:  s.totalFailures.Load(),
		TotalLogs:      s.TotalLogs(),
		Models:         models,
	}
}

// IncrRequests is kept for the middleware counter — RecordRequest is preferred when tokens are known.
func (s *Store) IncrRequests() {
	s.requests.Add(1)
}

func (s *Store) TotalRequests() int64 {
	return s.requests.Load()
}
