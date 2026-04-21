package ui

import (
	"embed"
	"html/template"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/mosajjal/frugalai/internal/openrouter"
	"github.com/mosajjal/frugalai/internal/store"
)

//go:embed static templates
var assets embed.FS

var tmpl = template.Must(template.ParseFS(assets,
	"templates/index.html",
	"templates/partials/stats.html",
	"templates/partials/logs.html",
	"templates/partials/model.html",
	"templates/partials/usage.html",
))

type Handler struct {
	store      *store.Store
	getManager func() *openrouter.ModelManager
	forceModel func(id string) bool
	startTime  time.Time
}

func NewHandler(s *store.Store, getManager func() *openrouter.ModelManager, forceModel func(id string) bool, startTime time.Time) *Handler {
	return &Handler{
		store:      s,
		getManager: getManager,
		forceModel: forceModel,
		startTime:  startTime,
	}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux, mw func(http.Handler) http.Handler) {
	w := func(next http.Handler) http.Handler {
		if mw != nil {
			return mw(next)
		}
		return next
	}
	mux.Handle("/admin/ui/static/", w(http.StripPrefix("/admin", http.FileServer(http.FS(assets)))))
	mux.Handle("/admin/ui/", w(http.HandlerFunc(h.handleIndex)))
	mux.Handle("/admin/ui/partials/stats", w(http.HandlerFunc(h.handleStatsPartial)))
	mux.Handle("/admin/ui/partials/logs", w(http.HandlerFunc(h.handleLogsPartial)))
	mux.Handle("/admin/ui/partials/model", w(http.HandlerFunc(h.handleModelPartial)))
	mux.Handle("/admin/ui/partials/usage", w(http.HandlerFunc(h.handleUsagePartial)))
	mux.Handle("/admin/ui/model/force", w(http.HandlerFunc(h.handleForceModel)))
}

type pageData struct {
	Stats statsData
	Logs  logsData
	Model modelData
	Usage []usageRow
}

type usageRow struct {
	Model     string
	Requests  int64
	TokensIn  int64
	TokensOut int64
	Failures  int64
}

type statsData struct {
	Uptime         string
	TotalRequests  int64
	TotalTokensIn  int64
	TotalTokensOut int64
	TotalFailures  int64
	TotalLogs      int
	CurrentModel   string
	Candidates     int
}

type logsData struct {
	Entries    []store.LogEntry
	Offset     int
	NextOffset int
	HasMore    bool
}

type modelData struct {
	Current    *openrouter.Model
	Candidates []openrouter.Model
	CurrentIdx int
}

const logsPerPage = 50

func (h *Handler) buildStats() statsData {
	mgr := h.getManager()
	model := ""
	candidates := 0
	if mgr != nil {
		if mgr.Current != nil {
			model = mgr.Current.ID
		}
		candidates = len(mgr.Candidates)
	}
	snap := h.store.Snapshot()
	return statsData{
		Uptime:         formatDuration(time.Since(h.startTime)),
		TotalRequests:  snap.TotalRequests,
		TotalTokensIn:  snap.TotalTokensIn,
		TotalTokensOut: snap.TotalTokensOut,
		TotalFailures:  snap.TotalFailures,
		TotalLogs:      snap.TotalLogs,
		CurrentModel:   model,
		Candidates:     candidates,
	}
}

func (h *Handler) buildLogs(offset int) logsData {
	entries := h.store.Logs(offset, logsPerPage)
	total := h.store.TotalLogs()
	nextOffset := offset + len(entries)
	return logsData{
		Entries:    entries,
		Offset:     offset,
		NextOffset: nextOffset,
		HasMore:    nextOffset < total,
	}
}

func (h *Handler) buildUsage() []usageRow {
	snap := h.store.Snapshot()
	rows := make([]usageRow, 0, len(snap.Models))
	for id, ms := range snap.Models {
		rows = append(rows, usageRow{
			Model:     id,
			Requests:  ms.Requests,
			TokensIn:  ms.TokensIn,
			TokensOut: ms.TokensOut,
			Failures:  ms.Failures,
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Requests > rows[j].Requests })
	return rows
}

func (h *Handler) buildModel() modelData {
	mgr := h.getManager()
	if mgr == nil {
		return modelData{}
	}
	return modelData{
		Current:    mgr.Current,
		Candidates: mgr.Candidates,
		CurrentIdx: mgr.CurrentIdx,
	}
}

func (h *Handler) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/ui/" && r.URL.Path != "/ui" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl.ExecuteTemplate(w, "index.html", pageData{
		Stats: h.buildStats(),
		Logs:  h.buildLogs(0),
		Model: h.buildModel(),
		Usage: h.buildUsage(),
	})
}

func (h *Handler) handleUsagePartial(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl.ExecuteTemplate(w, "usage.html", h.buildUsage())
}

func (h *Handler) handleStatsPartial(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl.ExecuteTemplate(w, "stats.html", h.buildStats())
}

func (h *Handler) handleLogsPartial(w http.ResponseWriter, r *http.Request) {
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl.ExecuteTemplate(w, "logs.html", h.buildLogs(offset))
}

func (h *Handler) handleModelPartial(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl.ExecuteTemplate(w, "model.html", h.buildModel())
}

func (h *Handler) handleForceModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	id := r.URL.Query().Get("id")
	if id != "" {
		h.forceModel(id)
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl.ExecuteTemplate(w, "model.html", h.buildModel())
}

func formatDuration(d time.Duration) string {
	d = d.Round(time.Second)
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := int(d.Seconds()) % 60
	if h > 0 {
		return strconv.Itoa(h) + "h " + strconv.Itoa(m) + "m"
	}
	if m > 0 {
		return strconv.Itoa(m) + "m " + strconv.Itoa(s) + "s"
	}
	return strconv.Itoa(s) + "s"
}
