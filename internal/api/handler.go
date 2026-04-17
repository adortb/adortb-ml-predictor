package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/adortb/adortb-ml-predictor/internal/metrics"
	"github.com/adortb/adortb-ml-predictor/internal/model"
	"github.com/adortb/adortb-ml-predictor/internal/prediction"
)

// Handler HTTP API 处理器
type Handler struct {
	pipeline *prediction.Pipeline
	registry *model.Registry
	ftrl     *model.FTRLModel
}

// NewHandler 创建处理器
func NewHandler(pipeline *prediction.Pipeline, registry *model.Registry, ftrl *model.FTRLModel) *Handler {
	return &Handler{pipeline: pipeline, registry: registry, ftrl: ftrl}
}

// HandlePredictCTR POST /v1/predict/ctr
func (h *Handler) HandlePredictCTR(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req prediction.Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		metrics.PredictTotal.WithLabelValues("/v1/predict/ctr", "error").Inc()
		return
	}

	start := time.Now()
	result := h.pipeline.Predict(req)
	elapsed := float64(time.Since(start).Milliseconds())

	metrics.PredictDuration.WithLabelValues("/v1/predict/ctr").Observe(elapsed)
	metrics.PredictTotal.WithLabelValues("/v1/predict/ctr", "ok").Inc()

	writeJSON(w, http.StatusOK, result)
}

// HandlePredictBatch POST /v1/predict/batch
func (h *Handler) HandlePredictBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req prediction.BatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		metrics.PredictTotal.WithLabelValues("/v1/predict/batch", "error").Inc()
		return
	}
	if len(req.Ads) == 0 {
		writeError(w, http.StatusBadRequest, "ads list is empty")
		return
	}

	start := time.Now()
	result := h.pipeline.PredictBatch(req)
	elapsed := float64(time.Since(start).Milliseconds())

	metrics.PredictDuration.WithLabelValues("/v1/predict/batch").Observe(elapsed)
	metrics.PredictTotal.WithLabelValues("/v1/predict/batch", "ok").Inc()

	writeJSON(w, http.StatusOK, result)
}

// HandleModelCurrent GET /v1/model/current
func (h *Handler) HandleModelCurrent(w http.ResponseWriter, r *http.Request) {
	m := h.registry.Current()
	writeJSON(w, http.StatusOK, map[string]string{
		"version": m.GetVersion(),
	})
}

// HandleModelReload POST /v1/model/reload
func (h *Handler) HandleModelReload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if err := h.registry.Reload(ctx); err != nil {
		metrics.ModelReloadTotal.WithLabelValues("error").Inc()
		writeError(w, http.StatusInternalServerError, "reload failed: "+err.Error())
		return
	}

	metrics.ModelReloadTotal.WithLabelValues("ok").Inc()
	m := h.registry.Current()
	writeJSON(w, http.StatusOK, map[string]string{
		"version": m.GetVersion(),
		"status":  "reloaded",
	})
}

// HandleListModels GET /v1/models：列出当前加载的模型及版本
func (h *Handler) HandleListModels(w http.ResponseWriter, r *http.Request) {
	type modelInfo struct {
		Name    string `json:"name"`
		Version string `json:"version"`
		Type    string `json:"type"`
	}
	models := []modelInfo{
		{Name: "lr", Version: h.registry.Current().GetVersion(), Type: "lr"},
	}
	if h.ftrl != nil {
		models = append(models, modelInfo{Name: "ftrl", Version: h.ftrl.GetVersion(), Type: "ftrl"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": models})
}

// HandleHealth GET /health
func (h *Handler) HandleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}
