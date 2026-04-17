package api

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/adortb/adortb-ml-predictor/internal/metrics"
	"github.com/adortb/adortb-ml-predictor/internal/model"
)

// TrainHandler 在线学习 API 处理器（FTRL 增量更新）
type TrainHandler struct {
	ftrl *model.FTRLModel
}

// NewTrainHandler 创建在线学习处理器
func NewTrainHandler(ftrl *model.FTRLModel) *TrainHandler {
	return &TrainHandler{ftrl: ftrl}
}

// TrainSampleRequest POST /v1/train_sample 请求体
type TrainSampleRequest struct {
	Features map[string]float64 `json:"features"` // 特征向量（已 hash）
	Label    float64            `json:"label"`    // 0 或 1（click/no-click）
	Weight   float64            `json:"weight"`   // 样本权重（默认 1.0）
}

// TrainSampleResponse 响应
type TrainSampleResponse struct {
	Status       string `json:"status"`
	ModelVersion string `json:"model_version"`
	LatencyUS    int64  `json:"latency_us"`
}

// HandleTrainSample POST /v1/train_sample：接收在线样本，触发 FTRL 一步更新
func (h *TrainHandler) HandleTrainSample(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req TrainSampleRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		metrics.PredictTotal.WithLabelValues("/v1/train_sample", "error").Inc()
		return
	}

	if req.Label != 0 && req.Label != 1 {
		writeError(w, http.StatusBadRequest, "label must be 0 or 1")
		return
	}
	if req.Weight <= 0 {
		req.Weight = 1.0
	}
	if len(req.Features) == 0 {
		writeError(w, http.StatusBadRequest, "features must not be empty")
		return
	}

	start := time.Now()
	h.ftrl.Update(req.Features, req.Label, req.Weight)
	elapsed := time.Since(start).Microseconds()

	metrics.PredictTotal.WithLabelValues("/v1/train_sample", "ok").Inc()
	writeJSON(w, http.StatusOK, TrainSampleResponse{
		Status:       "updated",
		ModelVersion: h.ftrl.GetVersion(),
		LatencyUS:    elapsed,
	})
}
