package prediction

import (
	"time"

	"github.com/adortb/adortb-ml-predictor/internal/feature"
	"github.com/adortb/adortb-ml-predictor/internal/model"
)

// Request 单次预测请求
type Request struct {
	User    feature.UserFeatures    `json:"user"`
	Context feature.ContextFeatures `json:"context"`
	Ad      feature.AdFeatures      `json:"ad"`
}

// BatchRequest 批量预测请求（同一用户多候选广告）
type BatchRequest struct {
	User    feature.UserFeatures    `json:"user"`
	Context feature.ContextFeatures `json:"context"`
	Ads     []feature.AdFeatures    `json:"ads"`
}

// Result 单次预测结果
type Result struct {
	PCTR         float64 `json:"pctr"`
	ModelVersion string  `json:"model_version"`
	LatencyUS    int64   `json:"latency_us"`
}

// BatchResult 批量预测结果
type BatchResult struct {
	Results      []Result `json:"results"`
	ModelVersion string   `json:"model_version"`
	LatencyUS    int64    `json:"latency_us"`
}

// Pipeline 预测流水线：特征提取 → 模型推理 → 校准
type Pipeline struct {
	extractor *feature.Extractor
	registry  *model.Registry
}

// NewPipeline 创建预测流水线
func NewPipeline(registry *model.Registry) *Pipeline {
	return &Pipeline{
		extractor: feature.NewExtractor(),
		registry:  registry,
	}
}

// Predict 单次预测
func (p *Pipeline) Predict(req Request) Result {
	start := time.Now()

	fv := p.extractor.Extract(req.User, req.Context, req.Ad)
	m := p.registry.Current()
	raw := m.Predict(fv)
	pctr := p.registry.CurrentCalibrator().Calibrate(raw)

	return Result{
		PCTR:         pctr,
		ModelVersion: m.GetVersion(),
		LatencyUS:    time.Since(start).Microseconds(),
	}
}

// PredictBatch 批量预测，并发处理多个候选广告
func (p *Pipeline) PredictBatch(req BatchRequest) BatchResult {
	start := time.Now()

	n := len(req.Ads)
	results := make([]Result, n)

	// 批量预测：提前提取一次用户/上下文特征，避免重复计算
	m := p.registry.Current()
	cal := p.registry.CurrentCalibrator()

	for i, ad := range req.Ads {
		fv := p.extractor.Extract(req.User, req.Context, ad)
		raw := m.Predict(fv)
		pctr := cal.Calibrate(raw)
		results[i] = Result{
			PCTR:         pctr,
			ModelVersion: m.GetVersion(),
		}
	}

	total := time.Since(start).Microseconds()
	return BatchResult{
		Results:      results,
		ModelVersion: m.GetVersion(),
		LatencyUS:    total,
	}
}
