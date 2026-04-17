package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// PredictDuration 预测延迟直方图（毫秒）
	PredictDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "adortb",
			Subsystem: "predictor",
			Name:      "predict_duration_ms",
			Help:      "预测延迟分布（毫秒）",
			Buckets:   []float64{0.1, 0.5, 1, 2, 5, 10, 20, 50},
		},
		[]string{"endpoint"},
	)

	// PredictTotal 预测总次数
	PredictTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "adortb",
			Subsystem: "predictor",
			Name:      "predict_total",
			Help:      "预测请求总次数",
		},
		[]string{"endpoint", "status"},
	)

	// ModelReloadTotal 模型重载次数
	ModelReloadTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "adortb",
			Subsystem: "predictor",
			Name:      "model_reload_total",
			Help:      "模型重载次数",
		},
		[]string{"status"},
	)

	// ActiveModelVersion 当前模型版本（gauge 用 label 表示版本）
	ActiveModelVersion = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "adortb",
			Subsystem: "predictor",
			Name:      "active_model_version",
			Help:      "当前活跃模型版本",
		},
		[]string{"version"},
	)
)
