package trainer

import (
	"math"
	"testing"
	"time"
)

func TestNextTrigger(t *testing.T) {
	base := time.Date(2026, 1, 1, 10, 0, 0, 0, time.UTC)

	// 当天 03:00 已过（现在是 10:00），应推到明天
	got := nextTrigger(base, 3, 0)
	want := time.Date(2026, 1, 2, 3, 0, 0, 0, time.UTC)
	if !got.Equal(want) {
		t.Errorf("got %v, want %v", got, want)
	}

	// 当天 14:00 未过，应返回当天
	got2 := nextTrigger(base, 14, 0)
	want2 := time.Date(2026, 1, 1, 14, 0, 0, 0, time.UTC)
	if !got2.Equal(want2) {
		t.Errorf("got %v, want %v", got2, want2)
	}
}

func TestNextTrigger_ExactTime(t *testing.T) {
	// 恰好等于触发时间，应推到明天
	now := time.Date(2026, 1, 1, 3, 0, 0, 0, time.UTC)
	got := nextTrigger(now, 3, 0)
	if got.Before(now) || got.Equal(now) {
		t.Errorf("恰好等于触发时间时应推到明天，got %v", got)
	}
}

func TestSigmoid(t *testing.T) {
	cases := []struct{ in, wantApprox, tol float64 }{
		{0, 0.5, 1e-9},
		{10, 0.9999546, 1e-5},
		{-10, 0.0000454, 1e-5},
		{100, 1.0, 1e-9},
		{-100, 0.0, 1e-9},
	}
	for _, c := range cases {
		got := sigmoid(c.in)
		if math.Abs(got-c.wantApprox) > c.tol {
			t.Errorf("sigmoid(%v) = %v, want ≈ %v", c.in, got, c.wantApprox)
		}
	}
}

func TestDefaultTrainConfig(t *testing.T) {
	cfg := DefaultTrainConfig()
	if cfg.LearningRate <= 0 {
		t.Error("学习率必须为正")
	}
	if cfg.MaxEpochs <= 0 {
		t.Error("训练轮数必须为正")
	}
	if cfg.BatchSize <= 0 {
		t.Error("批大小必须为正")
	}
	if cfg.MinSamples <= 0 {
		t.Error("最小样本数必须为正")
	}
}
