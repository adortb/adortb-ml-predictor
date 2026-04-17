package model

import (
	"math"
	"testing"
)

func TestCalibrator_Calibrate(t *testing.T) {
	tests := []struct {
		name      string
		a, b      float64
		rawScore  float64
		wantRange [2]float64
	}{
		{
			name:      "恒等校准器（A=1,B=0）应接近输入",
			a:         1.0, b: 0.0,
			rawScore:  0.5,
			wantRange: [2]float64{0.4, 0.6},
		},
		{
			name:      "压缩校准（A<1）",
			a:         0.5, b: 0.0,
			rawScore:  0.8,
			wantRange: [2]float64{0.6, 0.8},
		},
		{
			name:      "偏移校准（B<0）",
			a:         1.0, b: -1.0,
			rawScore:  0.5,
			wantRange: [2]float64{0.1, 0.4},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := NewCalibrator(tc.a, tc.b)
			got := c.Calibrate(tc.rawScore)
			if got < tc.wantRange[0] || got > tc.wantRange[1] {
				t.Errorf("Calibrate(%f) = %f, want [%f, %f]", tc.rawScore, got, tc.wantRange[0], tc.wantRange[1])
			}
		})
	}
}

func TestDefaultCalibrator(t *testing.T) {
	c := DefaultCalibrator()
	p := c.Params()
	if p.A != 1.0 || p.B != 0.0 {
		t.Errorf("DefaultCalibrator params = {%f, %f}, want {1.0, 0.0}", p.A, p.B)
	}
}

func TestCalibrator_OutputRange(t *testing.T) {
	c := NewCalibrator(1.0, 0.0)
	// 输出必须在 (0, 1) 内
	for _, score := range []float64{0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999} {
		got := c.Calibrate(score)
		if math.IsNaN(got) || got <= 0 || got >= 1 {
			t.Errorf("Calibrate(%f) = %f, 必须在 (0,1) 内", score, got)
		}
	}
}
