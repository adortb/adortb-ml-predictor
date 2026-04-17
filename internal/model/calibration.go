package model

import (
	"encoding/json"
	"math"
)

// PlattParams Platt scaling 校准参数
// 通过后验概率映射：p_cal = sigmoid(A * score + B)
type PlattParams struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

// Calibrator Platt scaling 校准器
type Calibrator struct {
	params PlattParams
}

// NewCalibrator 创建校准器
func NewCalibrator(a, b float64) *Calibrator {
	return &Calibrator{params: PlattParams{A: a, B: b}}
}

// DefaultCalibrator 返回恒等校准器（A=1, B=0）
func DefaultCalibrator() *Calibrator {
	return NewCalibrator(1.0, 0.0)
}

// LoadCalibratorFromJSON 从 JSON 字节加载
func LoadCalibratorFromJSON(data []byte) (*Calibrator, error) {
	var p PlattParams
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, err
	}
	return &Calibrator{params: p}, nil
}

// Calibrate 对模型原始输出做 Platt scaling 校准
func (c *Calibrator) Calibrate(rawScore float64) float64 {
	// logit(rawScore) 再做线性变换，避免精度丢失
	logit := math.Log(rawScore/(1-rawScore+1e-9) + 1e-9)
	return sigmoid(c.params.A*logit + c.params.B)
}

// Params 返回校准参数（只读副本）
func (c *Calibrator) Params() PlattParams {
	return c.params
}
