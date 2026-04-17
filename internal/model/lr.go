package model

import (
	"encoding/json"
	"math"
	"os"
)

// LRModel Logistic Regression 模型
type LRModel struct {
	Weights map[string]float64 `json:"weights"`
	Bias    float64            `json:"bias"`
	Version string             `json:"version"`
}

// LoadLRFromFile 从 JSON 文件加载 LR 模型
func LoadLRFromFile(path string) (*LRModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseLRModel(data)
}

// LoadLRFromBytes 从字节切片解析 LR 模型
func LoadLRFromBytes(data []byte) (*LRModel, error) {
	return parseLRModel(data)
}

func parseLRModel(data []byte) (*LRModel, error) {
	var m LRModel
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if m.Weights == nil {
		m.Weights = make(map[string]float64)
	}
	return &m, nil
}

// Predict 对给定特征向量预测 raw score（sigmoid 之前）
func (m *LRModel) Predict(features map[string]float64) float64 {
	z := m.Bias
	for k, v := range features {
		if w, ok := m.Weights[k]; ok {
			z += w * v
		}
	}
	return sigmoid(z)
}

// Version 返回模型版本
func (m *LRModel) GetVersion() string {
	return m.Version
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
