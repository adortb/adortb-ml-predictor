package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestLRModel_Predict(t *testing.T) {
	m := &LRModel{
		Weights: map[string]float64{
			"f1": 1.0,
			"f2": -0.5,
		},
		Bias:    -0.5,
		Version: "test_v1",
	}

	tests := []struct {
		name     string
		features map[string]float64
		wantMin  float64
		wantMax  float64
	}{
		{
			name:     "正常特征",
			features: map[string]float64{"f1": 1.0, "f2": 1.0},
			wantMin:  0.3,
			wantMax:  0.7,
		},
		{
			name:     "空特征",
			features: map[string]float64{},
			wantMin:  0.0,
			wantMax:  0.5, // sigmoid(-0.5) ≈ 0.378
		},
		{
			name:     "高权重特征",
			features: map[string]float64{"f1": 5.0},
			wantMin:  0.9,
			wantMax:  1.0,
		},
		{
			name:     "未知特征应被忽略",
			features: map[string]float64{"unknown_key": 100.0},
			wantMin:  0.0,
			wantMax:  0.5,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := m.Predict(tc.features)
			if got < tc.wantMin || got > tc.wantMax {
				t.Errorf("Predict() = %f, want [%f, %f]", got, tc.wantMin, tc.wantMax)
			}
		})
	}
}

func TestLoadLRFromFile(t *testing.T) {
	// 创建临时模型文件
	m := LRModel{
		Weights: map[string]float64{"f1": 0.5, "f2": -0.3},
		Bias:    -1.0,
		Version: "file_v1",
	}
	data, _ := json.Marshal(m)

	tmpFile := filepath.Join(t.TempDir(), "model.json")
	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadLRFromFile(tmpFile)
	if err != nil {
		t.Fatalf("LoadLRFromFile() error = %v", err)
	}

	if loaded.Version != m.Version {
		t.Errorf("version = %s, want %s", loaded.Version, m.Version)
	}
	if loaded.Bias != m.Bias {
		t.Errorf("bias = %f, want %f", loaded.Bias, m.Bias)
	}
	if len(loaded.Weights) != len(m.Weights) {
		t.Errorf("weights len = %d, want %d", len(loaded.Weights), len(m.Weights))
	}
}

func TestLoadLRFromFile_NotFound(t *testing.T) {
	_, err := LoadLRFromFile("/nonexistent/path/model.json")
	if err == nil {
		t.Error("期望返回错误，但没有")
	}
}

func TestLRModel_PredictConsistency(t *testing.T) {
	m := &LRModel{
		Weights: map[string]float64{"f1": 0.8, "f2": 0.3},
		Bias:    -1.0,
		Version: "v1",
	}
	features := map[string]float64{"f1": 1.0, "f2": 0.5}

	// 相同输入应产生相同输出（确定性）
	first := m.Predict(features)
	for i := 0; i < 10; i++ {
		if got := m.Predict(features); got != first {
			t.Errorf("预测不一致: iter %d, got %f, want %f", i, got, first)
		}
	}
}

func BenchmarkLRModel_Predict(b *testing.B) {
	// 构造 100 维特征模型
	weights := make(map[string]float64, 100)
	for i := 0; i < 100; i++ {
		weights[fmt.Sprintf("f%d", i)] = 0.01 * float64(i)
	}
	m := &LRModel{Weights: weights, Bias: -1.0, Version: "bench_v1"}

	features := make(map[string]float64, 50)
	for i := 0; i < 50; i++ {
		features[fmt.Sprintf("f%d", i)] = 1.0
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			m.Predict(features)
		}
	})
}
