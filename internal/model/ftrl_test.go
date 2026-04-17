package model

import (
	"encoding/json"
	"math"
	"sync"
	"testing"
)

func TestFTRLModel_PredictDefaultZero(t *testing.T) {
	m := NewFTRLModel(DefaultFTRLConfig())
	// 未训练时 z=0, n=0 → w=0, bias=0 → sigmoid(0)=0.5
	p := m.Predict(map[string]float64{"f1": 1.0})
	if math.Abs(p-0.5) > 1e-9 {
		t.Errorf("期望 0.5, 得到 %f", p)
	}
}

func TestFTRLModel_UpdateMovesPredict(t *testing.T) {
	cfg := DefaultFTRLConfig()
	cfg.L1 = 0  // 关闭 L1，方便验证收敛
	cfg.L2 = 0
	m := NewFTRLModel(cfg)

	features := map[string]float64{"f1": 1.0}

	// 正例样本训练 50 次，预测值应 > 0.6
	for i := 0; i < 50; i++ {
		m.Update(features, 1.0, 1.0)
	}
	p := m.Predict(features)
	if p <= 0.6 {
		t.Errorf("正例训练后预测值应 > 0.6，得到 %f", p)
	}

	// 负例样本训练 200 次，预测值应 < 0.5
	for i := 0; i < 200; i++ {
		m.Update(features, 0.0, 1.0)
	}
	p2 := m.Predict(features)
	if p2 >= 0.5 {
		t.Errorf("负例覆盖训练后预测值应 < 0.5，得到 %f", p2)
	}
}

func TestFTRLModel_L1Sparsity(t *testing.T) {
	cfg := DefaultFTRLConfig()
	cfg.L1 = 10.0 // 强 L1 强制稀疏
	m := NewFTRLModel(cfg)

	// 训练少量样本
	for i := 0; i < 5; i++ {
		m.Update(map[string]float64{"weak_feat": 0.01}, 1.0, 1.0)
	}
	// 弱特征应被 L1 截断为 0
	w := m.recoverWeight("weak_feat")
	if w != 0 {
		t.Errorf("强 L1 应令弱特征权重为 0，得到 %f", w)
	}
}

func TestFTRLModel_UpdateConcurrent(t *testing.T) {
	m := NewFTRLModel(DefaultFTRLConfig())
	features := map[string]float64{"f1": 1.0, "f2": 0.5}

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		label := float64(i % 2)
		go func(l float64) {
			defer wg.Done()
			m.Update(features, l, 1.0)
		}(label)
	}
	wg.Wait()

	// 并发更新后 Predict 应返回有效概率
	p := m.Predict(features)
	if p < 0 || p > 1 || math.IsNaN(p) {
		t.Errorf("并发更新后预测值无效: %f", p)
	}
}

func TestFTRLModel_Snapshot(t *testing.T) {
	m := NewFTRLModel(DefaultFTRLConfig())
	m.Update(map[string]float64{"f1": 1.0}, 1.0, 1.0)
	m.Update(map[string]float64{"f2": 0.5}, 0.0, 1.0)

	snap := m.Snapshot()
	if len(snap.State) == 0 {
		t.Error("快照中状态不应为空")
	}

	data, err := json.Marshal(snap)
	if err != nil {
		t.Fatalf("序列化快照失败: %v", err)
	}

	// 反序列化并验证预测结果一致
	m2, err := LoadFTRLFromBytes(data)
	if err != nil {
		t.Fatalf("从快照加载失败: %v", err)
	}
	features := map[string]float64{"f1": 1.0, "f2": 0.5}
	p1 := m.Predict(features)
	p2 := m2.Predict(features)
	if math.Abs(p1-p2) > 1e-9 {
		t.Errorf("快照恢复后预测不一致: %f vs %f", p1, p2)
	}
}

func TestFTRLModel_SaveLoadFile(t *testing.T) {
	m := NewFTRLModel(DefaultFTRLConfig())
	m.Update(map[string]float64{"f1": 1.0}, 1.0, 1.0)

	path := t.TempDir() + "/ftrl.json"
	if err := m.SaveToFile(path); err != nil {
		t.Fatalf("保存失败: %v", err)
	}

	m2, err := LoadFTRLFromFile(path)
	if err != nil {
		t.Fatalf("加载失败: %v", err)
	}
	if m2.GetVersion() != m.GetVersion() {
		t.Errorf("版本不匹配: %s vs %s", m2.GetVersion(), m.GetVersion())
	}
}

func TestFTRLModel_StateSize(t *testing.T) {
	m := NewFTRLModel(DefaultFTRLConfig())
	if m.StateSize() != 0 {
		t.Error("初始状态大小应为 0")
	}
	m.Update(map[string]float64{"f1": 1.0, "f2": 0.5}, 1.0, 1.0)
	if m.StateSize() == 0 {
		t.Error("更新后状态大小不应为 0")
	}
}

func TestFTRLModel_WeightSampleWeight(t *testing.T) {
	cfg := DefaultFTRLConfig()
	cfg.L1 = 0
	cfg.L2 = 0
	m1 := NewFTRLModel(cfg)
	m2 := NewFTRLModel(cfg)
	features := map[string]float64{"f1": 1.0}

	// 高权重样本应使模型变化更快
	for i := 0; i < 10; i++ {
		m1.Update(features, 1.0, 1.0)
		m2.Update(features, 1.0, 5.0)
	}
	p1 := m1.Predict(features)
	p2 := m2.Predict(features)
	if p2 <= p1 {
		t.Errorf("高权重训练应产生更高的预测值: p1=%f, p2=%f", p1, p2)
	}
}

func BenchmarkFTRLModel_Predict(b *testing.B) {
	m := NewFTRLModel(DefaultFTRLConfig())
	features := make(map[string]float64, 50)
	for i := 0; i < 50; i++ {
		features[hashFeatureKey(i)] = 1.0
	}
	// 先训练一些数据
	for i := 0; i < 100; i++ {
		m.Update(features, float64(i%2), 1.0)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			m.Predict(features)
		}
	})
}

func BenchmarkFTRLModel_Update(b *testing.B) {
	m := NewFTRLModel(DefaultFTRLConfig())
	features := make(map[string]float64, 20)
	for i := 0; i < 20; i++ {
		features[hashFeatureKey(i)] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Update(features, float64(i%2), 1.0)
	}
}

func hashFeatureKey(i int) string {
	return "f" + string(rune('0'+i%10)) + string(rune('a'+i%26))
}
