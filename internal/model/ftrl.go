package model

import (
	"encoding/json"
	"math"
	"os"
	"sync"
)

// FTRLConfig FTRL-Proximal 超参数
type FTRLConfig struct {
	Alpha   float64 `json:"alpha"`    // 学习率分子（典型 0.1）
	Beta    float64 `json:"beta"`     // 学习率平滑（典型 1.0）
	L1      float64 `json:"l1"`       // L1 正则化强度
	L2      float64 `json:"l2"`       // L2 正则化强度
	HashDim int     `json:"hash_dim"` // 特征哈希维度
	Version string  `json:"version"`
}

// DefaultFTRLConfig 默认超参数
func DefaultFTRLConfig() FTRLConfig {
	return FTRLConfig{
		Alpha:   0.1,
		Beta:    1.0,
		L1:      0.1,
		L2:      1.0,
		HashDim: 1_000_000,
		Version: "ftrl_v1",
	}
}

// ftrlState 稀疏存储的 z/n 两个辅助向量（用 map 节省内存）
type ftrlState struct {
	z float64 // 累积梯度（带 L1 截断）
	n float64 // 累积梯度平方和
}

// FTRLModel FTRL-Proximal 在线学习模型
// 论文：McMahan et al., "Ad Click Prediction: a View from the Trenches" (KDD 2013)
type FTRLModel struct {
	mu     sync.RWMutex
	cfg    FTRLConfig
	state  map[string]*ftrlState // 稀疏存储，仅保存非零特征
	bias   float64
	biasZ  float64
	biasN  float64
}

// FTRLSnapshot 用于持久化的快照（JSON 序列化）
type FTRLSnapshot struct {
	Config FTRLConfig              `json:"config"`
	State  map[string][2]float64   `json:"state"` // key -> [z, n]
	Bias   float64                 `json:"bias"`
	BiasZ  float64                 `json:"bias_z"`
	BiasN  float64                 `json:"bias_n"`
}

// NewFTRLModel 创建 FTRL 模型
func NewFTRLModel(cfg FTRLConfig) *FTRLModel {
	return &FTRLModel{
		cfg:   cfg,
		state: make(map[string]*ftrlState, 1024),
	}
}

// LoadFTRLFromFile 从文件加载快照
func LoadFTRLFromFile(path string) (*FTRLModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadFTRLFromBytes(data)
}

// LoadFTRLFromBytes 从字节切片加载快照
func LoadFTRLFromBytes(data []byte) (*FTRLModel, error) {
	var snap FTRLSnapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return nil, err
	}
	m := &FTRLModel{
		cfg:   snap.Config,
		state: make(map[string]*ftrlState, len(snap.State)),
		bias:  snap.Bias,
		biasZ: snap.BiasZ,
		biasN: snap.BiasN,
	}
	for k, v := range snap.State {
		m.state[k] = &ftrlState{z: v[0], n: v[1]}
	}
	return m, nil
}

// GetVersion 返回模型版本
func (m *FTRLModel) GetVersion() string {
	return m.cfg.Version
}

// Predict 推理：从 z/n 恢复权重后计算线性得分 + sigmoid
func (m *FTRLModel) Predict(features map[string]float64) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	z := m.recoverBiasWeight()
	for feat, val := range features {
		w := m.recoverWeight(feat)
		z += w * val
	}
	return sigmoid(z)
}

// Update 在线学习：给定样本 (features, label∈{0,1}, weight) 做一步 FTRL-Proximal 更新
func (m *FTRLModel) Update(features map[string]float64, label, sampleWeight float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 前向推理（不加锁，已在写锁中）
	score := m.recoverBiasWeight()
	for feat, val := range features {
		score += m.recoverWeight(feat) * val
	}
	pred := sigmoid(score)
	grad := sampleWeight * (pred - label)

	// 更新 bias
	m.updateCoord(&m.biasZ, &m.biasN, grad, 1.0, true)

	// 更新各特征坐标
	for feat, val := range features {
		s := m.getOrCreate(feat)
		m.updateCoord(&s.z, &s.n, grad*val, val, false)
	}
}

// updateCoord 更新单个坐标的 z/n，isBias=true 时跳过 L1 截断
func (m *FTRLModel) updateCoord(z, n *float64, g, x float64, isBias bool) {
	sigma := (math.Sqrt(*n+g*g) - math.Sqrt(*n)) / m.cfg.Alpha
	*z += g - sigma*x
	*n += g * g
}

// recoverWeight 从 z/n 恢复特征权重 w_i（FTRL-Proximal 公式）
// w_i = -(z_i - sign(z_i)*L1) / (L2 + (beta + sqrt(n_i)) / alpha)
// 当 |z_i| <= L1 时，w_i = 0（稀疏性）
func (m *FTRLModel) recoverWeight(feat string) float64 {
	s, ok := m.state[feat]
	if !ok {
		return 0
	}
	if math.Abs(s.z) <= m.cfg.L1 {
		return 0
	}
	sign := 1.0
	if s.z < 0 {
		sign = -1.0
	}
	return -(s.z - sign*m.cfg.L1) / (m.cfg.L2 + (m.cfg.Beta+math.Sqrt(s.n))/m.cfg.Alpha)
}

// recoverBiasWeight 恢复 bias（不做 L1 截断）
func (m *FTRLModel) recoverBiasWeight() float64 {
	denom := m.cfg.L2 + (m.cfg.Beta+math.Sqrt(m.biasN))/m.cfg.Alpha
	if denom == 0 {
		return 0
	}
	return -m.biasZ / denom
}

func (m *FTRLModel) getOrCreate(feat string) *ftrlState {
	s, ok := m.state[feat]
	if !ok {
		s = &ftrlState{}
		m.state[feat] = s
	}
	return s
}

// Snapshot 导出快照（用于持久化）
func (m *FTRLModel) Snapshot() FTRLSnapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()

	snap := FTRLSnapshot{
		Config: m.cfg,
		State:  make(map[string][2]float64, len(m.state)),
		Bias:   m.bias,
		BiasZ:  m.biasZ,
		BiasN:  m.biasN,
	}
	for k, v := range m.state {
		snap.State[k] = [2]float64{v.z, v.n}
	}
	return snap
}

// SaveToFile 将快照写入文件
func (m *FTRLModel) SaveToFile(path string) error {
	snap := m.Snapshot()
	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// StateSize 返回稀疏状态中的非零特征数量
func (m *FTRLModel) StateSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.state)
}
