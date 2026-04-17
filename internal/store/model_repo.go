// Package store 提供模型版本管理，支持 LR/FTRL 两种模型的持久化和版本追踪。
package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// ModelType 模型类型
type ModelType string

const (
	ModelTypeLR   ModelType = "lr"
	ModelTypeFTRL ModelType = "ftrl"
)

// ModelMeta 模型元数据
type ModelMeta struct {
	Name      string    `json:"name"`
	Version   string    `json:"version"`
	Type      ModelType `json:"type"`
	Path      string    `json:"path"`
	CreatedAt time.Time `json:"created_at"`
	AUC       float64   `json:"auc,omitempty"`
	Samples   int64     `json:"samples,omitempty"`
}

// Repo 模型版本仓库（本地文件系统）
type Repo struct {
	mu      sync.RWMutex
	baseDir string
	index   map[string]*ModelMeta // version -> meta
}

// NewRepo 创建模型仓库，baseDir 若不存在则自动创建
func NewRepo(baseDir string) (*Repo, error) {
	if err := os.MkdirAll(baseDir, 0o755); err != nil {
		return nil, fmt.Errorf("创建仓库目录失败: %w", err)
	}
	r := &Repo{
		baseDir: baseDir,
		index:   make(map[string]*ModelMeta),
	}
	if err := r.loadIndex(); err != nil {
		return nil, err
	}
	return r, nil
}

// Save 保存模型数据（JSON bytes）并注册元数据
func (r *Repo) Save(meta ModelMeta, data []byte) error {
	if meta.Version == "" {
		return fmt.Errorf("版本号不能为空")
	}
	fname := fmt.Sprintf("%s_%s.json", meta.Type, meta.Version)
	path := filepath.Join(r.baseDir, fname)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("写入模型文件失败: %w", err)
	}
	meta.Path = path
	if meta.CreatedAt.IsZero() {
		meta.CreatedAt = time.Now()
	}

	r.mu.Lock()
	r.index[meta.Version] = &meta
	r.mu.Unlock()

	return r.saveIndex()
}

// Get 获取指定版本元数据
func (r *Repo) Get(version string) (*ModelMeta, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.index[version]
	if !ok {
		return nil, false
	}
	cp := *m
	return &cp, true
}

// Latest 返回最新版本元数据（按 CreatedAt 降序取第一个）
func (r *Repo) Latest(modelType ModelType) (*ModelMeta, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var candidates []*ModelMeta
	for _, m := range r.index {
		if m.Type == modelType {
			candidates = append(candidates, m)
		}
	}
	if len(candidates) == 0 {
		return nil, false
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].CreatedAt.After(candidates[j].CreatedAt)
	})
	cp := *candidates[0]
	return &cp, true
}

// List 列出所有模型（按时间降序）
func (r *Repo) List() []ModelMeta {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]ModelMeta, 0, len(r.index))
	for _, m := range r.index {
		out = append(out, *m)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	return out
}

// ReadData 读取指定版本的模型文件字节
func (r *Repo) ReadData(version string) ([]byte, error) {
	meta, ok := r.Get(version)
	if !ok {
		return nil, fmt.Errorf("版本 %s 不存在", version)
	}
	return os.ReadFile(meta.Path)
}

// indexPath 返回索引文件路径
func (r *Repo) indexPath() string {
	return filepath.Join(r.baseDir, "index.json")
}

func (r *Repo) loadIndex() error {
	data, err := os.ReadFile(r.indexPath())
	if os.IsNotExist(err) {
		return nil // 空仓库
	}
	if err != nil {
		return fmt.Errorf("读取模型索引失败: %w", err)
	}
	var metas []ModelMeta
	if err := json.Unmarshal(data, &metas); err != nil {
		return fmt.Errorf("解析模型索引失败: %w", err)
	}
	for i := range metas {
		r.index[metas[i].Version] = &metas[i]
	}
	return nil
}

func (r *Repo) saveIndex() error {
	r.mu.RLock()
	metas := make([]ModelMeta, 0, len(r.index))
	for _, m := range r.index {
		metas = append(metas, *m)
	}
	r.mu.RUnlock()

	data, err := json.MarshalIndent(metas, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(r.indexPath(), data, 0o644)
}
