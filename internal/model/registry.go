package model

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	redisModelKey    = "model:lr_ctr:current_version"
	redisModelPrefix = "model:lr_ctr:weights:"
	pollInterval     = 30 * time.Second
)

// Registry 模型注册表，支持热更新（蓝绿切换）
type Registry struct {
	mu          sync.RWMutex
	current     atomic.Pointer[LRModel]
	calibrator  atomic.Pointer[Calibrator]
	rdb         *redis.Client
	localPath   string
	stopCh      chan struct{}
}

// NewRegistry 创建模型注册表
// rdb 为 nil 时仅使用本地文件
func NewRegistry(localPath string, rdb *redis.Client) (*Registry, error) {
	r := &Registry{
		rdb:       rdb,
		localPath: localPath,
		stopCh:    make(chan struct{}),
	}

	m, err := LoadLRFromFile(localPath)
	if err != nil {
		return nil, fmt.Errorf("加载本地模型失败: %w", err)
	}
	r.current.Store(m)
	r.calibrator.Store(DefaultCalibrator())

	return r, nil
}

// Current 返回当前活跃模型（无锁读，atomic）
func (r *Registry) Current() *LRModel {
	return r.current.Load()
}

// CurrentCalibrator 返回当前校准器
func (r *Registry) CurrentCalibrator() *Calibrator {
	return r.calibrator.Load()
}

// Reload 强制从 Redis 或本地重载模型
func (r *Registry) Reload(ctx context.Context) error {
	if r.rdb != nil {
		if err := r.reloadFromRedis(ctx); err == nil {
			return nil
		}
	}
	return r.reloadFromFile()
}

func (r *Registry) reloadFromRedis(ctx context.Context) error {
	version, err := r.rdb.Get(ctx, redisModelKey).Result()
	if err != nil {
		return err
	}

	weightsJSON, err := r.rdb.Get(ctx, redisModelPrefix+version).Bytes()
	if err != nil {
		return err
	}

	m, err := LoadLRFromBytes(weightsJSON)
	if err != nil {
		return fmt.Errorf("解析 Redis 模型失败: %w", err)
	}
	r.current.Store(m)
	return nil
}

func (r *Registry) reloadFromFile() error {
	m, err := LoadLRFromFile(r.localPath)
	if err != nil {
		return err
	}
	r.current.Store(m)
	return nil
}

// StartWatcher 启动后台热更新轮询
func (r *Registry) StartWatcher(ctx context.Context) {
	if r.rdb == nil {
		return
	}
	go func() {
		ticker := time.NewTicker(pollInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				_ = r.reloadFromRedis(ctx)
			case <-r.stopCh:
				return
			case <-ctx.Done():
				return
			}
		}
	}()
}

// Stop 停止后台轮询
func (r *Registry) Stop() {
	close(r.stopCh)
}
