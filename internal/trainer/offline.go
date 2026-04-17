// Package trainer 提供离线批量训练功能。
// 每天 03:00 从 ClickHouse 拉取前一天 impression+click 数据，训练 LR 模型，写入 model_repo。
package trainer

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	_ "github.com/ClickHouse/clickhouse-go/v2"

	"github.com/adortb/adortb-ml-predictor/internal/feature"
	"github.com/adortb/adortb-ml-predictor/internal/model"
	"github.com/adortb/adortb-ml-predictor/internal/store"
)

const (
	defaultLearningRate = 0.01
	defaultMaxEpochs    = 3
	defaultBatchSize    = 512
)

// TrainConfig 离线训练配置
type TrainConfig struct {
	LearningRate float64
	MaxEpochs    int
	BatchSize    int
	MinSamples   int // 样本数不足时跳过训练
}

// DefaultTrainConfig 返回默认配置
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		LearningRate: defaultLearningRate,
		MaxEpochs:    defaultMaxEpochs,
		BatchSize:    defaultBatchSize,
		MinSamples:   1000,
	}
}

// sample 单条训练样本
type sample struct {
	features map[string]float64
	label    float64
}

// OfflineTrainer 离线训练器
type OfflineTrainer struct {
	db        *sql.DB
	repo      *store.Repo
	extractor *feature.Extractor
	cfg       TrainConfig
	mu        sync.Mutex // 防止并发训练
}

// NewOfflineTrainer 创建离线训练器
func NewOfflineTrainer(db *sql.DB, repo *store.Repo, cfg TrainConfig) *OfflineTrainer {
	return &OfflineTrainer{
		db:        db,
		repo:      repo,
		extractor: feature.NewExtractor(),
		cfg:       cfg,
	}
}

// TrainForDate 对指定日期执行一次全量训练（幂等）
func (t *OfflineTrainer) TrainForDate(ctx context.Context, date time.Time) (*store.ModelMeta, error) {
	if !t.mu.TryLock() {
		return nil, fmt.Errorf("已有训练任务在执行中")
	}
	defer t.mu.Unlock()

	day := date.Format("2006-01-02")
	log.Printf("[trainer] 开始训练 date=%s", day)

	samples, err := t.fetchSamples(ctx, day)
	if err != nil {
		return nil, fmt.Errorf("拉取训练数据失败: %w", err)
	}
	if len(samples) < t.cfg.MinSamples {
		return nil, fmt.Errorf("样本数 %d < 最低要求 %d，跳过训练", len(samples), t.cfg.MinSamples)
	}
	log.Printf("[trainer] 加载样本 %d 条", len(samples))

	m, version := t.trainLR(samples)
	data, err := marshalLR(m, version)
	if err != nil {
		return nil, err
	}

	meta := store.ModelMeta{
		Name:      "lr_ctr",
		Version:   version,
		Type:      store.ModelTypeLR,
		Samples:   int64(len(samples)),
		CreatedAt: time.Now(),
	}
	if err := t.repo.Save(meta, data); err != nil {
		return nil, fmt.Errorf("保存模型失败: %w", err)
	}
	log.Printf("[trainer] 训练完成 version=%s samples=%d", version, len(samples))
	return &meta, nil
}

// StartScheduler 在 ctx 存活期间每天 03:00 自动触发训练
func (t *OfflineTrainer) StartScheduler(ctx context.Context) {
	go func() {
		for {
			next := nextTrigger(time.Now(), 3, 0)
			log.Printf("[trainer] 下次训练时间: %s", next.Format(time.RFC3339))
			select {
			case <-time.After(time.Until(next)):
				yesterday := time.Now().AddDate(0, 0, -1)
				if _, err := t.TrainForDate(ctx, yesterday); err != nil {
					log.Printf("[trainer] 自动训练失败: %v", err)
				}
			case <-ctx.Done():
				return
			}
		}
	}()
}

// fetchSamples 从 ClickHouse 拉取指定日期的训练样本
// 预期表结构：impressions(date, creative_id, campaign_id, ad_type, ad_size,
//
//	geo, device_type, os, hour, publisher_id, domain, is_click)
func (t *OfflineTrainer) fetchSamples(ctx context.Context, day string) ([]sample, error) {
	query := `
		SELECT
			creative_id, campaign_id, ad_type, ad_size,
			geo, device_type, os, toHour(impression_time) AS hour,
			publisher_id, domain, is_click
		FROM impressions
		WHERE toDate(impression_time) = ?
		ORDER BY rand()
		LIMIT 5000000
	`
	rows, err := t.db.QueryContext(ctx, query, day)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var samples []sample
	for rows.Next() {
		var (
			creativeID, campaignID, adType, adSize string
			geo, deviceType, os, publisherID, domain string
			hour                                     int
			isClick                                  uint8
		)
		if err := rows.Scan(
			&creativeID, &campaignID, &adType, &adSize,
			&geo, &deviceType, &os, &hour,
			&publisherID, &domain, &isClick,
		); err != nil {
			return nil, err
		}
		userFeats := feature.UserFeatures{
			Geo:        geo,
			DeviceType: deviceType,
			OS:         os,
		}
		ctxFeats := feature.ContextFeatures{
			HourOfDay:   hour,
			DayOfWeek:   0, // 简化版：不区分星期
			PublisherID: publisherID,
			Domain:      domain,
		}
		adFeats := feature.AdFeatures{
			CreativeID: creativeID,
			CampaignID: campaignID,
			AdType:     adType,
			AdSize:     adSize,
		}
		fv := t.extractor.Extract(userFeats, ctxFeats, adFeats)
		samples = append(samples, sample{features: fv, label: float64(isClick)})
	}
	return samples, rows.Err()
}

// trainLR 用 mini-batch SGD 训练 LR 模型
func (t *OfflineTrainer) trainLR(samples []sample) (*model.LRModel, string) {
	weights := make(map[string]float64)
	var bias float64
	lr := t.cfg.LearningRate

	for epoch := 0; epoch < t.cfg.MaxEpochs; epoch++ {
		rand.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })
		for start := 0; start < len(samples); start += t.cfg.BatchSize {
			end := start + t.cfg.BatchSize
			if end > len(samples) {
				end = len(samples)
			}
			batch := samples[start:end]
			gradW := make(map[string]float64, len(batch)*10)
			var gradB float64

			for _, s := range batch {
				score := bias
				for k, v := range s.features {
					score += weights[k] * v
				}
				pred := sigmoid(score)
				diff := pred - s.label
				gradB += diff
				for k, v := range s.features {
					gradW[k] += diff * v
				}
			}
			n := float64(len(batch))
			bias -= lr * gradB / n
			for k, g := range gradW {
				weights[k] -= lr * g / n
			}
		}
	}

	version := fmt.Sprintf("lr_ctr_%s", time.Now().Format("20060102_150405"))
	return &model.LRModel{
		Weights: weights,
		Bias:    bias,
		Version: version,
	}, version
}

func marshalLR(m *model.LRModel, version string) ([]byte, error) {
	type lrJSON struct {
		Version string             `json:"version"`
		Bias    float64            `json:"bias"`
		Weights map[string]float64 `json:"weights"`
	}
	return json.Marshal(lrJSON{Version: version, Bias: m.Bias, Weights: m.Weights})
}

// nextTrigger 计算下一次触发时间（当天 hour:minute，若已过则推到明天）
func nextTrigger(now time.Time, hour, minute int) time.Time {
	t := time.Date(now.Year(), now.Month(), now.Day(), hour, minute, 0, 0, now.Location())
	if t.Before(now) || t.Equal(now) {
		t = t.Add(24 * time.Hour)
	}
	return t
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
