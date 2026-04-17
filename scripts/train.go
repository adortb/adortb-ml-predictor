// 离线训练脚本：从 ClickHouse events 表拉取 impression/click 数据，
// 训练 LR 模型（SGD），导出 JSON 权重。
//
// 用法：
//   go run scripts/train.go \
//     -ch-addr=localhost:8123 \
//     -ch-db=adortb \
//     -days=7 \
//     -output=models/lr_ctr_v2.json

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
)

// TrainConfig 训练配置
type TrainConfig struct {
	CHAddr    string
	CHDB      string
	Days      int
	Output    string
	LR        float64 // 学习率
	Epochs    int
	BatchSize int
	HashDim   uint64
}

// Sample 训练样本
type Sample struct {
	Features map[string]float64
	Label    float64 // 1=click, 0=impression
}

// LRWeights 模型权重
type LRWeights struct {
	Version string             `json:"version"`
	Bias    float64            `json:"bias"`
	Weights map[string]float64 `json:"weights"`
}

func main() {
	cfg := parseFlags()

	log.Printf("连接 ClickHouse %s/%s，拉取近 %d 天数据...", cfg.CHAddr, cfg.CHDB, cfg.Days)

	samples, err := fetchSamples(cfg)
	if err != nil {
		log.Fatalf("拉取训练数据失败: %v", err)
	}
	log.Printf("拉取样本数: %d", len(samples))

	weights := trainLR(cfg, samples)

	if err := saveWeights(cfg.Output, weights); err != nil {
		log.Fatalf("保存模型失败: %v", err)
	}
	log.Printf("模型已保存到 %s", cfg.Output)
}

func parseFlags() TrainConfig {
	cfg := TrainConfig{}
	flag.StringVar(&cfg.CHAddr, "ch-addr", "localhost:9000", "ClickHouse TCP 地址")
	flag.StringVar(&cfg.CHDB, "ch-db", "adortb", "数据库名")
	flag.IntVar(&cfg.Days, "days", 7, "拉取最近 N 天数据")
	flag.StringVar(&cfg.Output, "output", "models/lr_ctr_trained.json", "输出路径")
	flag.Float64Var(&cfg.LR, "lr", 0.01, "学习率")
	flag.IntVar(&cfg.Epochs, "epochs", 5, "训练轮数")
	flag.IntVar(&cfg.BatchSize, "batch", 1024, "mini-batch 大小")
	cfg.HashDim = 10_000_000
	flag.Parse()
	return cfg
}

func fetchSamples(cfg TrainConfig) ([]Sample, error) {
	conn, err := clickhouse.Open(&clickhouse.Options{
		Addr: []string{cfg.CHAddr},
		Auth: clickhouse.Auth{Database: cfg.CHDB},
		DialTimeout: 10 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("连接 ClickHouse: %w", err)
	}
	defer conn.Close()

	since := time.Now().AddDate(0, 0, -cfg.Days).Format("2006-01-02")
	query := fmt.Sprintf(`
		SELECT
			event_type,
			user_id,
			age,
			geo,
			device_type,
			advertiser_id,
			creative_id,
			ad_type,
			toHour(event_time) AS hour_of_day,
			publisher_id
		FROM events
		WHERE event_time >= '%s'
		  AND event_type IN ('impression', 'click')
		LIMIT 10000000
	`, since)

	rows, err := conn.Query(context.Background(), query)
	if err != nil {
		return nil, fmt.Errorf("查询失败: %w", err)
	}
	defer rows.Close()

	var samples []Sample
	for rows.Next() {
		var (
			eventType, userID, geo, deviceType, advertiserID, creativeID, adType, publisherID string
			age, hour                                                                           int
		)
		if err := rows.Scan(&eventType, &userID, &age, &geo, &deviceType,
			&advertiserID, &creativeID, &adType, &hour, &publisherID); err != nil {
			continue
		}

		fv := buildFeatures(userID, geo, deviceType, advertiserID, creativeID, adType, publisherID, age, hour, cfg.HashDim)
		label := 0.0
		if eventType == "click" {
			label = 1.0
		}
		samples = append(samples, Sample{Features: fv, Label: label})
	}
	return samples, rows.Err()
}

func buildFeatures(userID, geo, device, advertiserID, creativeID, adType, publisherID string, age, hour int, hashDim uint64) map[string]float64 {
	fv := make(map[string]float64, 16)
	setFeat(fv, "user_geo", geo, hashDim)
	setFeat(fv, "user_device", device, hashDim)
	setFeat(fv, "ad_advertiser", advertiserID, hashDim)
	setFeat(fv, "ad_creative", creativeID, hashDim)
	setFeat(fv, "ad_type", adType, hashDim)
	setFeat(fv, "ctx_publisher", publisherID, hashDim)
	setFeat(fv, "user_age_bucket", ageBucket(age), hashDim)
	fv[fmt.Sprintf("f%d", hashStr("ctx_hour", "", hashDim))] = float64(hour) / 23.0
	return fv
}

func trainLR(cfg TrainConfig, samples []Sample) LRWeights {
	weights := make(map[string]float64)
	bias := 0.0
	rng := rand.New(rand.NewSource(42))

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		rng.Shuffle(len(samples), func(i, j int) {
			samples[i], samples[j] = samples[j], samples[i]
		})

		loss := 0.0
		for i := 0; i < len(samples); i += cfg.BatchSize {
			end := i + cfg.BatchSize
			if end > len(samples) {
				end = len(samples)
			}
			batch := samples[i:end]

			gradBias := 0.0
			gradW := make(map[string]float64)

			for _, s := range batch {
				z := bias
				for k, v := range s.Features {
					z += weights[k] * v
				}
				pred := sigmoid(z)
				diff := pred - s.Label
				loss += -s.Label*math.Log(pred+1e-9) - (1-s.Label)*math.Log(1-pred+1e-9)
				gradBias += diff
				for k, v := range s.Features {
					gradW[k] += diff * v
				}
			}

			n := float64(len(batch))
			bias -= cfg.LR * (gradBias / n)
			for k, g := range gradW {
				weights[k] -= cfg.LR * (g / n)
			}
		}
		log.Printf("Epoch %d/%d, avg loss=%.4f", epoch+1, cfg.Epochs, loss/float64(len(samples)))
	}

	return LRWeights{
		Version: fmt.Sprintf("lr_ctr_trained_%s", time.Now().Format("20060102")),
		Bias:    bias,
		Weights: weights,
	}
}

func saveWeights(path string, w LRWeights) error {
	data, err := json.MarshalIndent(w, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func ageBucket(age int) string {
	switch {
	case age < 18:
		return "u18"
	case age < 25:
		return "18-24"
	case age < 35:
		return "25-34"
	case age < 45:
		return "35-44"
	default:
		return "45+"
	}
}

func hashStr(field, value string, dim uint64) uint64 {
	raw := field + "\x00" + value
	h := uint64(14695981039346656037)
	for i := 0; i < len(raw); i++ {
		h ^= uint64(raw[i])
		h *= 1099511628211
	}
	return h % dim
}

func setFeat(fv map[string]float64, field, value string, dim uint64) {
	if value == "" {
		return
	}
	fv[fmt.Sprintf("f%d", hashStr(field, value, dim))] = 1.0
}
