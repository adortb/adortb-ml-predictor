package feature

import (
	"fmt"
	"math"

	"github.com/cespare/xxhash/v2"
)

const (
	// HashDim 特征哈希维度（约 10^7）
	HashDim = 10_000_000
	// maxDMPTags 最多取前 N 个 DMP 标签
	maxDMPTags = 10
)

// Extractor 特征提取器
type Extractor struct{}

// NewExtractor 创建特征提取器
func NewExtractor() *Extractor {
	return &Extractor{}
}

// Extract 从原始特征提取模型输入向量
func (e *Extractor) Extract(user UserFeatures, ctx ContextFeatures, ad AdFeatures) FeatureVector {
	fv := make(FeatureVector, 64)

	e.extractUserFeatures(fv, user)
	e.extractContextFeatures(fv, ctx)
	e.extractAdFeatures(fv, ad)
	e.extractCrossFeatures(fv, user, ctx, ad)

	return fv
}

func (e *Extractor) extractUserFeatures(fv FeatureVector, user UserFeatures) {
	hashSet(fv, "user_age_bucket", ageBucket(user.Age))
	hashSet(fv, "user_gender", user.Gender)
	hashSet(fv, "user_geo", user.Geo)
	hashSet(fv, "user_device", user.DeviceType)
	hashSet(fv, "user_os", user.OS)
	hashSet(fv, "user_conn", user.ConnectionType)

	tags := user.DMPTags
	if len(tags) > maxDMPTags {
		tags = tags[:maxDMPTags]
	}
	for _, tag := range tags {
		hashAdd(fv, "user_dmp_tag", tag)
	}

	// 连续特征归一化
	fv[hashKey("user_active_days_7d", "")] = normalize(float64(user.ActiveDays7d), 0, 30)
	fv[hashKey("user_click_count_7d", "")] = normalize(float64(user.ClickCount7d), 0, 100)
	fv[hashKey("user_buy_count_30d", "")] = normalize(float64(user.BuyCount30d), 0, 50)
}

func (e *Extractor) extractContextFeatures(fv FeatureVector, ctx ContextFeatures) {
	fv[hashKey("ctx_hour", "")] = float64(ctx.HourOfDay) / 23.0
	fv[hashKey("ctx_dow", "")] = float64(ctx.DayOfWeek) / 6.0
	hashSet(fv, "ctx_publisher", ctx.PublisherID)
	hashSet(fv, "ctx_domain", ctx.Domain)
	hashSet(fv, "ctx_slot_type", ctx.AdSlotType)
	hashSet(fv, "ctx_slot_size", ctx.AdSlotSize)
	hashSet(fv, "ctx_page_cat", ctx.PageCategory)
}

func (e *Extractor) extractAdFeatures(fv FeatureVector, ad AdFeatures) {
	hashSet(fv, "ad_advertiser", ad.AdvertiserID)
	hashSet(fv, "ad_campaign", ad.CampaignID)
	hashSet(fv, "ad_creative", ad.CreativeID)
	hashSet(fv, "ad_type", ad.AdType)
	hashSet(fv, "ad_size", ad.AdSize)
	hashSet(fv, "ad_category", ad.AdCategory)
	hashSet(fv, "ad_landing", ad.LandingDomain)
	fv[hashKey("ad_hist_ctr", "")] = clamp(ad.HistoricalCTR, 0, 1)
}

func (e *Extractor) extractCrossFeatures(fv FeatureVector, user UserFeatures, ctx ContextFeatures, ad AdFeatures) {
	// 用户 × 广告类型
	hashSet(fv, "cross_geo_adtype", user.Geo+"_"+ad.AdType)
	// 域名 × 广告主（过去点击率代理）
	hashSet(fv, "cross_domain_adv", ctx.Domain+"_"+ad.AdvertiserID)
	// 小时 × 广告类型
	hashSet(fv, "cross_hour_adtype", fmt.Sprintf("%d_%s", ctx.HourOfDay, ad.AdType))
	// 设备 × 广告大小
	hashSet(fv, "cross_device_size", user.DeviceType+"_"+ad.AdSize)
}

// hashKey 将 (field, value) 哈希到固定维度 key
func hashKey(field, value string) string {
	raw := field + "\x00" + value
	h := xxhash.Sum64String(raw) % HashDim
	return fmt.Sprintf("f%d", h)
}

// hashSet 设置二值特征
func hashSet(fv FeatureVector, field, value string) {
	if value == "" {
		return
	}
	fv[hashKey(field, value)] = 1.0
}

// hashAdd 累加特征（多值）
func hashAdd(fv FeatureVector, field, value string) {
	if value == "" {
		return
	}
	fv[hashKey(field, value)] += 1.0
}

// ageBucket 年龄分桶
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
	case age < 55:
		return "45-54"
	default:
		return "55+"
	}
}

// normalize 线性归一化到 [0,1]
func normalize(v, min, max float64) float64 {
	if max == min {
		return 0
	}
	return clamp((v-min)/(max-min), 0, 1)
}

// clamp 裁剪到 [lo, hi]
func clamp(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}
