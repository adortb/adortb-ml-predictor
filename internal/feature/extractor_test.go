package feature

import (
	"testing"
)

func TestExtractor_Extract(t *testing.T) {
	e := NewExtractor()

	user := UserFeatures{
		UserID:         "u001",
		Age:            28,
		Gender:         "male",
		Geo:            "CN",
		DeviceType:     "mobile",
		OS:             "iOS",
		ConnectionType: "4G",
		DMPTags:        []string{"sports", "tech", "gaming"},
		ActiveDays7d:   5,
		ClickCount7d:   20,
		BuyCount30d:    3,
	}
	ctx := ContextFeatures{
		HourOfDay:    14,
		DayOfWeek:    2,
		PublisherID:  "pub001",
		Domain:       "news.example.com",
		AdSlotType:   "banner",
		AdSlotSize:   "300x250",
		PageCategory: "tech",
	}
	ad := AdFeatures{
		AdvertiserID:  "adv001",
		CampaignID:    "cam001",
		CreativeID:    "cr001",
		AdType:        "banner",
		AdSize:        "300x250",
		AdCategory:    "tech",
		LandingDomain: "shop.example.com",
		HistoricalCTR: 0.05,
	}

	fv := e.Extract(user, ctx, ad)

	if len(fv) == 0 {
		t.Fatal("特征向量不应为空")
	}

	// 验证所有特征值在合理范围 [0, ∞)
	for k, v := range fv {
		if v < 0 {
			t.Errorf("特征 %s 值为负: %f", k, v)
		}
	}
}

func TestExtractor_DMPTagsLimit(t *testing.T) {
	e := NewExtractor()

	// 超过 maxDMPTags 的标签应被截断
	tags := make([]string, 20)
	for i := range tags {
		tags[i] = "tag_" + string(rune('a'+i))
	}

	user := UserFeatures{DMPTags: tags}
	ctx := ContextFeatures{}
	ad := AdFeatures{}

	fv := e.Extract(user, ctx, ad)

	// 只取前 maxDMPTags 个标签的哈希，不应有额外标签造成特征数量异常
	if len(fv) > 100 {
		t.Errorf("特征数量 %d 超出预期，可能没有截断 DMP 标签", len(fv))
	}
}

func TestHashKey_Distribution(t *testing.T) {
	// 验证哈希分布不产生大量碰撞
	seen := make(map[string]struct{}, 1000)
	collisions := 0

	fields := []string{"user_geo", "user_device", "ad_type", "ctx_domain"}
	values := []string{"CN", "US", "mobile", "desktop", "banner", "native", "news.com", "sport.com"}

	for _, f := range fields {
		for _, v := range values {
			k := hashKey(f, v)
			if _, ok := seen[k]; ok {
				collisions++
			}
			seen[k] = struct{}{}
		}
	}

	if collisions > 2 {
		t.Errorf("哈希碰撞过多: %d", collisions)
	}
}

func TestNormalize(t *testing.T) {
	tests := []struct {
		v, min, max float64
		want        float64
	}{
		{5, 0, 10, 0.5},
		{0, 0, 10, 0.0},
		{10, 0, 10, 1.0},
		{-1, 0, 10, 0.0}, // clamp
		{11, 0, 10, 1.0}, // clamp
		{5, 5, 5, 0.0},   // min == max
	}

	for _, tc := range tests {
		got := normalize(tc.v, tc.min, tc.max)
		if got != tc.want {
			t.Errorf("normalize(%f, %f, %f) = %f, want %f", tc.v, tc.min, tc.max, got, tc.want)
		}
	}
}

func BenchmarkExtractor_Extract(b *testing.B) {
	e := NewExtractor()
	user := UserFeatures{
		UserID: "u001", Age: 28, Gender: "male", Geo: "CN",
		DeviceType: "mobile", OS: "iOS", ConnectionType: "4G",
		DMPTags: []string{"sports", "tech", "gaming", "travel", "food"},
		ActiveDays7d: 5, ClickCount7d: 20, BuyCount30d: 3,
	}
	ctx := ContextFeatures{
		HourOfDay: 14, DayOfWeek: 2, PublisherID: "pub001",
		Domain: "news.example.com", AdSlotType: "banner",
		AdSlotSize: "300x250", PageCategory: "tech",
	}
	ad := AdFeatures{
		AdvertiserID: "adv001", CampaignID: "cam001", CreativeID: "cr001",
		AdType: "banner", AdSize: "300x250", AdCategory: "tech",
		LandingDomain: "shop.example.com", HistoricalCTR: 0.05,
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			e.Extract(user, ctx, ad)
		}
	})
}
