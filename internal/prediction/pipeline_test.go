package prediction

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/adortb/adortb-ml-predictor/internal/feature"
	"github.com/adortb/adortb-ml-predictor/internal/model"
)

func makeTestRegistry(t *testing.T) *model.Registry {
	t.Helper()
	m := map[string]any{
		"version": "test_v1",
		"bias":    -1.5,
		"weights": map[string]float64{
			"f1": 0.8,
			"f2": 0.3,
		},
	}
	data, _ := json.Marshal(m)
	tmpFile := filepath.Join(t.TempDir(), "model.json")
	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		t.Fatal(err)
	}
	reg, err := model.NewRegistry(tmpFile, nil)
	if err != nil {
		t.Fatal(err)
	}
	return reg
}

func TestPipeline_Predict(t *testing.T) {
	reg := makeTestRegistry(t)
	p := NewPipeline(reg)

	req := Request{
		User: feature.UserFeatures{
			UserID: "u001", Age: 28, Geo: "CN",
			DeviceType: "mobile", OS: "iOS",
			ActiveDays7d: 5, ClickCount7d: 10, BuyCount30d: 2,
		},
		Context: feature.ContextFeatures{
			HourOfDay: 14, DayOfWeek: 2,
			PublisherID: "pub001", Domain: "news.example.com",
			AdSlotType: "banner", AdSlotSize: "300x250",
		},
		Ad: feature.AdFeatures{
			AdvertiserID: "adv001", CreativeID: "cr001",
			AdType: "banner", AdSize: "300x250",
			HistoricalCTR: 0.05,
		},
	}

	result := p.Predict(req)

	if result.PCTR < 0 || result.PCTR > 1 {
		t.Errorf("pCTR = %f, 必须在 [0,1]", result.PCTR)
	}
	if result.ModelVersion == "" {
		t.Error("ModelVersion 不应为空")
	}
	if result.LatencyUS < 0 {
		t.Error("LatencyUS 不应为负")
	}
}

func TestPipeline_PredictBatch(t *testing.T) {
	reg := makeTestRegistry(t)
	p := NewPipeline(reg)

	ads := []feature.AdFeatures{
		{AdvertiserID: "adv001", CreativeID: "cr001", AdType: "banner", HistoricalCTR: 0.05},
		{AdvertiserID: "adv002", CreativeID: "cr002", AdType: "native", HistoricalCTR: 0.03},
		{AdvertiserID: "adv003", CreativeID: "cr003", AdType: "banner", HistoricalCTR: 0.08},
	}

	req := BatchRequest{
		User:    feature.UserFeatures{UserID: "u001", Age: 28, Geo: "CN", DeviceType: "mobile"},
		Context: feature.ContextFeatures{HourOfDay: 14, PublisherID: "pub001", Domain: "news.example.com"},
		Ads:     ads,
	}

	result := p.PredictBatch(req)

	if len(result.Results) != len(ads) {
		t.Errorf("批量结果数 = %d, want %d", len(result.Results), len(ads))
	}
	for i, r := range result.Results {
		if r.PCTR < 0 || r.PCTR > 1 {
			t.Errorf("result[%d].PCTR = %f, 必须在 [0,1]", i, r.PCTR)
		}
	}
}

func BenchmarkPipeline_Predict(b *testing.B) {
	m := map[string]any{
		"version": "bench_v1",
		"bias":    -1.5,
		"weights": map[string]float64{"f1": 0.5},
	}
	data, _ := json.Marshal(m)
	tmpFile := filepath.Join(b.TempDir(), "model.json")
	os.WriteFile(tmpFile, data, 0644)
	reg, _ := model.NewRegistry(tmpFile, nil)
	p := NewPipeline(reg)

	req := Request{
		User:    feature.UserFeatures{UserID: "u001", Age: 28, Geo: "CN", DeviceType: "mobile"},
		Context: feature.ContextFeatures{HourOfDay: 14, PublisherID: "pub001", Domain: "news.example.com"},
		Ad:      feature.AdFeatures{AdvertiserID: "adv001", CreativeID: "cr001", AdType: "banner", HistoricalCTR: 0.05},
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p.Predict(req)
		}
	})
}
