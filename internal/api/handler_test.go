package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/adortb/adortb-ml-predictor/internal/feature"
	"github.com/adortb/adortb-ml-predictor/internal/model"
	"github.com/adortb/adortb-ml-predictor/internal/prediction"
)

func makeTestHandler(t *testing.T) *Handler {
	t.Helper()
	m := map[string]any{
		"version": "handler_test_v1",
		"bias":    -1.5,
		"weights": map[string]float64{"f1": 0.5},
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
	pipeline := prediction.NewPipeline(reg)
	ftrl := model.NewFTRLModel(model.DefaultFTRLConfig())
	return NewHandler(pipeline, reg, ftrl)
}

func TestHandlePredictCTR_OK(t *testing.T) {
	h := makeTestHandler(t)

	reqBody := prediction.Request{
		User:    feature.UserFeatures{UserID: "u001", Age: 28, Geo: "CN"},
		Context: feature.ContextFeatures{HourOfDay: 14, Domain: "news.example.com"},
		Ad:      feature.AdFeatures{AdvertiserID: "adv001", CreativeID: "cr001", AdType: "banner"},
	}
	data, _ := json.Marshal(reqBody)

	r := httptest.NewRequest(http.MethodPost, "/v1/predict/ctr", bytes.NewReader(data))
	w := httptest.NewRecorder()
	h.HandlePredictCTR(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var result prediction.Result
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("解析响应失败: %v", err)
	}
	if result.PCTR < 0 || result.PCTR > 1 {
		t.Errorf("pCTR = %f, 必须在 [0,1]", result.PCTR)
	}
}

func TestHandlePredictCTR_InvalidMethod(t *testing.T) {
	h := makeTestHandler(t)
	r := httptest.NewRequest(http.MethodGet, "/v1/predict/ctr", nil)
	w := httptest.NewRecorder()
	h.HandlePredictCTR(w, r)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

func TestHandlePredictCTR_InvalidBody(t *testing.T) {
	h := makeTestHandler(t)
	r := httptest.NewRequest(http.MethodPost, "/v1/predict/ctr", bytes.NewReader([]byte("invalid json")))
	w := httptest.NewRecorder()
	h.HandlePredictCTR(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandlePredictBatch_OK(t *testing.T) {
	h := makeTestHandler(t)

	reqBody := prediction.BatchRequest{
		User:    feature.UserFeatures{UserID: "u001", Age: 28, Geo: "CN"},
		Context: feature.ContextFeatures{HourOfDay: 14, Domain: "news.example.com"},
		Ads: []feature.AdFeatures{
			{AdvertiserID: "adv001", AdType: "banner"},
			{AdvertiserID: "adv002", AdType: "native"},
		},
	}
	data, _ := json.Marshal(reqBody)

	r := httptest.NewRequest(http.MethodPost, "/v1/predict/batch", bytes.NewReader(data))
	w := httptest.NewRecorder()
	h.HandlePredictBatch(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var result prediction.BatchResult
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("解析响应失败: %v", err)
	}
	if len(result.Results) != 2 {
		t.Errorf("results count = %d, want 2", len(result.Results))
	}
}

func TestHandlePredictBatch_EmptyAds(t *testing.T) {
	h := makeTestHandler(t)

	reqBody := prediction.BatchRequest{
		User:    feature.UserFeatures{UserID: "u001"},
		Context: feature.ContextFeatures{},
		Ads:     []feature.AdFeatures{},
	}
	data, _ := json.Marshal(reqBody)

	r := httptest.NewRequest(http.MethodPost, "/v1/predict/batch", bytes.NewReader(data))
	w := httptest.NewRecorder()
	h.HandlePredictBatch(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandleModelCurrent(t *testing.T) {
	h := makeTestHandler(t)
	r := httptest.NewRequest(http.MethodGet, "/v1/model/current", nil)
	w := httptest.NewRecorder()
	h.HandleModelCurrent(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp map[string]string
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["version"] == "" {
		t.Error("version 不应为空")
	}
}

func TestHandleHealth(t *testing.T) {
	h := makeTestHandler(t)
	r := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	h.HandleHealth(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
}
