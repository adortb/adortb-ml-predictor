package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/adortb/adortb-ml-predictor/internal/model"
)

func makeTestTrainHandler() *TrainHandler {
	return NewTrainHandler(model.NewFTRLModel(model.DefaultFTRLConfig()))
}

func TestHandleTrainSample_OK(t *testing.T) {
	h := makeTestTrainHandler()

	req := TrainSampleRequest{
		Features: map[string]float64{"f1": 1.0, "f2": 0.5},
		Label:    1.0,
		Weight:   1.0,
	}
	data, _ := json.Marshal(req)

	r := httptest.NewRequest(http.MethodPost, "/v1/train_sample", bytes.NewReader(data))
	w := httptest.NewRecorder()
	h.HandleTrainSample(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	var resp TrainSampleResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("解析响应失败: %v", err)
	}
	if resp.Status != "updated" {
		t.Errorf("status = %s, want updated", resp.Status)
	}
}

func TestHandleTrainSample_InvalidMethod(t *testing.T) {
	h := makeTestTrainHandler()
	r := httptest.NewRequest(http.MethodGet, "/v1/train_sample", nil)
	w := httptest.NewRecorder()
	h.HandleTrainSample(w, r)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

func TestHandleTrainSample_InvalidBody(t *testing.T) {
	h := makeTestTrainHandler()
	r := httptest.NewRequest(http.MethodPost, "/v1/train_sample", bytes.NewReader([]byte("invalid")))
	w := httptest.NewRecorder()
	h.HandleTrainSample(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandleTrainSample_InvalidLabel(t *testing.T) {
	h := makeTestTrainHandler()
	req := TrainSampleRequest{
		Features: map[string]float64{"f1": 1.0},
		Label:    0.5, // 无效，必须是 0 或 1
		Weight:   1.0,
	}
	data, _ := json.Marshal(req)
	r := httptest.NewRequest(http.MethodPost, "/v1/train_sample", bytes.NewReader(data))
	w := httptest.NewRecorder()
	h.HandleTrainSample(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandleTrainSample_EmptyFeatures(t *testing.T) {
	h := makeTestTrainHandler()
	req := TrainSampleRequest{
		Features: map[string]float64{},
		Label:    1.0,
		Weight:   1.0,
	}
	data, _ := json.Marshal(req)
	r := httptest.NewRequest(http.MethodPost, "/v1/train_sample", bytes.NewReader(data))
	w := httptest.NewRecorder()
	h.HandleTrainSample(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandleTrainSample_DefaultWeight(t *testing.T) {
	h := makeTestTrainHandler()
	// weight <= 0 时应默认为 1.0
	req := TrainSampleRequest{
		Features: map[string]float64{"f1": 1.0},
		Label:    1.0,
		Weight:   0, // 触发默认值
	}
	data, _ := json.Marshal(req)
	r := httptest.NewRequest(http.MethodPost, "/v1/train_sample", bytes.NewReader(data))
	w := httptest.NewRecorder()
	h.HandleTrainSample(w, r)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
}
