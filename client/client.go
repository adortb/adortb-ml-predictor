// Package client 提供 adortb-ml-predictor 的 Go 客户端，供 DSP 引用。
//
// 使用示例（DSP 集成）：
//
//	url := os.Getenv("ML_PREDICTOR_URL") // e.g. http://predictor:8087
//	if url == "" {
//	    // 未配置时跳过，使用默认出价
//	    return baseBidCPM
//	}
//	c := client.New(url, 5*time.Millisecond)
//	pctr, err := c.PredictCTR(ctx, user, ctxFeatures, ad)
//	if err != nil {
//	    return baseBidCPM // 降级到默认出价
//	}
//	finalBid := baseBidCPM * pctr * calibrationFactor
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/adortb/adortb-ml-predictor/internal/feature"
	"github.com/adortb/adortb-ml-predictor/internal/prediction"
)

// Client adortb-ml-predictor HTTP 客户端
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// New 创建客户端
// timeout 建议设置为 DSP 竞价超时的 1/3（如 5ms）
func New(baseURL string, timeout time.Duration) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

// PredictCTR 预测单个广告的 CTR
// 错误时调用方应降级到默认出价，不应阻塞竞价流程
func (c *Client) PredictCTR(
	ctx context.Context,
	user feature.UserFeatures,
	ctxFeatures feature.ContextFeatures,
	ad feature.AdFeatures,
) (float64, error) {
	req := prediction.Request{
		User:    user,
		Context: ctxFeatures,
		Ad:      ad,
	}

	var result prediction.Result
	if err := c.post(ctx, "/v1/predict/ctr", req, &result); err != nil {
		return 0, err
	}
	return result.PCTR, nil
}

// PredictBatch 批量预测同一用户多个候选广告
func (c *Client) PredictBatch(
	ctx context.Context,
	user feature.UserFeatures,
	ctxFeatures feature.ContextFeatures,
	ads []feature.AdFeatures,
) ([]float64, error) {
	req := prediction.BatchRequest{
		User:    user,
		Context: ctxFeatures,
		Ads:     ads,
	}

	var result prediction.BatchResult
	if err := c.post(ctx, "/v1/predict/batch", req, &result); err != nil {
		return nil, err
	}

	pctrs := make([]float64, len(result.Results))
	for i, r := range result.Results {
		pctrs[i] = r.PCTR
	}
	return pctrs, nil
}

func (c *Client) post(ctx context.Context, path string, body, out any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("predictor returned %d", resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}
