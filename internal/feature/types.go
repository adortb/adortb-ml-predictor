package feature

// UserFeatures 用户特征（20 维）
type UserFeatures struct {
	UserID         string   `json:"user_id"`
	Age            int      `json:"age"`
	Gender         string   `json:"gender"`
	Geo            string   `json:"geo"`
	DeviceType     string   `json:"device_type"`
	OS             string   `json:"os"`
	ConnectionType string   `json:"connection_type"`
	DMPTags        []string `json:"dmp_tags"`
	ActiveDays7d   int      `json:"active_days_7d"`
	ClickCount7d   int      `json:"click_count_7d"`
	BuyCount30d    int      `json:"buy_count_30d"`
}

// ContextFeatures 上下文特征
type ContextFeatures struct {
	HourOfDay    int    `json:"hour_of_day"`
	DayOfWeek    int    `json:"day_of_week"`
	PublisherID  string `json:"publisher_id"`
	Domain       string `json:"domain"`
	AdSlotType   string `json:"adslot_type"`
	AdSlotSize   string `json:"adslot_size"`
	PageCategory string `json:"page_category"`
}

// AdFeatures 广告特征
type AdFeatures struct {
	AdvertiserID  string  `json:"advertiser_id"`
	CampaignID    string  `json:"campaign_id"`
	CreativeID    string  `json:"creative_id"`
	AdType        string  `json:"ad_type"`
	AdSize        string  `json:"ad_size"`
	AdCategory    string  `json:"ad_category"`
	LandingDomain string  `json:"landing_domain"`
	HistoricalCTR float64 `json:"historical_ctr"`
}

// FeatureVector 最终特征向量（用于模型输入）
type FeatureVector map[string]float64
