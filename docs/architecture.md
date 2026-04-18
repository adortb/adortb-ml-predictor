# Architecture — adortb-ml-predictor

## 系统概览

```
竞价请求
    │
    ▼
┌─────────────────────────────────────────┐
│              HTTP API Server            │
│  POST /v1/predict   POST /v1/train      │
└────────────┬────────────────────────────┘
             │
             ▼
┌────────────────────────┐
│   Prediction Pipeline  │  internal/prediction/pipeline.go
│  ┌──────────────────┐  │
│  │FeatureExtractor  │  │  Extract(user, ctx, ad) → FeatureVector
│  └────────┬─────────┘  │
│           │             │
│  ┌────────▼─────────┐  │
│  │  Model Registry  │  │  LRModel | FTRLModel
│  └────────┬─────────┘  │
│           │             │
│  ┌────────▼─────────┐  │
│  │   Calibrator     │  │  Platt Scaling
│  └──────────────────┘  │
└────────────────────────┘
             │
             ▼
        CTR Score (float64)
```

## 训练流程

```
ClickHouse（曝光/点击日志）
    │
    ▼
┌───────────────────────────────────────┐
│          Offline Trainer               │
│                                       │
│  1. 批量读取样本（正样本=点击，负样本=曝光未点击）
│  2. Extractor.Extract → FeatureVector │
│  3. 梯度下降训练 LR / FTRL            │
│  4. 验证集评估（AUC, LogLoss）        │
│  5. Platt Scaling 校准参数拟合        │
│  6. 模型导出 JSON                     │
└───────────────────────────────────────┘
    │
    ▼
models/lr_ctr_v1.json
    │
    ▼ （服务启动加载）
Model Store（内存）
```

## 推理流程时序图

```
Client          Handler         Pipeline        Extractor       Model
  │               │                │               │              │
  │─ POST /predict►│                │               │              │
  │               │─ Predict(req) ►│               │              │
  │               │                │─ Extract() ──►│              │
  │               │                │◄── FeatureVec ─│              │
  │               │                │─────────────────────────────►│
  │               │                │                │  LR.Predict()│
  │               │                │◄─ raw_score ────────────────  │
  │               │                │─ Calibrate(raw_score)         │
  │               │◄── CTR score──  │               │              │
  │◄─ response ───│                │               │              │
```

## 在线学习流程（FTRL）

```
曝光结果（click=1/0）
    │
    ▼
POST /v1/train
    │
    ▼
┌──────────────────────────────┐
│   FTRLModel.Update()         │
│                              │
│  1. 写锁 (sync.RWMutex)     │
│  2. 前向推理 → pred          │
│  3. grad = w * (pred - label)│
│  4. 更新 z/n（每个特征坐标）  │
│  5. 解锁                     │
└──────────────────────────────┘
    │
    ▼
（定期）FTRLModel.SaveToFile()
    │
    ▼
models/ftrl_snapshot.json
```

## 特征工程流程

```
UserFeatures + ContextFeatures + AdFeatures
    │
    ▼
┌─────────────────────────────────────────┐
│             Extractor.Extract()          │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ User Features                   │   │
│  │  age → ageBucket() → hashSet()  │   │
│  │  dmp_tags → hashAdd() × N       │   │
│  │  active_days → normalize[0,1]   │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ Context Features                │   │
│  │  hour → float64/23.0            │   │
│  │  publisher/domain → hashSet()   │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ Ad Features                     │   │
│  │  advertiser/campaign → hashSet()│   │
│  │  historical_ctr → clamp[0,1]    │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ Cross Features                  │   │
│  │  geo×adtype, domain×advertiser  │   │
│  │  hour×adtype, device×size       │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    │
    ▼
FeatureVector (map[string]float64, 维度≤HashDim=10^7)
```

## 数据输入输出

### 输入

| 字段 | 类型 | 来源 |
|------|------|------|
| UserFeatures | struct | DMP / 用户画像服务 |
| ContextFeatures | struct | 竞价请求（Publisher信号） |
| AdFeatures | struct | 广告元数据 + 历史统计 |

### 输出

| 字段 | 类型 | 说明 |
|------|------|------|
| CTR | float64 | 经Platt校准的点击概率 [0,1] |
| ModelVersion | string | 模型版本（用于AB实验追踪） |

## 模型评估指标

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| AUC-ROC | 离线验证集 | ≥ 0.75 |
| LogLoss | -Σ[y·log(p) + (1-y)·log(1-p)] / N | 越小越好 |
| ECE | 分桶校准误差 | < 0.02（Platt后） |
| P99推理延迟 | 端到端（含特征提取） | < 2ms |

## 依赖关系

```
adortb-ml-predictor
├── ClickHouse  （历史曝光/点击日志，训练数据源）
├── Redis       （模型版本缓存，热更新）
└── Prometheus  （推理延迟、QPS、AUC监控）
```
