# adortb-ml-predictor

五期核心服务。实时 CTR（点击率）预估引擎，支持 Logistic Regression 离线推理与 FTRL-Proximal 在线学习，提供毫秒级竞价决策。

## 算法概述

### Logistic Regression (LR)

标准二分类对数回归：

```
score = bias + Σ(w_i · x_i)
p(click) = sigmoid(score) = 1 / (1 + e^(-score))
```

- 特征权重以 JSON 文件存储，冷启动加载
- 支持 `LoadLRFromFile` / `LoadLRFromBytes` 两种加载方式

### FTRL-Proximal（Follow The Regularized Leader）

在线学习算法，来自论文 *McMahan et al., "Ad Click Prediction: a View from the Trenches" (KDD 2013)*：

```
权重恢复公式：
w_i = -(z_i - sign(z_i)·L1) / (L2 + (β + √n_i) / α)

当 |z_i| ≤ L1 时，w_i = 0（天然稀疏）
```

超参：`alpha=0.1`（学习率分子），`beta=1.0`（平滑），`L1=0.1`，`L2=1.0`，`hash_dim=1,000,000`

每次曝光后调用 `Update(features, label, sampleWeight)` 做一步梯度更新，支持高并发（`sync.RWMutex` 保护）。

### Platt Scaling 校准

消除模型输出与真实概率的偏差：

```
logit(p) = A · logit(raw_score) + B
p_calibrated = sigmoid(A · logit(raw_score) + B)
```

默认 A=1, B=0（恒等），可从离线校准数据拟合参数。

### 特征工程

特征哈希（Feature Hashing）：使用 xxhash 将 `(field, value)` 对映射到 `10^7` 维稀疏向量：

| 特征组 | 特征名 | 处理方式 |
|--------|--------|----------|
| 用户 | 年龄段、性别、地域、设备、OS、网络类型 | one-hot hash |
| 用户 | DMP标签（最多10个）| multi-hot hash |
| 用户 | 7日活跃天数、7日点击数、30日购买数 | 线性归一化[0,1] |
| 上下文 | 小时/星期、Publisher、域名、广告位类型/尺寸、页面类别 | one-hot hash |
| 广告 | 广告主/活动/创意ID、广告类型/尺寸/类别、落地页域名 | one-hot hash |
| 广告 | 历史CTR | clamp[0,1] |
| 交叉 | geo×adtype、domain×advertiser、hour×adtype、device×size | 组合hash |

## 快速开始

```bash
# 编译服务
go build -o bin/predictor ./cmd/predictor

# 运行
./bin/predictor -port 8080 -model models/lr_ctr_v1.json

# 离线批量训练
go run scripts/train.go -data data/train.csv -out models/lr_v2.json
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/predict` | 批量预测 CTR |
| POST | `/v1/train` | 在线更新（FTRL） |
| GET  | `/metrics` | Prometheus 指标 |
| GET  | `/healthz` | 健康检查 |

### 预测请求示例

```json
POST /v1/predict
{
  "requests": [
    {
      "user": { "age": 28, "gender": "M", "geo": "US", "device_type": "mobile" },
      "context": { "hour_of_day": 14, "domain": "news.example.com" },
      "ad": { "advertiser_id": "adv_123", "historical_ctr": 0.023 }
    }
  ]
}
```

## 模型评估指标

| 指标 | 说明 |
|------|------|
| AUC | 排序能力，通常需 > 0.75 |
| LogLoss | 概率校准质量 |
| ECE | 期望校准误差（Platt scaling 后） |

## 技术栈

- **语言**: Go 1.25
- **特征哈希**: cespare/xxhash v2
- **存储**: ClickHouse（历史日志）、Redis（模型缓存）
- **监控**: Prometheus
- **并发**: sync.RWMutex（FTRL 在线更新读写分离）

## 目录结构

```
adortb-ml-predictor/
├── cmd/predictor/      # 服务入口
├── client/             # Go SDK
├── internal/
│   ├── api/            # HTTP handler（预测/训练）
│   ├── feature/        # 特征提取器（Extractor）
│   ├── model/          # LR、FTRL、Platt校准、模型注册
│   ├── prediction/     # 推理流水线
│   ├── store/          # 模型持久化仓库
│   └── trainer/        # 离线批量训练
├── models/             # 预置模型文件
└── scripts/            # 训练脚本
```
