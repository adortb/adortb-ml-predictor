# CLAUDE.md — adortb-ml-predictor

## 项目角色

五期核心服务：实时 CTR 预估。对延迟极敏感（竞价窗口 < 10ms），任何变更都需评估推理路径的耗时影响。

## 关键函数与复杂度

| 函数 | 文件 | 复杂度 | 说明 |
|------|------|--------|------|
| `Extractor.Extract` | `internal/feature/extractor.go` | O(F)，F=特征数 | 单次推理特征提取，约40+维 |
| `LRModel.Predict` | `internal/model/lr.go` | O(F) | 稀疏点积 + sigmoid |
| `FTRLModel.Predict` | `internal/model/ftrl.go` | O(F) | 从z/n恢复权重 + 点积，读锁保护 |
| `FTRLModel.Update` | `internal/model/ftrl.go` | O(F) | 写锁，在线梯度更新 |
| `Calibrator.Calibrate` | `internal/model/calibration.go` | O(1) | Platt scaling，logit → sigmoid |
| `hashKey` | `internal/feature/extractor.go` | O(1) | xxhash，碰撞率极低 |

## 并发安全

- `FTRLModel` 用 `sync.RWMutex`：Predict 持读锁，Update 持写锁
- 高并发场景下 Update 是写锁瓶颈，若 QPS > 10k 可考虑分片（按特征子集）或异步批量更新
- `LRModel` 无锁（只读），安全并发

## 特征工程规范

- 所有分类特征必须通过 `hashKey(field, value)` 编码，**不能**直接用字符串作 key
- 连续特征统一归一化到 [0,1]，使用 `normalize(v, min, max)` 或 `clamp`
- DMP 标签限最多取前 `maxDMPTags=10` 个，防止稀疏爆炸
- 新增交叉特征在 `extractCrossFeatures` 中添加，命名格式：`cross_<field1>_<field2>`

## 模型训练流程

```
离线批量训练（offline.go）：
  1. 读取 ClickHouse 曝光/点击日志
  2. Extractor.Extract → FeatureVector
  3. LRModel / FTRLModel 批量训练（梯度下降）
  4. 在验证集上计算 AUC / LogLoss
  5. Platt scaling 拟合校准参数
  6. 导出模型 JSON 到 models/

在线更新（FTRL）：
  每次竞价后 → API /v1/train → FTRLModel.Update
```

## 模型持久化

- `FTRLModel.Snapshot()` → JSON，包含完整 z/n 稀疏 map
- `FTRLModel.SaveToFile(path)` 写磁盘；`LoadFTRLFromFile` 加载
- LR 模型格式：`{"weights": {"f12345": 0.03}, "bias": -1.2, "version": "v1"}`
- 模型文件放 `models/` 目录，版本号写入 `Version` 字段

## 评估指标目标

| 指标 | 目标 |
|------|------|
| AUC | ≥ 0.75 |
| LogLoss | 尽可能小（与基线对比） |
| P99 推理延迟 | < 2ms |

## 开发注意事项

- 修改 `feature/extractor.go` 后必须更新特征文档，线上特征版本不兼容会导致模型失效
- FTRL `hash_dim` 默认 1,000,000，改动后旧模型 Snapshot 不再可用
- `sigmoid` 函数在 `lr.go` 中定义，`ftrl.go` 中复用（同包）
- 添加新 API 端点需同步更新 Prometheus 指标（`internal/metrics/metrics.go`）

## 测试

```bash
go test -race ./...
go test -race -bench=. ./internal/model/
go test -race -bench=. ./internal/feature/
```

关键测试文件：
- `internal/model/ftrl_test.go` — 在线更新正确性、并发安全
- `internal/model/lr_test.go` — sigmoid 正确性、稀疏权重匹配
- `internal/feature/extractor_test.go` — 特征提取确定性
- `internal/model/calibration_test.go` — Platt scaling 数值正确性
