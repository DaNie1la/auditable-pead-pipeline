# 第三阶段 FinBERT 文本指标报告

## 结论

已对 17,123 个唯一事件的 Prepared Remarks 与 Q&A 分别完成长文本 FinBERT 打分。

## 模型与口径

- 模型：`ProsusAI/finbert`
- 模型提交：`0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`
- 最大窗口：512 tokens
- 重叠：64 tokens
- 聚合：`new_token_weighted_mean_probability`
- 精度：`cuda_float16`
- 事件 batch：8
- 窗口 batch：16
- 配置哈希：`1a2535ffc20e4358b4ffc7a37cc4f481d3b63f5a808629ed7a13940050504a00`
- 语调公式：`P(positive) - P(negative)`

## 运行

- 设备：NVIDIA GeForce RTX 5060 Laptop GPU
- PyTorch：2.11.0+cu128
- Transformers：4.57.6
- 本次运行耗时：18.5 分钟
- 断点续跑前已有：10,464 条
- 本次新增：6,659 条

## 输出

- 行数：17,123
- Prepared 窗口：157,744
- Q&A 窗口：241,420
- 文件：`<LOCAL_PATH>\Desktop\文件\科研项目\CIS\data\processed\finbert_section_scores.csv`
- SHA-256：`5fa976c1351fab5c3a8d89af6322f056294fee4a8172708e690b33c759364c24`
