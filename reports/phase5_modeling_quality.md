# 第五阶段统计与样本外预测报告

## 样本与识别边界

- 分析单位：单次业绩电话会事件。
- 训练期：2017-11-13 至 2021-09-28，7,305 条。
- 另有 1,567 条临近测试边界的事件不参与预测训练，以避免 CAR60 标签跨越边界。
- 测试期：2022-01-05 至 2023-02-23，2,804 条。
- 模型规格在 2020 和 2021 年验证折上预选；各折留出 95 个日历日的 CAR60 标签缓冲期。
- 2022–2023 年测试集只用于锁定模型的最终汇报；其他模型的测试指标保留作审计，不按测试表现重新选择。
- 特征缩尾和标准化参数只由训练期估计；测试期目标保持原值。
- 回归为关联性分析，不支持因果推断。

## 验证期预选模型的最终测试表现

- CAR_5: CatBoost，月度 Rank IC 均值 0.2164，汇总 Rank IC 0.1848，RMSE 0.1109，方向准确率 56.06%。
- CAR_10: CatBoost，月度 Rank IC 均值 0.1679，汇总 Rank IC 0.1419，RMSE 0.1382，方向准确率 54.46%。
- CAR_60: LightGBM，月度 Rank IC 均值 0.0949，汇总 Rank IC 0.0822，RMSE 0.2390，方向准确率 51.82%。

## 诚信边界

组合排序只是按事件月计算的多空五分位诊断，未建模持仓重叠、交易成本、停牌或可做空性，因此不应解释为可实现的策略收益。

## 输出文件

- `validation_metrics`: 36 行，SHA-256 `926cd8f0c40c570ae7f4b9a9bb0a6ebf6b697b4fbe6f54f8c4b622e6178e5a04`
- `modeling_sample`: 11,676 行，SHA-256 `8925a6398ef3f5858a80b927cbd9093fa523530ef4a8e93a9e742424386bb6cb`
- `transform_parameters`: 7 行，SHA-256 `b7bf7d6c4d5274ba5fa47dfb958586de8e6228f139130908bfcfddd2c3676a94`
- `target_clip_bounds`: 3 行，SHA-256 `d758bcf0de2f36fe7bc3bca31dc2e19c7b1fbe5608f1c77f24c152f5182e81d8`
- `descriptive`: 10 行，SHA-256 `d4602982f1eeb2eeadf72665091700ad8774f26c81144844cf311a7e60a89792`
- `correlations`: 45 行，SHA-256 `1f9687113de958269a99ae45ce8e07cb48316c398e4bd0d7523082f3de6531ec`
- `ols_coefficients`: 63 行，SHA-256 `66d16af299be147d8b5616add13be5b7fc4eb2050cfb230da6eb4fc34ee7de4a`
- `ols_models`: 9 行，SHA-256 `46257c2698abedcb6aaf268d7bbf5e12ed5160e57f6666a8b85b64266d628122`
- `metrics`: 21 行，SHA-256 `10b96571ee7bd53832fdbf50eedd4cad5011ce9f0c3da2394ccf096d8579e3fe`
- `predictions`: 2,804 行，SHA-256 `2789d6e2b036eced09f167df359f752139a23c34e2524c0523975f91799bed27`
- `importance`: 252 行，SHA-256 `fda9991b6f078d4d21e6dee9c4e09ff8e8a19667528c48c3373a58347a187b2c`
- `portfolio_sorts`: 168 行，SHA-256 `085d34c954ed4e1708edeec26c6c306a2d1f50eb96b301a1c211ed294b5882d9`
