# 第五阶段统计与样本外预测报告

## 样本与识别边界

- 分析单位：单次业绩电话会事件。
- 训练期：2017-11-13 至 2021-12-29，8,872 条。
- 测试期：2022-01-05 至 2023-02-23，2,804 条。
- 特征缩尾和标准化参数只由训练期估计；测试期目标保持原值。
- 回归为关联性分析，不支持因果推断。

## 样本外最佳月度 Rank IC

- CAR_5: CatBoost，月度 Rank IC 均值 0.2173，汇总 Rank IC 0.1884，RMSE 0.1109，方向准确率 55.81%。
- CAR_10: CatBoost，月度 Rank IC 均值 0.1660，汇总 Rank IC 0.1542，RMSE 0.1383，方向准确率 53.60%。
- CAR_60: TIDE，月度 Rank IC 均值 0.1426，汇总 Rank IC 0.1132，RMSE 0.2392，方向准确率 51.60%。

## 诚信边界

组合排序只是按事件月计算的多空五分位诊断，未建模持仓重叠、交易成本、停牌或可做空性，因此不应解释为可实现的策略收益。

## 输出文件

- `modeling_sample`: 11,676 行，SHA-256 `4164ddd4ecc7b69b013501a776eea3f63d965af75f857afe81449c04d6c7d210`
- `transform_parameters`: 7 行，SHA-256 `6c2270ac4edb08af3b582bd67b9cb9a11997b6d6becd78fb5f859243e14edad6`
- `target_clip_bounds`: 3 行，SHA-256 `46d3e38ea8575aae9079a1870db961fa980ed8674dbeebae32ac707879789a3d`
- `descriptive`: 10 行，SHA-256 `d4602982f1eeb2eeadf72665091700ad8774f26c81144844cf311a7e60a89792`
- `correlations`: 45 行，SHA-256 `1f9687113de958269a99ae45ce8e07cb48316c398e4bd0d7523082f3de6531ec`
- `ols_coefficients`: 63 行，SHA-256 `d09888c93263a7cb3167fb0eb24c7572a6e26b9ffa50c4827ec963a8f36fe295`
- `ols_models`: 9 行，SHA-256 `3e8c79f16454d71fe6e50dd7814cd4bc3014235f8b97f43a267b6c8078173cf6`
- `metrics`: 21 行，SHA-256 `c376e724bc1602d0f2c250068c8e9270f123165b09d264fe640c71e68c7a79db`
- `predictions`: 2,804 行，SHA-256 `b41455ac92c6ffa35c2e0e99c35505723d580fb17c43580f61fd02f4b79ab543`
- `importance`: 252 行，SHA-256 `34790b7f99b4d88230103a29cbe6ea275f41114b48a9b2599194bdaf67291e1b`
- `portfolio_sorts`: 168 行，SHA-256 `b0e92a515185a178b01406ce5b3ef183626db7f9795b9c27de6456ef5a5007a8`
