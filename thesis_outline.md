# 三阶段论文大纲

## 题目建议

FedHDS-assisted Federated Unlearning with Retained-Set Curvature and Guarded Trust-Region Updates

中文题目可写为：

一种结合 FedHDS、保留集曲率近似与受控信赖域更新的联邦遗忘方法

## 论文中心论点

本文不主张 retained-set Hessian 在所有指标上全面优于 forget-batch Hessian。更稳妥的中心论点是：

在 FedHDS 训练底座上，使用 retained-set curvature approximation、update norm clipping 和 stepped trust-region guard，可以构建一条可运行、可控、能保护 global utility 的联邦遗忘链路；进一步通过 update sign 验证和 forget-loss guard，可以修正 retained-set update 的遗忘方向，使 forget-client loss 在 global loss 受控的条件下上升。

## 章节结构

### 1. Introduction

要写清楚：

- 大模型和联邦学习场景下，客户端可能提出数据删除或遗忘请求。
- 直接重训代价高，federated unlearning 希望在不完整重训的情况下移除指定客户端影响。
- 二阶近似 unlearning 可以用梯度和 Hessian 近似修正模型，但 Hessian reference 的选择会影响 utility 和 forgetting trade-off。
- forget-batch Hessian 更直接服务遗忘目标，但可能损伤 global utility。
- retained-set Hessian 更符合保留任务稳定性，但更新方向需要 guard 和方向验证。

贡献点：

- 构建 FedHDS + federated unlearning 实验链路。
- 提出 retained-set curvature approximation。
- 引入 clipping 和 stepped trust-region guard。
- 增加 direction probing 和 forget-loss guard，修正 retained-set update 的方向问题。

### 2. Related Work

建议分三类：

- Federated Learning and FedHDS
- Federated Unlearning
- Second-order Approximation and Guarded Model Updates

目前先写概述性文字，不急着补全引用格式。

### 3. Method

核心小节：

- Problem Formulation
- FedHDS Training Backbone
- Forget-client Gradient
- Retained-set Curvature Approximation
- LiSSA/HVP Approximation
- Global Update Norm Clipping
- Stepped Trust-region Guard
- Direction Probing and Forget-loss Guard

方法边界：

- 不引入 KD。
- 不做重训 residual 存储。
- 不做差分隐私补偿。
- 不做大模型剪枝。
- 重点保持方法主线集中。

### 4. Experimental Setup

写清楚三阶段配置：

- 模型：Qwen2-0.5B
- 数据集：Dolly
- 客户端数：20
- forget client：0
- 阶段一：data_sample=0.2, rounds=10
- 阶段二：data_sample=0.4, rounds=20
- 阶段三：沿用阶段二配置，做 direction validation 和 negative sign 小范围验证

对比方法：

- FL
- FedHDS
- FedHDS + forget-batch Hessian
- FedHDS + retained-set Hessian + guard
- Stage-3 negative sign guarded retained update

指标：

- final global loss
- forget-client loss before/after unlearning
- update L2 norm
- accepted/attempted guard steps
- unlearning time

### 5. Results and Analysis

建议按逻辑写，不按流水账写：

- Stage 1: feasibility check
- Stage 2: main trade-off
- Stage 3: direction validation

关键结论：

- 阶段一证明链路可运行。
- 阶段二显示 FedHDS 在更大设置下优于 FL。
- forget-batch Hessian forgetting signal 更强，但 global loss 损伤更大。
- retained-set guard 能保护 global utility，但原 positive/auto 方向遗忘不足。
- 阶段三说明问题主要是 update direction misalignment。
- negative sign + forget-loss guard 能让 forget loss 上升，同时 global loss 受控。

### 6. Limitations

必须诚实写：

- 阶段三是小范围 direction validation，不是完整大规模实验。
- 目前没有多 seed、多 forget client。
- retained-set Hessian 计算成本高。
- reference size 和 guard threshold 对显存和效果敏感。
- 大模型和更大数据规模留作后续工作。

### 7. Conclusion

总结：

- 本文提出并验证了一种 FedHDS-assisted retained-set guarded federated unlearning 框架。
- retained-set curvature + clipping + stepped guard 能控制 global utility。
- direction validation 和 forget-loss guard 修正了 retained-set update 的方向问题。
- 当前结果支持小论文主线，但大规模泛化仍需后续验证。

## 当前写作策略

先写 Method 和 Experimental Setup，再写 Results and Analysis。Introduction 和 Conclusion 最后定稿。



## 2026-05-04 更新

阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。


## 2026-05-04 更新

阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。


## 2026-05-04 更新

阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。


## 2026-05-04 阶段四更新

阶段四A/B/C结果已纳入论文：阶段四使用 2 x 2 模型和数据规模泛化矩阵，把阶段三作为小模型小数据格子，并补充小模型大数据、大模型小数据、大模型大数据三组完整 9 组实验。论文中应使用 `stage4a_qwen05b_dsample06_full9_20260504`、`stage4b_qwen15b_dsample04_full9_20260504` 和 `stage4_large_model_data_20260504` 的 summary.csv、paper_table.md 和 figures。


## 2026-05-04 更新

阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。


## 2026-05-04 阶段四更新

阶段四A/B/C结果已纳入论文：阶段四使用 2 x 2 模型和数据规模泛化矩阵，把阶段三作为小模型小数据格子，并补充小模型大数据、大模型小数据、大模型大数据三组完整 9 组实验。论文中应使用 `stage4a_qwen05b_dsample06_full9_20260504`、`stage4b_qwen15b_dsample04_full9_20260504` 和 `stage4_large_model_data_20260504` 的 summary.csv、paper_table.md 和 figures。


## 2026-05-04 更新

阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。


## 2026-05-04 阶段四更新

阶段四A/B/C结果已纳入论文：阶段四使用 2 x 2 模型和数据规模泛化矩阵，把阶段三作为小模型小数据格子，并补充小模型大数据、大模型小数据、大模型大数据三组完整 9 组实验。论文中应使用 `stage4a_qwen05b_dsample06_full9_20260504`、`stage4b_qwen15b_dsample04_full9_20260504` 和 `stage4_large_model_data_20260504` 的 summary.csv、paper_table.md 和 figures。


## 2026-05-04 更新

阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。


## 2026-05-04 阶段四更新

阶段四A/B/C结果已纳入论文：阶段四使用 2 x 2 模型和数据规模泛化矩阵，把阶段三作为小模型小数据格子，并补充小模型大数据、大模型小数据、大模型大数据三组完整 9 组实验。论文中应使用 `stage4a_qwen05b_dsample06_full9_20260504`、`stage4b_qwen15b_dsample04_full9_20260504` 和 `stage4_large_model_data_20260504` 的 summary.csv、paper_table.md 和 figures。


## 2026-05-04 更新

阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。


## 2026-05-04 阶段四更新

阶段四A/B/C结果已纳入论文：阶段四使用 2 x 2 模型和数据规模泛化矩阵，把阶段三作为小模型小数据格子，并补充小模型大数据、大模型小数据、大模型大数据三组完整 9 组实验。论文中应使用 `stage4a_qwen05b_dsample06_full9_20260504`、`stage4b_qwen15b_dsample04_full9_20260504` 和 `stage4_large_model_data_20260504` 的 summary.csv、paper_table.md 和 figures。


## 2026-05-04 更新

阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。


## 2026-05-04 阶段四更新

阶段四A/B/C结果已纳入论文：阶段四使用 2 x 2 模型和数据规模泛化矩阵，把阶段三作为小模型小数据格子，并补充小模型大数据、大模型小数据、大模型大数据三组完整 9 组实验。论文中应使用 `stage4a_qwen05b_dsample06_full9_20260504`、`stage4b_qwen15b_dsample04_full9_20260504` 和 `stage4_large_model_data_20260504` 的 summary.csv、paper_table.md 和 figures。


## 2026-05-04 更新

阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。


## 2026-05-04 阶段四更新

阶段四A/B/C结果已纳入论文：阶段四使用 2 x 2 模型和数据规模泛化矩阵，把阶段三作为小模型小数据格子，并补充小模型大数据、大模型小数据、大模型大数据三组完整 9 组实验。论文中应使用 `stage4a_qwen05b_dsample06_full9_20260504`、`stage4b_qwen15b_dsample04_full9_20260504` 和 `stage4_large_model_data_20260504` 的 summary.csv、paper_table.md 和 figures。
