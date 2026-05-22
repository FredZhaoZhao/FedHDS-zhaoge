# V3.2 Increment Report

## 执行结果

- 新增有效 seed：`seed47`（`alpha=0.5`）
- 本次没有触发 fallback：未使用 `seed48`、`seed49` 或 `alpha=0.7`
- `seed47` 健康检查通过：
  - `fedhds final_global_loss = 3.0237`
  - `fl final_global_loss = 2.9291`
  - `retrain_oracle final_global_loss = 2.9960`
  - `mia_member_count >= 150` 成立（`fedhds=269`, `fl=269`, `retrain_oracle=471`）
- 顶层结果已重整为 `5` 个有效 seed：
  - `seed42, seed43, seed44, seed45, seed47`
- 异常 `seed46` 已移至：
  - `formal_cloud_results/v3_dir05_seed5/appendix_seed46/`

## 验收清单

- [x] `formal_cloud_results/v3_dir05_seed5/seed47/` 存在且包含完整 7 个方法子目录
- [x] `seed47` 的 `fedhds/fl/retrain_oracle final_global_loss < 5.0`
- [x] `seed47` 的 `mia_member_count >= 150`
- [x] `seed46` 已移动到 `appendix_seed46/`，原始结果保留
- [x] 顶层 `summary.csv` 已重生成
- [x] 顶层 `paper_table.csv` 与 `paper_table.md` 已重生成
- [x] 顶层 `figures/` 已重生成
- [x] 新 `summary.csv` 恰好 `35` 行（`5 seeds x 7 methods`）
- [x] `summary.csv` 仅包含 `seed42, seed43, seed44, seed45, seed47`
- [x] `v3_2_increment_report.md` 已更新

## 新 Seed 的关键数字

| Method | Forget Loss Delta | Final Global Loss | Update L2 | Accepted Steps |
|---|---:|---:|---:|---:|
| `fedhds` | - | 3.0237 | - | - |
| `fl` | - | 2.9291 | - | - |
| `retrain_oracle` | `2.3598 -> 2.3598` | 2.9960 | - | 0/0 |
| `forget_batch_hessian` | `2.3402 -> 3.1277` | 3.4247 | 1.6664 | 1/1 |
| `retained_negative_guard` | `2.3402 -> 2.5375` | 3.0968 | 0.0600 | 3/4 |
| `ga` | `2.3402 -> 2.3531` | 3.0279 | 0.0071 | 1/1 |
| `ga_guarded` | `2.3402 -> 2.3527` | 3.0250 | 0.0071 | 5/5 |

## 与现有 4 个有效 Seed 的一致性

`seed47` 的基线三组 `final_global_loss` 均明显低于 5.0，数量级与 `seed42~45` 同属“健康训练”范围；`retained_negative_guard` 和 `ga_guarded` 的 update L2、accepted steps 也落在现有结果可解释区间内。`seed47` 不是新的异常点，可以作为第 5 个有效 seed 纳入主结果。

## Seed 46 处置

- `seed46` 已从主结果集合移出，保留到：
  - `formal_cloud_results/v3_dir05_seed5/appendix_seed46/`
- 顶层 `summary.csv` 不再统计 `seed46`
- 为了支持这个目录结构，更新了聚合逻辑，使其跳过任意 `appendix_*` 目录下的 `final_results.json`

## 生成产物

已更新：

- `formal_cloud_results/v3_dir05_seed5/summary.csv`
- `formal_cloud_results/v3_dir05_seed5/paper_table.csv`
- `formal_cloud_results/v3_dir05_seed5/paper_table.md`
- `formal_cloud_results/v3_dir05_seed5/figures/forget_loss_before_after.{png,pdf}`
- `formal_cloud_results/v3_dir05_seed5/figures/global_loss_final.{png,pdf}`
- `formal_cloud_results/v3_dir05_seed5/figures/update_norm_vs_global_loss.{png,pdf}`
- `formal_cloud_results/v3_dir05_seed5/figures/update_norm_vs_forget_gain.{png,pdf}`
- `formal_cloud_results/v3_dir05_seed5/figures/guard_steps.{png,pdf}`
- `formal_cloud_results/v3_dir05_seed5/figures/unlearning_time.{png,pdf}`

## 文件变更

### 源码

- 修改：`scripts/summarize_experiments.py`

### 测试

- 修改：`tests/test_summarize_experiments.py`

### 结果目录整理

- 新纳入主目录：`formal_cloud_results/v3_dir05_seed5/seed47/`
- 移动到附录：`formal_cloud_results/v3_dir05_seed5/appendix_seed46/`

## 验证

结果文件核对：

- `summary.csv` 行数：`35`
- seed 集合：`seed42, seed43, seed44, seed45, seed47`
- `appendix_seed46/` 存在
- `figures/` 共重新生成 `12` 个文件

## 偏差与说明

- 本轮不再需要追加云端实验；`seed47` 已满足任务书中的“补一个健康 5th seed”目标。
- 用户下载归档后留下的嵌套解压目录仍在本地，但主结果目录已经整理完毕；本次未删除该副本，避免误删用户文件。
