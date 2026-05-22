# Codex 任务委托：FedHDS V3 实验补强（v3.2 增量，简化版）

## 一、项目背景（30 秒读完）

仓库根目录：F:\FedHDS-zhaoge
论文当前版本：paper_draft_submission_v2_uestc_template.docx
主计划文档：F:\FedHDS-zhaoge\第三次补强计划.md（v2.1，先读一遍）
当前结果目录：formal_cloud_results/v3_dir05_seed5/

V3 core suite 已完成：Qwen2-0.5B + Dolly + 20 clients + Dirichlet α=0.5
+ 5 seeds (42-46) × 7 methods (fl, fedhds, retrain_oracle, forget_batch_hessian,
retained_negative_guard, ga, ga_guarded)。

数据复盘后发现 **seed 46 训练发散**，导致主表实际只有 4 个有效 seed。
本次增量只做一件事：**补一个有效 seed**，让 "5 valid seeds" 叙事完整。

MIA 信号弱的问题（loss-based MIA AUC ≈ 0.5）**本次不修**。
经过成本/收益评估，论文将把 MIA 作为 supplementary indicator 主动声明
其在当前 setup 下信号弱，作为 future work 处理。**不要顺手做 MIA 强化、
不要做 LiRA、不要换 partition α**。

---

## 二、唯一任务：补一个有效 seed 替换 seed 46

### 2.1 现状证据（请先确认你看懂）

打开 formal_cloud_results/v3_dir05_seed5/summary.csv 验证以下三点：

- seed46/fedhds、seed46/fl、seed46/retrain_oracle 三行的
  `final_global_loss` 全部为 10.0
- `round2_global_loss` 也是 10.0（说明 round 2 即已发散，与 unlearning 无关）
- seed46 各行的 `mia_member_count` = 112，是 5 seeds 中最小
  （其他 seed: 261-638），疑因 Dirichlet 划分极端

结论：seed46 是 training-time 发散，不是方法失败，不能进 method 比较。

### 2.2 执行步骤

**步骤 1：环境与脚本核对（10 分钟）**

在跑实验之前，先确认以下文件存在并可执行：

- `scripts/run_v3_core_suite.sh`（V3 套件运行入口）
- `scripts/summarize_experiments.py`（产生 summary.csv）
- `scripts/make_paper_table.py`（产生 paper_table.md）
- `utils_data/load_data.py`（Dirichlet 划分实现，line 94 附近）

如果 `scripts/run_v3_core_suite.sh` 不存在或不可执行，停下来报告，
**不要自己重新发明运行入口**。

**步骤 2：跑 seed 47（半天）**

使用 seed 47 跑完整 7 方法 suite，配置必须与现有 seed 42-45 完全一致：

- `--seed 47`
- `--iid dir0.5`
- `--num_clients 20`
- forget client 0
- 其余超参（learning rate、local_step、rounds、data_sample 等）**严格沿用
  seed 42-45 在 summary.csv 中记录的值**，从 csv 反查后再执行，
  **不要凭记忆设置**

输出目录命名：`formal_cloud_results/v3_dir05_seed5/seed47/`，目录结构与
`seed42/` 完全对齐。

**步骤 3：健康检查（关键，不能跳过）**

跑完 seed 47 后，验证以下 3 个硬指标：

1. `seed47/fedhds/.../final_results.json` → `final_global_loss < 5.0`
2. `seed47/fl/.../final_results.json` → `final_global_loss < 5.0`
3. `seed47/retrain_oracle/.../final_results.json` → `final_global_loss < 5.0`
4. 任一组的 `mia_member_count ≥ 150`

**4 条全部通过** → seed 47 健康，进步骤 5
**任一条不过** → seed 47 也是异常，进步骤 4

**步骤 4：fallback chain（仅在 seed 47 不健康时执行）**

按顺序尝试以下方案，每次完成后回步骤 3 重新做健康检查：

| 尝试 | 配置 | 决策 |
|---|---|---|
| Try 2 | seed 48，α=0.5 | 健康即用 |
| Try 3 | seed 49，α=0.5 | 健康即用 |
| Try 4（兜底） | seed 47，**α=0.7**（轻度异构） | 健康即用，
    但需在报告里写明使用了 α=0.7 而非 α=0.5 |

如果 Try 4 仍不健康，**停下来报告**，不要无限尝试。

**步骤 5：重新生成聚合产物**

拿到健康新 seed 后，更新顶层文件：

1. 把 seed46 整组移到
   `formal_cloud_results/v3_dir05_seed5/appendix_seed46/`
   （保留原始数据，不删除）
2. 在 `formal_cloud_results/v3_dir05_seed5/` 顶层重新跑
   `scripts/summarize_experiments.py` 和 `scripts/make_paper_table.py`，
   生成新的 `summary.csv`、`paper_table.csv`、`paper_table.md`
3. 验证新 summary.csv 恰好包含 5 个有效 seed × 7 methods = 35 行
4. 验证 paper_table.md 中新 seed 行的 forget loss / global loss / update L2
   数量级与现有 4 个有效 seed 一致（无 outlier）

### 2.3 不要做的事

- 不要重跑 seed 42-45（它们结果已经稳定）
- 不要修改任何 method 实现或训练超参
- 不要改 LoRA target modules
- 不要顺手补 MIA、LiRA、α=0.1、更大模型
- 不要 cosmetic refactor 现有脚本
- 不要删除 seed 46 数据（移到 appendix 子目录即可）

---

## 三、通用约束

1. 不修改 main.py / server.py / client.py 的训练或 unlearning 逻辑
2. 单条路径阻塞超过 4 小时立刻停下来报告
3. 完成后回报：
   - 简短总结（≤200 字）
   - 验收清单勾选状态
   - 新增 / 修改文件列表
   - 关键数字（新 seed 各组 final_global_loss、forget loss Δ、accepted_steps）
4. 把汇总写到 `F:\FedHDS-zhaoge\v3_2_increment_report.md`

---

## 四、最终验收清单

- [ ] `formal_cloud_results/v3_dir05_seed5/seed{47或后续}/` 存在且包含
      完整 7 个方法子目录
- [ ] 该 seed 的 fedhds、fl、retrain_oracle 三组 `final_global_loss < 5.0`
- [ ] 该 seed 的 `mia_member_count ≥ 150`
- [ ] seed 46 整组已移到 `appendix_seed46/`，未删除
- [ ] `formal_cloud_results/v3_dir05_seed5/summary.csv` 重新生成，
      恰好 35 行（5 有效 seed × 7 methods）
- [ ] `formal_cloud_results/v3_dir05_seed5/paper_table.md` 重新生成
- [ ] 新 seed 行的指标无明显 outlier（与现有 4 个 seed 量级一致）
- [ ] `v3_2_increment_report.md` 已写

---

## 五、时间预算与降级

预算：**半天**（happy path，seed 47 一次健康）

降级链：
- seed 47 不健康 → 立刻试 seed 48，再不行试 seed 49（再加 0.5~1 天）
- seed 47/48/49 都不健康 → 切 α=0.7 重试 seed 47（再加 0.5 天）
- 仍不健康 → 停止，写报告说明"在当前 setup 下补 seed 失败"，
  论文继续用 4 个有效 seed 投稿

**云成本上限：1.5 天**。超过这个预算直接停，不要硬怼。

---

## 六、完成后向用户回报的报告模板

写到 `F:\FedHDS-zhaoge\v3_2_increment_report.md`，至少包含：

```
# V3.2 Increment Report

## 执行结果
- 新增有效 seed: seed XX (α=Y)
- 健康检查: fedhds=X.XX, fl=X.XX, retrain_oracle=X.XX, member_count=XXX

## 新 seed 在 7 方法上的关键数字
| Method | Δ Forget Loss | Final Global Loss | Update L2 | Accepted Steps |
| ... |

## 与现有 4 个 seed 的一致性
（一句话评价：是否与现有数据量级一致 / 是否有 outlier）

## seed 46 处置
- 已移至 appendix_seed46/
- 顶层 summary.csv 不再包含 seed 46

## 文件变更
- 新增: ...
- 修改: ...
- 移动: ...

## 异常或偏离
（如有：什么 fallback 触发了、什么决策做了、用户是否需要确认）
```

---

## 七、一句话总结

只做一件事：拿到一个健康的 5th seed，更新聚合产物，把 seed 46 移到附录。
**不要做任何其他事**。Happy path 半天，绝对上限 1.5 天。
