FedHDS 小论文最小方法方案：当前进度精简版

最后更新：2026-05-04

说明：
1. 本文件用于后续接手和论文写作，不再保留 SSH 教程、截图过程、误粘贴命令、下载进度、终端乱码等流水账。
2. 旧版完整过程已备份到：
   F:\FedHDS-zhaoge\readme3_full_history_20260503.txt
3. 后续所有代码改动、实验配置、结果目录、关键数值和阶段性结论，继续追加到本文件。


一、项目当前状态

第一阶段已经完成。

已完成内容：
1. retained-set Hessian unlearning 核心代码改造。
2. clipping 与 stepped trust-region guard 稳定性机制。
3. 结果汇总、论文表格和出图工具链。
4. AutoDL 云端环境搭建与 smoke 验证。
5. Qwen2-0.5B 第一阶段正式最小四组实验。
6. 第一阶段结果文字版和 Word 报告版整理。

当前不建议继续做大量相近参数 sweep。下一步优先进入论文写作整理，必要时再进入第二阶段扩大同模型实验规模。


二、核心代码改造

涉及主要文件：
1. client.py
2. server.py
3. main.py
4. scripts/summarize_experiments.py
5. scripts/make_paper_table.py
6. scripts/plot_experiments.py
7. scripts/run_cloud_experiments.sh

已完成的关键功能：

1. retained reference 采样
   - client.py 新增 `sample_reference_examples(sample_size, shuffle=False)`。
   - 从 `self.full_train_dataset` 中采样，不依赖当前 train loader。

2. retained-set Hessian
   - server.py 新增 retained reference loader。
   - 使用非 forget client 的 reference samples 构建 `D_ref`。
   - retained-set 路径使用 `D_ref` 的平均 loss 作为 Hessian reference。
   - 保留 forget-batch Hessian baseline。

3. unlearning 稳定性参数
   - main.py 已新增：
     - `--use_retained_hessian`
     - `--hessian_ref_per_client`
     - `--unlearn_max_update_norm`
     - `--unlearn_num_steps`
     - `--unlearn_ref_loss_guard_ratio`
     - `--unlearn_global_loss_guard_max`

4. HVP 和二阶计算稳定性
   - unlearning 前关闭 gradient checkpointing。
   - 二阶 HVP 路径强制使用 math SDP attention。
   - retained-set Hessian reference 可用。
   - inverse_hvp * scale 的 global L2 clipping 可用。

5. stepped trust-region guard
   - 先计算总 update。
   - 先按 `max_update_norm` 裁剪。
   - 再切成 `unlearn_num_steps` 个小步。
   - 每步后可检查 global loss / reference loss。
   - 超过 guard 时回滚当前小步并停止。
   - 记录 actual_param_delta_l2_norm、steps accepted、guard stop reason 等指标。

6. 工具链
   - `scripts/summarize_experiments.py`：递归汇总 `final_results.json` 到 `summary.csv`。
   - `scripts/make_paper_table.py`：生成 `paper_table.csv` 和 `paper_table.md`。
   - `scripts/plot_experiments.py`：生成 forget/global/update/guard/time 相关图。
   - `scripts/run_cloud_experiments.sh`：云端四组实验模板，支持 `RUN_SMOKE=1`。

语法验证：
```bash
python -m py_compile main.py server.py client.py scripts/summarize_experiments.py scripts/make_paper_table.py scripts/plot_experiments.py
```
已在本地和云端通过。


三、本地阶段重要结果

本地结果根目录：
F:\FedHDS-zhaoge\formal_local_compare_20260429

主要结论：
1. retained-set Hessian 方向能提高 forget client loss，但会损伤 global utility。
2. 单步 retained-set Hessian 不稳定；eta=0.01 未控制时 global loss 可崩到约 9.93。
3. global L2 clipping 能避免灾难性崩溃。
4. stepped guard 能把不可控单步更新变成可截停、可回滚的 trust-region 更新。
5. 本地小实验中，在 global loss < 0.8 约束下，较好的候选为：
   - max_norm=0.3
   - steps=5 或 steps=10
   - global_guard=0.8
   - actual update L2 约 0.18
   - forget loss 约 2.578
   - global loss 约 0.775-0.778
6. 如果允许 global loss 到 0.87 附近，可用：
   - max_norm=0.3
   - steps=5
   - global_guard=0.87
   - forget loss 2.6534
   - global loss 0.8646

本地阶段已经足够说明方法链路和稳定性机制，不建议继续做大量相近 sweep。


四、AutoDL 云端环境

云端第一阶段使用：
1. GPU：AutoDL 单卡 RTX 5090 32GB
2. 镜像：PyTorch 2.8.0 / Python 3.12 / CUDA 12.8
3. 模型：Qwen2-0.5B
4. 数据集：Dolly

云端关键环境结论：
1. PyTorch/CUDA 可用：
   - torch 2.8.0+cu128
   - torch.cuda.is_available() = True
   - GPU = NVIDIA GeForce RTX 5090
2. Hugging Face 直连不可用，需使用：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   export HF_HUB_DISABLE_XET=1
   export HF_HOME=/root/autodl-tmp/cache
   ```
3. hf-mirror 下载大权重较慢，最终使用 ModelScope 下载模型。
4. 云端本地模型路径：
   ```text
   /root/autodl-tmp/models/Qwen2-0.5B-ms
   ```
5. 该本地模型已验证可加载：
   - Qwen2ForCausalLM
   - Qwen2Tokenizer


五、云端 smoke

云端 smoke 已完成。

smoke 结果目录：
1. 云端：
   /root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results/smoke
2. 本地：
   F:\FedHDS-zhaoge\formal_cloud_results_smoke

smoke 完成内容：
1. FL 可跑。
2. FedHDS 可跑。
3. forget-batch Hessian unlearning 可跑。
4. retained-set Hessian + guard 可跑。
5. summary、paper_table 和 figures 均可生成。

说明：
smoke 只验证云端链路，不作为论文正式结果。


六、第一阶段云端正式实验

正式 combined 结果目录：
1. 本地主结果：
   F:\FedHDS-zhaoge\formal_cloud_results_formal_minimal_20260503_combined
2. 本地完整归档：
   F:\FedHDS-zhaoge\formal_cloud_results_autodl_20260503
3. 云端原始目录：
   /root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results

正式 combined 关键文件：
1. summary.csv
2. paper_table.csv
3. paper_table.md
4. figures/*.png
5. figures/*.pdf
6. 各组 final_results.json 和 logs 位于完整归档中。

正式实验配置：
1. model = /root/autodl-tmp/models/Qwen2-0.5B-ms
2. dataset = dolly
3. data_sample = 0.2
4. num_clients = 20
5. client fraction k = 0.2
6. rounds = 10
7. local_step = 2
8. batch_size = 1
9. max_length = 64
10. iid = 0
11. forget_client_idx = 0
12. lissa_depth = 1
13. lissa_damping = 0.01
14. unlearn_eta = 0.01

正式四组：
1. FL
2. FedHDS
3. FedHDS + forget-batch Hessian
4. FedHDS + retained-set Hessian + stepped guard


七、第一阶段正式主结果

主表来自：
F:\FedHDS-zhaoge\formal_cloud_results_formal_minimal_20260503_combined\summary.csv

| 组别 | Hessian | Forget Loss | Final Global Loss | Update L2 | Steps | Time |
|---|---|---:|---:|---:|---:|---:|
| FL | none | - | 1.6443 | - | - | - |
| FedHDS | none | - | 1.6657 | - | - | - |
| FedHDS + forget-batch Hessian | forget-batch | 2.3381 -> 2.3489 | 1.6867 | 0.0872 | 1/1 | 1.44s |
| FedHDS + retained-set Hessian + stepped guard | retained-set | 2.3381 -> 2.3460 | 1.6615 | 0.0427 | 5/5 | 71.13s |

更精确数值：

1. FL
   - round2_global_loss = 1.783530279994011
   - final_global_loss = 1.6442784418662388

2. FedHDS
   - round2_global_loss = 1.7811286449432373
   - final_global_loss = 1.6656732633709908

3. FedHDS + forget-batch Hessian
   - global loss before unlearning = 1.6656732633709908
   - global loss after unlearning = 1.6867005775372188
   - forget client loss = 2.338137240240387 -> 2.3488887650709525
   - unlearning time = 1.4364721775054932s
   - last_hvp_l2_norm = 166.2401686318689
   - actual_param_delta_l2_norm = 0.0871853766921942
   - steps accepted = 1/1

4. FedHDS + retained-set Hessian + stepped guard
   - group = retained_guard_ref1_guard170
   - global loss before unlearning = 1.6656732633709908
   - global loss after unlearning = 1.6614613855878513
   - forget client loss = 2.338137240240387 -> 2.346023207786037
   - unlearning time = 71.13468289375305s
   - retained reference set size = 19
   - last_hvp_l2_norm = 81.22970454183088
   - actual_param_delta_l2_norm = 0.0426737862876952
   - max_update_norm = 0.3
   - unlearn_num_steps = 5
   - steps accepted = 5/5
   - guard stop reason = completed
   - global guard max = 1.7
   - clipping enabled = True
   - clipping applied = False


八、云端正式实验中的关键修正

1. retained-set Hessian OOM
   - 原正式 retained_guard 使用 `hessian_ref_per_client=4`。
   - 在 RTX 5090 32GB 上 HVP 阶段 CUDA OOM。
   - 修正为 `hessian_ref_per_client=1` 后可运行。
   - 该修正属于 OOM 修复，不是参数 sweep。

2. global guard 阈值尺度不匹配
   - 本地小实验中 global loss 约 0.33，因此 `global_guard=0.8` 合理。
   - 云端正式实验中 unlearning 前 global loss 已是 1.6657。
   - 继续使用 `global_guard=0.8` 会导致 0 步接受。
   - 修正为 `global_guard=1.7` 后，retained guard 5/5 小步接受。
   - 该修正属于 loss 尺度匹配，不是 sweep。

云端正式 retained-set guard 可用配置：
```text
hessian_ref_per_client = 1
unlearn_grad_sample_size = 8
unlearn_eta = 0.01
max_update_norm = 0.3
unlearn_num_steps = 5
global_guard = 1.7
```


九、阶段性结论

1. 第一阶段云端正式最小四组实验已经完成。
2. 当前代码链路已经在云端真实 GPU 环境中闭环。
3. forget-batch Hessian baseline 能带来略强的 forget loss 提升：
   - 2.3381 -> 2.3489
   - 提升约 0.0108
4. retained-set Hessian + stepped guard 也能带来小幅 forget loss 提升：
   - 2.3381 -> 2.3460
   - 提升约 0.0079
5. forget-batch Hessian 使 global loss 变差：
   - 1.6657 -> 1.6867
6. retained-set guard 没有使 global loss 变差：
   - 1.6657 -> 1.6615
7. retained-set guard 的遗忘提升弱于 forget-batch baseline，但 utility 更稳。
8. retained-set guard 的计算成本明显更高：
   - 71.13s vs 1.44s
9. 当前不能宣称 retained-set Hessian 全面优于 forget-batch baseline。
10. 更稳妥的论文表述是：
    retained-set curvature approximation + clipping + stepped trust-region guard 提供了一条可运行、可控、utility 稳定的 federated unlearning 路径，但当前正式配置下 forgetting 提升较弱，且二阶计算成本较高。


十、retained guard 的优缺点

优点：
1. 方法逻辑清晰：
   - forget client gradient 定义“删除什么”。
   - retained reference curvature 定义“保留任务附近的曲率结构”。
   - clipping 和 stepped guard 控制更新幅度。
2. 更新过程可控：
   - 5/5 小步被接受。
   - actual update L2 明确记录为 0.0427。
3. global utility 稳：
   - global loss 未恶化，略低于 unlearning 前。
4. 工程链路完整：
   - 可训练、可 unlearn、可汇总、可出表、可出图。

缺点：
1. forgetting 提升弱：
   - retained guard 提升约 0.0079，低于 forget-batch 的约 0.0108。
2. 计算耗时高：
   - 71.13s，远高于 forget-batch 的 1.44s。
3. 显存敏感：
   - hessian_ref_per_client=4 在 32GB GPU 上 OOM。
4. guard 阈值依赖 loss 尺度：
   - 本地 0.8 不能直接迁移到云端正式实验。


十一、论文写法建议

方法段落建议围绕三部分：
1. FedHDS retained-set curvature approximation
2. global update norm clipping
3. stepped trust-region guard

实验消融逻辑建议：
1. FL vs FedHDS：说明训练底座和 global utility。
2. FedHDS + forget-batch Hessian：作为 unlearning baseline。
3. FedHDS + retained-set Hessian + guard：展示 retained curvature + guard 的可控更新。
4. OOM 和 guard 阈值尺度：放在 limitation 或 implementation notes 中。

论文结论边界：
1. 可以说：
   - retained-set guard 可运行。
   - retained-set guard 可控。
   - retained-set guard 在当前正式配置下保持 global utility 稳定。
   - clipping/guard 能避免 retained-set Hessian 的不可控更新。
2. 不应说：
   - retained-set Hessian 全面优于 forget-batch baseline。
   - 当前方法显著提升 forgetting。
   - 当前结果已经充分证明大规模泛化能力。


十二、已生成整理文件

1. 文字整理版：
   F:\FedHDS-zhaoge\readme_result1.txt

2. Word 报告版：
   F:\FedHDS-zhaoge\readme_result_1.docx

3. combined 正式结果：
   F:\FedHDS-zhaoge\formal_cloud_results_formal_minimal_20260503_combined

4. 完整云端归档：
   F:\FedHDS-zhaoge\formal_cloud_results_autodl_20260503


十三、下一步建议

短期：
1. 先基于第一阶段结果写论文方法段、实验设置段、结果分析和 limitation。
2. 不建议立刻继续大量相近 sweep。
3. 如果 AutoDL 暂时不用，应关机止损。

第二阶段：
1. 仍使用 Qwen2-0.5B。
2. 优先扩大：
   - data_sample
   - rounds
   - seed
   - forget client
3. 保留当前已验证的 retained guard 安全配置：
   - hessian_ref_per_client=1
   - max_update_norm=0.3
   - steps=5
   - global_guard 按训练后 global loss 尺度设置

第三阶段：
1. 再考虑更大模型：
   - Qwen2.5-1.5B
   - Qwen2-1.5B
   - Qwen2.5-3B
2. 不建议直接上 7B。


十五、阶段二当前建议运行方案

用户当前 AutoDL 环境尚未释放，依赖、模型和项目都已经可用，因此阶段二不需要重装环境。

阶段二推荐先跑一组最小放大实验：

```text
data_sample = 0.4
rounds = 20
num_clients = 20
client fraction = 0.2
local_step = 2
hessian_ref_per_client = 1
unlearn_grad_sample_size = 8
max_update_norm = 0.3
steps = 5
```

global guard 不固定写死，而是在跑完 FedHDS 后自动读取 FedHDS final_global_loss，并设置为：

```text
global_guard = FedHDS final_global_loss + 0.05
```

这样可以避免第一阶段出现过的两个问题：

1. `global_guard=0.8` 与云端 loss 尺度不匹配导致 0 步接受。
2. `hessian_ref_per_client=4` 导致 32GB GPU OOM。

阶段二建议使用 `nohup` 后台运行，而不是 screen。这样 SSH 断开后仍可继续跑，且用户不用盯着乱码进度条。

阶段二推荐结果目录：

```text
formal_cloud_results/stage2_dsample04_round20_20260503
```

阶段二脚本：
```text
F:\FedHDS-zhaoge\scripts\run_stage2_dsample04_round20.sh
```

说明：
1. AutoDL 镜像中没有 `nano`。
2. 因此阶段二脚本已在本地生成，建议通过 `scp` 上传到云端：
   ```powershell
   scp -P 56117 "F:\FedHDS-zhaoge\scripts\run_stage2_dsample04_round20.sh" "root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/scripts/"
   ```
3. 云端启动命令：
   ```bash
   cd /root/autodl-tmp/FedHDS-zhaoge
   chmod +x scripts/run_stage2_dsample04_round20.sh
   nohup bash scripts/run_stage2_dsample04_round20.sh > formal_cloud_results/stage2_dsample04_round20_20260503/nohup.out 2>&1 &
   echo $! > formal_cloud_results/stage2_dsample04_round20_20260503/pid.txt
   ```


十四、阶段二放大实验建议

阶段二目标不是继续扫很多相近参数，而是让第一阶段结果更可信、更稳定，并判断 retained-set guard 在稍大规模下是否仍然可控。

根据第一阶段结果，当前最值得放大的维度是：

1. 优先增加训练轮数：
   - 从 rounds=10 增加到 rounds=20
   - 原因：第一阶段 forget loss 提升很小，说明训练底座和 unlearning 信号都偏弱；rounds 增加会直接增加实际训练更新步数，通常比单纯增加 data_sample 更可能让结果信号变清楚。

2. 第二优先增加数据比例：
   - 从 data_sample=0.2 增加到 data_sample=0.4
   - 原因：更大的数据子集能提高 eval/forget 评估稳定性，也让 retained/reference 数据更有代表性。
   - 但由于当前训练使用 batch 模式且 local_step 固定，单独增加 data_sample 不一定显著增加每轮训练强度，所以不应只增加 data_sample 而不增加 rounds。

3. 暂时不要优先增加 num_clients：
   - 保持 num_clients=20
   - 原因：更多客户端会增加数据划分和 retained reference 复杂度，也可能放大 retained Hessian 显存问题。阶段二先验证同客户端规模下的稳定性。

4. 暂时不要增加 hessian_ref_per_client：
   - 保持 hessian_ref_per_client=1
   - 原因：第一阶段 hessian_ref_per_client=4 已在 RTX 5090 32GB 上 OOM。

5. global_guard 不要固定沿用 0.8：
   - 应按该轮 FedHDS 的 unlearning 前 global loss 重新设置。
   - 建议初始规则：global_guard = FedHDS final/global-before-unlearning loss + 0.03 到 0.05。
   - 第一阶段正式实验中 FedHDS global loss 为 1.6657，因此使用 1.7 合理。

推荐阶段二最小实验：

1. 保守版：
   - data_sample=0.2
   - rounds=20
   - num_clients=20
   - retained guard: hessian_ref_per_client=1, max_norm=0.3, steps=5, global_guard 按 FedHDS loss +0.03~0.05 设置

2. 推荐版：
   - data_sample=0.4
   - rounds=20
   - num_clients=20
   - retained guard: hessian_ref_per_client=1, max_norm=0.3, steps=5, global_guard 按 FedHDS loss +0.03~0.05 设置

如果只跑一组阶段二实验，推荐直接跑“data_sample=0.4, rounds=20, num_clients=20”。这比第一阶段更有信息量，又不会像 data_sample=1.0 或更多客户端那样显著提高风险。

阶段二暂不建议：
1. 不建议 data_sample 直接上 1.0。
2. 不建议 num_clients 直接上 40 或更多。
3. 不建议 hessian_ref_per_client 从 1 增加到 4。
4. 不建议换 1.5B/3B 模型；那属于第三阶段。


十五、基于阶段二结果的阶段三实验方向

阶段二已经完成，正式结果目录为：

```text
F:\FedHDS-zhaoge\formal_cloud_results\stage2_dsample04_round20_20260503
```

阶段二核心结果如下：

1. FedHDS 在扩大设置下优于 FL：
   - FedHDS final global loss = 1.5250
   - FL final global loss = 1.5441
   - 说明 data_sample=0.4、rounds=20 后，FedHDS 可以作为更好的 utility baseline。

2. forget-batch Hessian 的 forgetting signal 更强，但损伤 global utility：
   - forget loss: 2.2581 -> 2.2866，提升约 +0.0285
   - global loss: 1.5250 -> 1.6257，恶化约 +0.1007

3. retained-set Hessian + auto guard 能保住 global utility，但 forgetting 方向不理想：
   - forget loss: 2.2581 -> 2.2329，下降约 -0.0252
   - global loss: 1.5250 -> 1.5482，仅恶化约 +0.0232
   - global guard = 1.5750，5/5 个小步全部接受

因此，阶段三不应直接扩大模型或扩大数据规模。当前瓶颈不是规模不足，而是 retained-set guarded unlearning 的遗忘方向和遗忘强度不足。直接换更大模型会增加耗时和 OOM 风险，但不一定解决 forget loss 下降的问题。

阶段三的定位应调整为：

```text
retained-set guarded unlearning 的方向验证与小范围消融修正
```

阶段三目标：

1. 验证 Hessian update 方向是否与 forget objective 对齐。
2. 在保留 global utility guard 的前提下，增强 forget loss 上升信号。
3. 找到一组比阶段二 retained_guard 更合理的 trade-off 配置。
4. 不追求大规模结论，先修正方法行为，再考虑扩大规模。

阶段三推荐保持不变的基础配置：

```text
model = Qwen2-0.5B
dataset = Dolly
num_clients = 20
client fraction k = 0.2
rounds = 20
local_step = 2
batch_size = 1
max_length = 64
data_sample = 0.4
iid = 0
forget_client_idx = 0
lissa_depth = 1
lissa_damping = 0.01
unlearn_grad_sample_size = 8
```

阶段三优先检查项：

1. Hessian update 符号方向检查
   - 当前 forget-batch Hessian 能让 forget loss 上升。
   - retained-set Hessian + guard 反而让 forget loss 下降。
   - 因此需要检查 retained-set HVP / inverse-HVP / apply update 的符号是否与“增加 forget loss”的目标一致。
   - 建议增加一个轻量测试：同一 FedHDS checkpoint 下分别尝试 `+update` 和 `-update`，观察 forget loss 与 global loss 的方向变化。

2. retained guard 参数小范围消融
   - 不重新做大范围 sweep。
   - 只围绕阶段二失败点做 4 到 6 组配置。
   - 推荐候选：
     ```text
     eta = 0.01, 0.02, 0.05
     max_update_norm = 0.3, 0.5
     global_guard = FedHDS final_global_loss + 0.05 或 +0.10
     hessian_ref_per_client = 1，必要时尝试 2
     steps = 5
     ```

3. 加入 forget-gain check
   - 当前 stepped guard 只检查 global loss，没有检查 forget loss 是否真的改善。
   - 阶段二 retained guard 说明：global loss 可以被保护住，但 forgetting 方向可能无效。
   - 阶段三建议增加一个可选接受条件：
     ```text
     global loss <= global_guard
     and forget loss >= forget_loss_before - tolerance
     ```
   - 更严格版本：
     ```text
     forget loss >= forget_loss_before + min_gain
     ```
   - 该机制可以避免“utility 被保护，但忘却目标没有实现”的小步被继续接受。

4. 结果选择标准
   阶段三不要只看 forget loss，也不要只看 global loss。建议按以下优先级判断：
   - 首先，global loss 不超过 FedHDS final_global_loss + 0.05 或 +0.10。
   - 其次，forget loss 至少不下降，最好能上升。
   - 再次，update L2 和 HVP norm 不能异常放大。
   - 最后，unlearning time 可接受，不把运行时间作为第一目标。

阶段三暂不建议：

1. 暂不换 1.5B、3B 或 7B 模型。
2. 暂不把 data_sample 直接扩大到 1.0。
3. 暂不增加 num_clients 到 40 或更多。
4. 暂不把 hessian_ref_per_client 直接增加到 4，因为第一阶段已经证明 32GB GPU 上有 OOM 风险。

阶段三完成后的扩大规模路线：

如果阶段三能找到一组 retained guard 配置，使得：

```text
forget loss 不下降或小幅上升
global loss 控制在 FedHDS baseline + 0.05 到 +0.10 内
运行稳定且无 OOM
```

则再进入后续扩大规模阶段。扩大规模的顺序建议为：

1. 先扩大 data_sample：0.4 -> 0.6 或 0.8。
2. 再增加 rounds：20 -> 30。
3. 最后再考虑模型规模：Qwen2-0.5B -> Qwen2/Qwen2.5 1.5B。
4. 7B 暂时不建议作为近期目标，除非前面阶段已经得到稳定且值得放大的结论。

当前决策：

```text
可以进入阶段三。
阶段三先做 retained guard 的方向验证和小范围消融修正。
不急于扩大模型或数据规模。
最后再做扩大规模验证。
```


十六、阶段三代码修改进展

本轮已经开始实现“阶段二之后的修改方向建议”，目标是先修正 retained-set guarded unlearning 的方向验证和接受条件，而不是扩大模型或数据规模。

已修改文件：

1. `main.py`
2. `server.py`
3. `scripts/summarize_experiments.py`
4. `scripts/make_paper_table.py`

新增命令行参数：

```text
--unlearn_update_sign {positive,negative,auto}
--unlearn_direction_check
--unlearn_forget_loss_guard
--unlearn_forget_loss_min_gain
--unlearn_forget_loss_tolerance
--unlearn_forget_guard_sample_size
```

默认行为保持兼容：

```text
--unlearn_update_sign positive
不启用 --unlearn_direction_check
不启用 --unlearn_forget_loss_guard
```

也就是说，不加新参数时，旧的 forget-batch Hessian 和 retained-set Hessian 路径仍按原来的正方向更新，原有实验脚本可继续运行。

新增功能 1：Hessian update 方向验证

`server.py` 中新增方向探测逻辑：

1. 在同一个 unlearning update 上分别临时尝试正方向和负方向。
2. 每次临时更新后评估：
   - forget loss
   - global loss
3. 评估完成后回滚参数，不污染正式 unlearning。
4. 记录：
   - `unlearn_direction_probe_records`
   - `unlearn_best_probe_update_sign`
   - `unlearn_best_probe_direction_forget_delta`
   - `unlearn_best_probe_direction_global_loss_after`

如果使用：

```text
--unlearn_update_sign auto
```

则代码会在满足 global guard 的候选方向中，自动选择 forget loss 增益更大的方向。若没有候选方向满足 global guard，则在两个方向中选择 forget loss 增益更大的方向。

如果使用：

```text
--unlearn_direction_check --unlearn_update_sign positive
```

则只做正负方向探测和记录，但正式更新仍按指定的 positive 方向执行。

新增功能 2：forget-gain check

`server.py` 中的 stepped guard 已新增可选 forget loss 接受条件。启用：

```text
--unlearn_forget_loss_guard
```

后，每个小步除了原有 reference/global guard 外，还会评估 forget loss。

两种接受规则：

1. 非下降规则：
   ```text
   forget_loss_after_step >= forget_loss_before_guard - tolerance
   ```
   对应参数：
   ```text
   --unlearn_forget_loss_tolerance
   ```

2. 最小增益规则：
   ```text
   forget_loss_after_step >= forget_loss_before_guard + min_gain
   ```
   对应参数：
   ```text
   --unlearn_forget_loss_min_gain
   ```

若小步不满足 forget loss guard，则回滚当前小步并停止，`unlearn_guard_stop_reason` 记录为：

```text
forget_loss_guard
```

新增汇总字段：

`scripts/summarize_experiments.py` 已加入阶段三相关字段，后续跑完实验后 `summary.csv` 会包含：

```text
unlearn_update_sign_mode
unlearn_selected_update_sign
unlearn_direction_check_enabled
unlearn_best_probe_update_sign
unlearn_best_probe_direction_forget_delta
unlearn_best_probe_direction_global_loss_after
unlearn_forget_loss_guard_enabled
unlearn_forget_guard_sample_size
unlearn_forget_loss_min_gain
unlearn_forget_loss_tolerance
unlearn_forget_loss_before_guard
unlearn_forget_loss_limit
unlearn_forget_loss_after_last_accepted
```

`scripts/make_paper_table.py` 已更新，后续 `paper_table.md` 能显示：

1. 自动选择或手动指定的 update sign。
2. forget guard 配置。
3. 方向探测中最优方向的 forget delta。
4. guard 是否因为 forget loss 不满足条件而停止。

建议阶段三第一轮运行方式：

```text
保持阶段二基础配置：
model = Qwen2-0.5B
data_sample = 0.4
rounds = 20
num_clients = 20
hessian_ref_per_client = 1
unlearn_grad_sample_size = 8
max_update_norm = 0.3
steps = 5
global_guard = FedHDS final_global_loss + 0.05 或 +0.10
```

优先跑一组方向验证：

```text
--use_retained_hessian
--unlearn_update_sign auto
--unlearn_direction_check
--unlearn_global_loss_guard_max <FedHDS_loss_plus_margin>
```

再跑一组带 forget-gain check 的 retained guard：

```text
--use_retained_hessian
--unlearn_update_sign auto
--unlearn_forget_loss_guard
--unlearn_forget_loss_tolerance 0.0
--unlearn_global_loss_guard_max <FedHDS_loss_plus_margin>
```

如果该配置出现 0 步接受，再尝试放宽为：

```text
--unlearn_forget_loss_tolerance 0.005
```

或者把 global guard 从 FedHDS loss +0.05 放宽到 +0.10。

当前本地验证：

1. 已用 AST 方式检查 `main.py`、`server.py`、`client.py` 和脚本语法通过。
2. 已用旧阶段二结果重新运行 `summarize_experiments.py`，新字段兼容旧结果。
3. 已用旧阶段二结果重新运行 `make_paper_table.py`，表格生成正常。


十七、2026-05-04 阶段三最小验证补跑计划

当前决策：

```text
明天只补跑阶段三最小验证。
不重跑阶段一、阶段二完整实验。
不直接换大模型或大数据重新跑完整实验。
```

原因：

1. 阶段一和阶段二正式结果已经下载到本地，可以继续作为论文主实验依据。
2. 2026-05-03 已经运行过阶段三最小验证，并观察到较理想结论：
   - forget loss 不下降或上升
   - global loss 仍然可控
3. 但阶段三结果文件没有及时从云端下载回来，原云服务器已关机并被其他人租用，因此本地缺少阶段三的 `final_results.json`、`summary.csv`、`paper_table.md` 和日志证据。
4. 论文写作和后续复现不能只依赖记忆，因此需要补跑一次阶段三最小验证，把结果文件下载并归档。

明天优先运行脚本：

```bash
cd /root/autodl-tmp/FedHDS-zhaoge
chmod +x scripts/run_stage3_direction_forget_guard.sh
nohup bash scripts/run_stage3_direction_forget_guard.sh > formal_cloud_results/stage3_direction_forget_guard_20260503/nohup.out 2>&1 &
echo $! > formal_cloud_results/stage3_direction_forget_guard_20260503/pid.txt
```

该脚本会跑：

1. FedHDS baseline，用于自动读取 `final_global_loss` 并设置 guard。
2. retained auto direction check，验证正负 update sign。
3. retained auto + forget-loss guard，要求 forget loss 不下降。
4. retained auto + 放宽 tolerance/global guard 的备用配置。
5. 自动生成：
   - `summary.csv`
   - `paper_table.csv`
   - `paper_table.md`
   - `figures/*.png`
   - `figures/*.pdf`

阶段三补跑重点检查字段：

```text
forget_client_loss_before_unlearning
forget_client_loss_after_unlearning
global_loss_before_unlearning
global_loss_after_unlearning
unlearn_update_sign_mode
unlearn_selected_update_sign
unlearn_best_probe_update_sign
unlearn_best_probe_direction_forget_delta
unlearn_best_probe_direction_global_loss_after
unlearn_forget_loss_guard_enabled
unlearn_forget_loss_before_guard
unlearn_forget_loss_limit
unlearn_forget_loss_after_last_accepted
unlearn_guard_stop_reason
unlearn_steps_accepted
unlearn_steps_attempted
actual_param_delta_l2_norm
```

阶段三成功标准：

```text
forget loss 不下降或小幅上升
global loss 控制在 FedHDS baseline + 0.05 或 +0.10 范围内
update sign / direction probe 有明确记录
forget-loss guard 未接受无效遗忘方向
运行无 OOM、无中断
```

跑完后必须立刻打包：

```bash
cd /root/autodl-tmp/FedHDS-zhaoge
tar -czf stage3_direction_forget_guard_20260503.tar.gz \
  formal_cloud_results/stage3_direction_forget_guard_20260503 \
  main.py server.py client.py \
  scripts/run_stage3_direction_forget_guard.sh \
  scripts/run_stage3b_negative_norm_sweep.sh \
  scripts/summarize_experiments.py \
  scripts/make_paper_table.py \
  scripts/plot_experiments.py
```

然后从本地 PowerShell 下载，端口和地址按新租到的云服务器 SSH 信息替换：

```powershell
scp -P 新端口 root@新地址:/root/autodl-tmp/FedHDS-zhaoge/stage3_direction_forget_guard_20260503.tar.gz F:\FedHDS-zhaoge\
```

下载后在本地解压到项目目录：

```powershell
cd F:\FedHDS-zhaoge
tar -xzf stage3_direction_forget_guard_20260503.tar.gz
```

下载后本地需要确认：

```powershell
Get-Item F:\FedHDS-zhaoge\stage3_direction_forget_guard_20260503.tar.gz
Get-ChildItem F:\FedHDS-zhaoge\formal_cloud_results\stage3_direction_forget_guard_20260503
Get-Content F:\FedHDS-zhaoge\formal_cloud_results\stage3_direction_forget_guard_20260503\summary.csv
Get-Content F:\FedHDS-zhaoge\formal_cloud_results\stage3_direction_forget_guard_20260503\paper_table.md
```

如果阶段三复现成功，论文主线可以写成：

```text
在阶段二中，retained-set guard 主要体现为 utility preservation，但 forgetting 方向不足。
阶段三通过 update direction probing 和 forget-loss guard 修正了该问题：
自动选择更合适的 update sign，并用 forget-loss guard 避免接受 forget loss 下降的小步。
最终 retained-set guarded unlearning 可以在 global utility 可控的前提下，使 forget loss 不下降或小幅上升。
```

后续扩大策略：

1. 阶段三结果下载并确认前，不进入大模型或大数据实验。
2. 阶段三复现成功后，优先考虑多 seed 或更换 forget client，而不是立刻换大模型。
3. 若要扩大规模，推荐顺序为：
   - Qwen2-0.5B，data_sample 0.4，rounds 20，多 seed / 多 forget client
   - Qwen2-0.5B，data_sample 0.6 或 0.8
   - Qwen2/Qwen2.5-1.5B 小规模补充验证
4. 暂不建议直接上 7B，也不建议一开始就 data_sample=1.0 或 num_clients=40。

十八、2026-05-04 阶段三完整实验后续任务安排

当前状态：

1. 阶段三完整实验已经在云服务器后台启动。
2. 实验脚本为：
   ```text
   scripts/run_stage3_full_direction_guard.sh
   ```
3. 云端结果目录为：
   ```text
   /root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results/stage3_full_direction_guard_20260504
   ```
4. 该实验不是更换大模型或大数据，而是在阶段二相同基础配置上，把阶段三从“最小验证”补成“完整对照消融”。
5. 当前不需要重复启动实验，等待后台任务完成即可。

本次阶段三完整实验配置：

```text
model = Qwen2-0.5B
model path = /root/autodl-tmp/models/Qwen2-0.5B-ms
dataset = Dolly
data_sample = 0.4
rounds = 20
num_clients = 20
client fraction k = 0.2
local_step = 2
batch_size = 1
max_length = 64
iid = 0
forget_client_idx = 0
lissa_depth = 1
lissa_damping = 0.01
unlearn_grad_sample_size = 8
hessian_ref_per_client = 1
unlearn_eta = 0.01
retained update steps = 5
```

本次阶段三完整实验包含 9 组：

1. `fl`
2. `fedhds`
3. `forget_batch_hessian`
4. `retained_auto_direction_guard05`
5. `retained_auto_forget_guard_tol0_guard05`
6. `retained_auto_forget_guard_tol005_guard10`
7. `retained_negative_norm005_guard05`
8. `retained_negative_norm010_guard10`
9. `retained_negative_norm015_guard10`

这些组的作用：

1. `fl` 和 `fedhds` 用于确认训练侧 utility baseline。
2. `forget_batch_hessian` 作为传统 forget-batch Hessian unlearning baseline。
3. `retained_auto_direction_guard05` 用于验证 retained-set Hessian 在自动方向选择下的行为。
4. 两个 `retained_auto_forget_guard` 组用于验证 forget-loss guard 是否能阻止 forget loss 下降的小步被接受。
5. 三个 `retained_negative_norm` 组用于检查 negative update sign 在不同更新范数下的 forgetting / utility trade-off。

等待实验期间需要做的事：

1. 不要重复启动 `run_stage3_full_direction_guard.sh`。
2. 可以每隔 20 到 30 分钟查看一次日志。
3. 如果只是查看日志，按 `Ctrl + C` 退出不会停止实验。
4. 若日志停留时间异常过长，再检查进程和 GPU 状态。

本地 PowerShell 查看日志命令：

```powershell
ssh -p 56117 root@connect.westc.seetacloud.com 'cd /root/autodl-tmp/FedHDS-zhaoge && tail -n 80 -f formal_cloud_results/stage3_full_direction_guard_20260504/nohup.out'
```

AutoDL 远程黑窗查看日志命令：

```bash
cd /root/autodl-tmp/FedHDS-zhaoge
tail -n 80 -f formal_cloud_results/stage3_full_direction_guard_20260504/nohup.out
```

实验完成标志：

```text
[DONE] Stage 3 full ablation finished
[DONE] Archive: /root/autodl-tmp/FedHDS-zhaoge/stage3_full_direction_guard_20260504.tar.gz
```

实验完成后第一步：下载归档文件。

本地 PowerShell 下载命令：

```powershell
scp -P 56117 root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/stage3_full_direction_guard_20260504.tar.gz F:\FedHDS-zhaoge\
```

下载后本地解压命令：

```powershell
cd F:\FedHDS-zhaoge
tar -xzf F:\FedHDS-zhaoge\stage3_full_direction_guard_20260504.tar.gz -C F:\FedHDS-zhaoge\
```

解压后的本地结果目录应为：

```text
F:\FedHDS-zhaoge\formal_cloud_results\stage3_full_direction_guard_20260504
```

下载后需要检查的关键文件：

```text
F:\FedHDS-zhaoge\formal_cloud_results\stage3_full_direction_guard_20260504\summary.csv
F:\FedHDS-zhaoge\formal_cloud_results\stage3_full_direction_guard_20260504\paper_table.md
F:\FedHDS-zhaoge\formal_cloud_results\stage3_full_direction_guard_20260504\paper_table.csv
F:\FedHDS-zhaoge\formal_cloud_results\stage3_full_direction_guard_20260504\figures\
```

还需要检查各实验组内部的：

```text
final_results.json
logs
```

下载后第二步：读取并确认阶段三完整实验数值。

重点确认字段：

```text
final_global_loss
forget_client_loss_before_unlearning
forget_client_loss_after_unlearning
global_loss_before_unlearning
global_loss_after_unlearning
actual_param_delta_l2_norm
unlearn_steps_accepted
unlearn_steps_attempted
unlearn_guard_stop_reason
unlearn_update_sign_mode
unlearn_selected_update_sign
unlearn_best_probe_update_sign
unlearn_best_probe_direction_forget_delta
unlearn_best_probe_direction_global_loss_after
unlearn_forget_loss_guard_enabled
unlearn_forget_loss_before_guard
unlearn_forget_loss_limit
unlearn_forget_loss_after_last_accepted
unlearning_time_seconds
```

阶段三完整实验的判断标准：

1. `forget_batch_hessian` 是否仍能提高 forget loss，但伴随 global loss 明显变差。
2. `retained_auto_direction_guard05` 是否复现阶段二中 retained-set guard 的 utility 可控特征。
3. `retained_auto_forget_guard` 是否能阻止 forget loss 下降的小步。
4. `retained_negative_norm005_guard05`、`retained_negative_norm010_guard10`、`retained_negative_norm015_guard10` 中，是否存在 forget loss 上升且 global loss 仍可控的配置。
5. 如果 negative sign 配置稳定表现为 forget loss 上升，阶段三结论可以写成：方向验证表明 retained-set Hessian 的有效遗忘方向与默认方向相反，加入 direction probing / forget-loss guard 后，方法可以避免无效遗忘更新。
6. 如果某些组被 global guard 截停，也要保留该结果，因为它说明遗忘强度和 utility preservation 之间存在可观察 trade-off。

实验下载并确认后，需要更新的本地文件：

1. `readme_result3.docx.txt`
2. `readme_result_3.docx.docx`
3. `Summary_of_the_three_stages_of_the_thesis.txt`
4. `Summary_of_the_three_stages_of_the_thesis.docx`
5. `paper_draft.md`
6. `paper_draft.docx`
7. 如有需要，补充或更新 `thesis_outline.md`

更新原则：

1. 把阶段三从“最小验证”改写为“完整方向与 guard 消融实验”。
2. 阶段三不再只依赖 2026-05-03 的记忆性结果，而应引用 `stage3_full_direction_guard_20260504` 的实际 `summary.csv`、`paper_table.md` 和 `final_results.json`。
3. 论文中要明确区分：
   - 阶段一：方法链路可运行，retained-set Hessian + guard 初步成立。
   - 阶段二：同模型放大后，FedHDS utility 更强，但 retained-set 默认方向导致 forget loss 下降。
   - 阶段三：通过 update sign / direction probing / forget-loss guard 修正方向问题，并给出更完整的消融证据。
4. 不应夸大结论为“大模型大数据上已验证”。
5. 可以强调阶段三提高了实验严谨性，因为它把方向选择、forget guard、negative sign 强度都放进了同一套完整对照表中。

阶段三完整实验完成后的论文主线建议：

```text
阶段一证明 retained-set curvature approximation 与 trust-region guard 的工程链路可以稳定运行。
阶段二在更长训练和更大数据比例下发现，retained-set guard 能保护 global utility，但默认 Hessian update 方向可能降低 forget loss。
阶段三进一步引入 update direction probing、forget-loss guard 和 negative-sign norm sweep，验证遗忘方向和更新强度是影响 retained-set guarded unlearning 的关键因素。
最终结果应围绕 utility-preserving unlearning trade-off 来表述，而不是简单宣称 retained-set Hessian 全面优于 forget-batch Hessian。
```

阶段三完成后的下一步决策：

1. 如果阶段三完整结果稳定支持“forget loss 不下降或上升，同时 global loss 可控”，优先进入论文定稿和图表整理。
2. 如果某些组失败，但 negative sign sweep 中有可用 trade-off，也可以写成消融发现和局限性，不必重跑全部实验。
3. 如果所有 retained-set 方向都无法取得可接受 forgetting，则不建议继续扩大模型，应先回到方法层面检查 Hessian / inverse-HVP / update application 的符号和目标定义。
4. 暂不建议马上进入 Qwen2/Qwen2.5-1.5B 或 data_sample=1.0，除非阶段三完整结果已经明确稳定。

当前最重要的事项：

```text
等待 stage3_full_direction_guard_20260504 跑完。
看到 [DONE] 后立即下载 stage3_full_direction_guard_20260504.tar.gz。
下载后基于真实结果更新阶段三报告、三阶段总结和论文正文。
```

十九、2026-05-04 阶段四实验矩阵与论文收口判断

当前进展更新：

1. 阶段三完整 9 组方向与 guard 消融实验已经完成，并且已经基于 `stage3_full_direction_guard_20260504` 重写阶段三报告、三阶段总结和论文 Word/Markdown。
2. 阶段四的大模型 + 大数据实验已经完成，结果目录为：

```text
formal_cloud_results/stage4_large_model_data_20260504
```

3. 阶段四A和阶段四B正在云服务器上运行：

```text
阶段四A：小模型 + 大数据
模型：Qwen2-0.5B
data_sample = 0.6
rounds = 20
完整 9 组
结果目录：formal_cloud_results/stage4a_qwen05b_dsample06_full9_20260504

阶段四B：大模型 + 小数据
模型：Qwen2.5-1.5B
data_sample = 0.4
rounds = 20
完整 9 组
结果目录：formal_cloud_results/stage4b_qwen15b_dsample04_full9_20260504

阶段四C：大模型 + 大数据
模型：Qwen2.5-1.5B
data_sample = 0.6
rounds = 20
完整 9 组
结果目录：formal_cloud_results/stage4_large_model_data_20260504
状态：已完成
```

阶段四的核心实验设计是 2 x 2 泛化矩阵：

```text
小模型 + 小数据：Qwen2-0.5B，data_sample=0.4，阶段三
小模型 + 大数据：Qwen2-0.5B，data_sample=0.6，阶段四A
大模型 + 小数据：Qwen2.5-1.5B，data_sample=0.4，阶段四B
大模型 + 大数据：Qwen2.5-1.5B，data_sample=0.6，阶段四C
```

这套矩阵的意义：

1. 阶段三负责证明方法机制和完整消融。
2. 阶段四A单独放大数据规模，用来观察数据规模变化是否影响 retained-set Hessian、negative sign 和 guard 的稳定性。
3. 阶段四B单独放大模型规模，用来观察模型规模变化是否影响遗忘方向和 utility preservation。
4. 阶段四C同时放大模型和数据，用来验证方法是否能在更接近大规模设置的场景中保持结论。

论文主线建议：

```text
阶段一证明 FedHDS + Hessian unlearning 的工程链路可运行。
阶段二发现 naive forget-batch Hessian 与 retained-set Hessian 在 utility/forgetting trade-off 上存在明显差异，且 retained-set 默认方向可能降低 forget loss。
阶段三通过完整 9 组方向、guard 和 norm sweep 消融，证明 update sign、direction probing 和 forget-loss guard 是稳定遗忘更新的关键。
阶段四用 2 x 2 模型规模和数据规模矩阵验证该结论的泛化性。
```

是否需要阶段五：

严格来说，这不应该再定义为新的“阶段五主实验”。

原因：

1. 当前阶段一到阶段四已经形成完整研究闭环：基线、问题发现、机制修正、规模泛化。
2. 继续扩展新模型、新数据集、新客户端数量或全新 baseline，容易让论文从“解决一个清楚问题”变成“什么都试了一点”。
3. 对研究生小论文而言，现在的实验完整度已经足够，后续重点应转向讲清楚问题、方法、消融逻辑、结果解释和局限性。

更合理的命名：

```text
不要称为阶段五主实验。
可以称为：论文收口阶段、稳定性补充、seed robustness check 或 final verification。
```

如果云服务器仍有余量，建议只做小规模可信度补充，而不是开辟新研究方向。

最值得补的实验：

```text
多随机种子验证：
配置：Qwen2.5-1.5B + data_sample=0.6
seed：43、44
实验组：核心 3 到 5 组即可
```

优先核心 3 组：

```text
fedhds
forget_batch_hessian
retained_negative_norm010_guard10
```

如果时间充足，可扩展为核心 5 组：

```text
fl
fedhds
forget_batch_hessian
retained_auto_forget_guard_tol0_guard10
retained_negative_norm010_guard10
```

补 seed 的价值：

1. 可以证明最终核心结论不是 seed=42 的偶然结果。
2. 比继续扩大模型或数据更适合论文可信度提升。
3. 不会改变主线，只是加强已有结论。

不建议继续增加的内容：

```text
新数据集
新模型系列
更大模型
更多客户端数
全新 baseline
data_sample=1.0 的完整 9 组大实验
```

这些内容成本高、解释负担重，而且可能让论文主线发散。当前最重要的是把已有结果组织成清楚的故事：

```text
问题：联邦大模型微调中的 Hessian-based unlearning 容易在 forgetting 与 utility 之间失衡。
发现：forget-batch Hessian 往往增强遗忘但损害全局性能，retained-set Hessian 能保护 utility 但方向选择很关键。
方法：加入 retained-set Hessian、direction probing、forget-loss guard 和 global-loss guard。
证据：阶段三完整消融 + 阶段四 2 x 2 规模泛化矩阵。
边界：本工作不声称解决所有数据集和所有模型规模，只证明在当前 FedHDS/Dolly/Qwen 设置下，guarded retained-set Hessian unlearning 能提供更可控的 trade-off。
```

阶段四A/B完成后的收口任务：

1. 下载并解压阶段四A和阶段四B结果。
2. 检查 `summary.csv`、`paper_table.md`、`paper_table.csv`、`figures/` 和各组 `final_results.json`。
3. 把阶段四A/B/C结果合并成 2 x 2 泛化表。
4. 更新 `readme_result3.docx.txt` 或新增阶段四结果说明。
5. 更新三阶段总结，必要时改名为“四阶段实验总结”。
6. 更新 `paper_draft.md` 和 `paper_draft.docx`。
7. 在论文中加入阶段四图表、泛化矩阵表和最终 discussion。
8. 最后再决定是否补 seed=43/44 的核心组稳定性实验。

阶段四A/B完成后的判断标准：

```text
如果 retained_negative_norm010_guard10 或相近 negative norm 配置在 A/B/C 中大多能提高 forget loss，并且 final global loss 不明显劣于 FedHDS，则阶段四结论成立。
如果某些放大配置中 forget loss 没有提升，但 global loss 保持更好，也可以写成 trade-off 和规模敏感性。
如果 forget-batch Hessian 在大模型或大数据下继续损害 global loss，则可以作为 retained-set guard 必要性的反证。
如果所有 retained-set negative 配置在某个规模上失败，则不需要硬说成功，应写成局限性：retained-set Hessian 的有效方向和更新强度对模型/数据规模敏感，需要 guard 和 probing 控制。
```

当前决策：

```text
阶段四A/B完成后，论文主体可以进入完整写作和定稿。
不再主动扩展新的大方向。
云服务器余量只用于补 seed 或复核关键配置，不作为新的阶段五主实验。
```

二十、阶段四 seed robustness 完成后的收口清单

当前补充实验定位：

```text
名称：Stage 4 seed robustness check
性质：稳定性补充，不是阶段五主实验
模型：Qwen2.5-1.5B
data_sample = 0.6
rounds = 20
seed = 43, 44
核心组：fl, fedhds, forget_batch_hessian, retained_auto_forget_guard_tol0_guard10, retained_negative_norm010_guard10
目标：验证阶段四C的大模型大数据核心结论不是 seed=42 的偶然结果。
```

seed robustness 跑完后，原则上不再新增实验方向。

不再主动补充：

```text
新数据集
新模型系列
更大模型
更多客户端数
全新 baseline
data_sample=1.0 的完整 9 组大实验
新的阶段五主实验
```

后续工作重点转为结果整合和论文定稿。

一、下载并检查 seed robustness 结果

完成后下载：

```text
stage4_seed_robustness_qwen15b_dsample06_core5_20260504.tar.gz
```

解压后结果目录应为：

```text
formal_cloud_results/stage4_seed_robustness_qwen15b_dsample06_core5_20260504
```

需要检查：

```text
summary.csv
paper_table.md
paper_table.csv
figures/
seed43/summary.csv
seed44/summary.csv
seed43/paper_table.md
seed44/paper_table.md
各实验组 final_results.json
```

判断标准：

```text
总 summary.csv 应有 10 行：2 个 seed x 5 个核心组。
每个 seed 的 summary.csv 应有 5 行。
每个 seed 的 figures/ 应有图表输出。
总 figures/ 应有汇总图表输出。
文本文件不应包含乱码替换字符。
```

二、整理阶段四完整结果

最终实验矩阵：

```text
阶段三：小模型 + 小数据
Qwen2-0.5B，data_sample=0.4，完整 9 组

阶段四A：小模型 + 大数据
Qwen2-0.5B，data_sample=0.6，完整 9 组

阶段四B：大模型 + 小数据
Qwen2.5-1.5B，data_sample=0.4，完整 9 组

阶段四C：大模型 + 大数据
Qwen2.5-1.5B，data_sample=0.6，完整 9 组

稳定性补充：
Qwen2.5-1.5B，data_sample=0.6，seed=43/44，核心 5 组
```

需要从每组提取的核心字段：

```text
final_global_loss
forget_client_loss_before_unlearning
forget_client_loss_after_unlearning
forget loss gain
global_loss_after_unlearning
actual_param_delta_l2_norm
unlearn_steps_accepted
unlearn_steps_attempted
unlearning_time_sec
unlearn_selected_update_sign
unlearn_guard_stop_reason
```

三、制作 2 x 2 泛化矩阵表

矩阵结构：

```text
                 data_sample=0.4          data_sample=0.6
Qwen2-0.5B       阶段三                    阶段四A
Qwen2.5-1.5B     阶段四B                   阶段四C
```

每个格子至少放三类结果：

```text
FedHDS baseline final global loss
forget_batch_hessian forget loss before -> after, final global loss
retained_negative_norm010_guard10 forget loss before -> after, final global loss
```

如果空间允许，可额外加入：

```text
retained_auto_forget_guard_tol0_guard10
最佳 retained negative norm 配置
forget loss gain
global loss gap relative to FedHDS
```

四、写阶段四结果分析

阶段四分析要回答四个问题：

1. 数据规模从 0.4 放大到 0.6 后，retained-set negative + guard 是否仍能提高 forget loss？
2. 模型规模从 Qwen2-0.5B 放大到 Qwen2.5-1.5B 后，遗忘方向是否仍以 negative sign 更稳定？
3. forget-batch Hessian 是否仍存在 utility 损伤或遗忘不稳定问题？
4. retained-set Hessian + guard 是否提供了更可控的 forgetting / utility trade-off？

写作时不要简单宣称“全面优于”，而应写成：

```text
retained-set Hessian with negative-direction update and guard constraints provides a more controllable trade-off between forgetting strength and global utility than naive forget-batch Hessian.
```

中文解释：

```text
retained-set Hessian 加上 negative direction 和 guard 约束，相比 naive forget-batch Hessian，不一定总是取得最大 forget loss，但更容易在提高遗忘强度的同时控制 global utility 损失。
```

五、整理 seed robustness 结果

seed robustness 的表述重点：

```text
在大模型大数据设置下，补充 seed=43/44 的核心组复现实验。
如果 retained_negative_norm010_guard10 在多数 seed 中保持 forget loss 不下降或上升，并且 final global loss 接近 FedHDS，则说明阶段四C结论不是单一 seed 偶然现象。
```

如果 seed 结果不完全一致，也不要强行写成全部稳定。可写成：

```text
The robustness check shows that the retained-set negative update is generally less destructive to global utility than forget-batch Hessian, while the exact forgetting gain remains seed-sensitive.
```

中文解释：

```text
稳定性补充说明 retained-set negative update 通常比 forget-batch Hessian 更少破坏全局性能，但具体遗忘增益仍存在随机种子敏感性。
```

六、更新论文和报告文件

需要更新或生成：

```text
paper_draft.md
paper_draft.docx
readme_result3.docx.txt
readme_result_3.docx.docx
Summary_of_the_three_stages_of_the_thesis.txt
Summary_of_the_three_stages_of_the_thesis.docx
thesis_outline.md
```

建议把三阶段总结改为更准确的名称或内容：

```text
四阶段实验总结
阶段一到阶段四及稳定性补充总结
FedHDS Hessian Unlearning 实验总结
```

如果不改文件名，也必须在正文中说明已经包含阶段四和 seed robustness。

七、整理最终图表

论文中建议保留的核心表：

```text
阶段三完整 9 组消融表
阶段四 2 x 2 泛化矩阵表
阶段四A/B/C 关键结果表
seed robustness 表
```

论文中建议保留的核心图：

```text
forget loss before/after
final/global loss
update norm vs global loss
update norm vs forget gain
guard steps
unlearning time
```

图表原则：

```text
不要把所有图都堆进正文。
正文保留最能支撑主线的图表。
其余图可放附录或 Word 附图部分。
所有表格数值必须与 summary.csv / paper_table.md 对齐。
```

八、写局限性和未来工作

必须承认的边界：

```text
单数据集：Databricks Dolly 15k
单模型家族：Qwen/Qwen2.5
随机种子数量有限
forget client 固定为 client 0
unlearning 评估主要基于 loss，而不是更复杂的 membership inference、privacy attack 或生成质量评估
未覆盖 7B 级模型和更多真实联邦数据分布
```

推荐写法：

```text
本工作重点不是证明方法在所有模型和数据集上普遍最优，而是在 FedHDS 联邦指令微调场景中，系统分析 Hessian-based unlearning 的方向、reference set 和 guard 机制，证明 retained-set guarded update 可以提供更可控的 utility-preserving unlearning trade-off。
```

九、最终收口判断

当以下事项全部完成后，可以进入论文最终定稿：

```text
阶段四A结果已下载并检查。
阶段四B结果已下载并检查。
阶段四C结果已下载并检查。
seed robustness 结果已下载并检查。
2 x 2 泛化矩阵已生成。
seed robustness 表已生成。
paper_draft.md 已更新。
paper_draft.docx 已更新。
阶段总结文档已更新。
所有核心数值与 summary.csv 对齐。
正文结论没有夸大为“所有模型/所有数据集普适”。
局限性和未来工作已写清楚。
```

最终方向：

```text
停止扩展新实验。
集中完成结果整合、图表核对、论文叙事和定稿。
```
