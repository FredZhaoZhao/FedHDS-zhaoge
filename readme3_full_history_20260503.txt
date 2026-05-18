【本轮进度 1】
  已完成第一步最小改造：只修改 client.py，新增 retained reference 采样接口 `sample_reference_examples(sample_size,
  shuffle=False)`。

  本轮改动内容：
  1. 从 `self.full_train_dataset` 中采样参考样本，而不是从当前训练 loader 取数据。
  2. 支持两种最小采样方式：
     - 顺序取前 N 条
     - 随机取 N 条
  3. 返回结果为 `list[example]`，供后续 server 端拼接 `D_ref` 使用。

  本轮未改动：
  1. 未修改 server.py 的 Hessian / LiSSA 逻辑。
  2. 未修改 main.py 参数。
  3. 未改训练主流程、FedHDS 聚类逻辑、数据规模和实验超参。

  本轮验证：
  1. `python -m py_compile client.py` 通过。
  2. 当前改动属于独立接口补充，风险较低。

【本轮进度 2】
  已完成第二步最小改造：只修改 server.py，把 retained reference set 的核心逻辑接入 unlearning，但暂时未修改 main.py 参数
  入口。

  本轮改动内容：
  1. 新增 `build_retained_reference_loader(forget_client_idx, client_list)`：
     - 遍历所有非 forget client
     - 从每个客户端取少量 reference examples
     - 拼接成 retained reference set `D_ref`
     - 用 retained client 的 `collate_fn` 构造 DataLoader

  2. 新增 `_compute_average_loss(data_loader)`：
     - 对 retained reference loader 遍历求平均 loss
     - 返回一个标量 loss，供 Hessian-vector product / LiSSA 使用

  3. 修改 `apply_fedhds_unlearning(...)`：
     - 保留旧 baseline 路径：forget-batch Hessian
     - 新增 retained-Hessian 路径：如果 `use_retained_hessian=True`，则优先使用 `D_ref` 的平均 loss 作为 Hessian 参考目
  标
     - 如果 retained reference 不可用，则自动回退到旧 baseline 路径

  4. 新增日志输出：
     - forget gradient sample size
     - retained reference set size
     - LiSSA depth / damping

  本轮未改动：
  1. 未修改 main.py 参数区。
  2. 未改变训练主流程。
  3. 未改变 FedHDS 聚类/筛样本逻辑。
  4. 未改数据规模和实验超参。

  本轮验证：
  1. `python -m py_compile server.py` 通过。
  2. 当前 retained-Hessian 分支已经在 server 端实现，但命令行参数入口还未开放，默认仍走旧路径。


【本轮进度 3】
  已完成第三步最小改造：只修改 main.py，补齐 retained-Hessian 的最小参数入口。

  本轮改动内容：
  1. 新增参数 `--use_retained_hessian`
     - 作为 retained-set Hessian 路径开关
     - 默认关闭，保持旧 baseline 路径不变

  2. 新增参数 `--hessian_ref_per_client`
     - 表示从每个非 forget client 抽取多少条 reference samples 进入 `D_ref`
     - 默认值设为 2

  3. 新增一个最小合法性检查
     - 如果 `hessian_ref_per_client < 1`，直接报错
     - 避免无效配置进入 unlearning 过程

  本轮未改动：
  1. 未改变训练主流程
  2. 未改变 FedHDS 聚类/筛样本逻辑
  3. 未改变数据规模和正式实验超参
  4. 未增加额外机制

  本轮验证：
  1. `python -m py_compile main.py client.py server.py` 通过
  2. 尝试执行 `python main.py --help` 时失败，原因是当前本地环境缺少 `torch`，不是本轮参数修改本身出错

  当前整体状态：
  1. client.py 已有 retained reference 采样接口
  2. server.py 已有 retained reference loader、average loss 和 retained-Hessian 分支
  3. main.py 已补齐最小参数入口


【本轮进度 4】
  已完成第四轮小改造：不增加新机制，只补运行保护和模式日志，方便本地 smoke test。

  本轮改动内容：
  1. main.py 增加最小参数合法性检查：
     - `--use_retained_hessian` 必须和 `--use_fedhds_unlearn` 同时使用
     - 开启 unlearning 时，`forget_client_idx` 必须落在 `[0, num_clients)` 范围内
     - `hessian_ref_per_client` 必须 >= 1
     - `unlearn_grad_sample_size` 必须 >= 0

  2. server.py 增加更清晰的运行日志：
     - unlearning 开始时打印当前 Hessian reference mode
     - retained-Hessian 路径下打印 `Retained samples per client`
     - retained reference 不可用时，明确打印回退到 forget-batch Hessian
     - fallback 到旧路径时，明确打印当前使用 forget client batch 作为 Hessian reference

  本轮未改动：
  1. 未新增方法机制
  2. 未改训练主流程
  3. 未改数据规模和实验超参
  4. 未改 FedHDS 聚类/筛样本逻辑

  本轮验证：
  1. `python -m py_compile main.py server.py client.py` 通过

  当前整体状态：
  1. 最小方法改造已经完成
  2. baseline 和 retained-Hessian 两条路径都已具备代码入口
  3. 运行时日志已经足够支持本地 smoke test 排查


【本轮进度 5：本地 smoke test】
  已在 WSL 的 HG 环境中完成本地最小 smoke test，验证 baseline 和 retained-Hessian 两条路径都能端到端运行。

  测试环境：
  1. Python 环境：`/mnt/f/ANACONDA/envs/HG/bin/python`
  2. 数据集：dolly
  3. 模型：Qwen/Qwen2-0.5B
  4. 测试配置：4 clients，1 round，每客户端 1 batch，data_sample=0.05

  测试 1：baseline 路径
  1. 命令未开启 `--use_retained_hessian`
  2. 日志显示 Hessian reference mode 为 `forget-batch (baseline)`
  3. `Forget gradient sample size: 4`
  4. unlearning 正常完成
  5. post-unlearning global loss 为 0.7462

  测试 2：retained-Hessian 路径
  1. 命令开启 `--use_retained_hessian --hessian_ref_per_client 2`
  2. 日志显示 Hessian reference mode 为 `retained-set`
  3. `Retained samples per client: 2`
  4. `Retained reference set size: 6`
  5. unlearning 正常完成
  6. post-unlearning global loss 为 0.7462

  当前结论：
  1. 最小方法改造已经完成
  2. 旧 baseline 路径未被破坏
  3. retained-Hessian 新路径已经成功进入并执行完成
  4. 本地最小闭环验证通过

  说明：
  1. 本轮属于 smoke test，只验证“代码可运行、分支正确进入、流程能完成”
  2. 当前结果不能直接作为方法优劣结论
  3. 下一步应进行更正式的小规模本地实验，再考虑上云跑更大规模数据

  附注：
  运行过程中出现若干非阻塞 warning（如 torch_dtype deprecated、use_cache 与 gradient checkpointing 提示），但未影响本次
  测试完成。

  现在可以进入下一阶段了：先做一组本地小规模正式对比，再上云跑大规模。



确认：现在这一步就是按 README 里的最小方案，先做一组本地小规模正式对比实验。我已经继续推进并跑完了四组。

  改动位置：

  - server.py:156：修了二阶 HVP 的 attention backend，强制关闭 flash/mem-efficient SDP，避免 retained-set Hessian 报二阶
    导不支持。
  - server.py:312：补了 unlearning 过程指标记录，并在 unlearning 前关闭 gradient checkpointing，解决梯度全 0 问题。
  - main.py:235：补了 forget-client loss、global loss、unlearning time 的正式记录。

  本地正式小实验结果，配置为 Qwen/Qwen2-0.5B、dolly、4 clients、2 rounds、data_sample=0.05、eta=0.01：

  | 组别 | 训练后 Global Loss | Forget Loss | Unlearn 后 Global Loss | Time |
  |---|---:|---:|---:|---:|
  | FL | 0.2983 | - | - | - |
  | FedHDS | 0.3310 | - | - | - |
  | FedHDS + forget-batch Hessian | 0.3310 | 2.4274 -> 3.1713 | 1.1036 | 2.78s |
  | FedHDS + retained-set Hessian | 0.3310 | 2.4274 -> 9.7236 | 9.9319 | 21.52s |

  结论很直接：这组本地正式对比已经完成，代码路径也跑通了；但 eta=0.01 对 retained-set Hessian 太大，更新范数更高，导致全
  局性能崩掉。当前结果可以作为“本地验证和问题定位”记录，但不适合作为小论文主表结果。下一步应做 retained-set 的小步长
  sweep，比如 eta=0.001 / 0.003 / 0.005，找一个能提高 forget loss 但不把 global loss 打爆的设置。

  验证已做：

  - python -m py_compile main.py server.py client.py
  - 四组结果已保存到 formal_local_compare_20260429/final_eta001/ 下各自的 final_results.json。


【本轮进度 6：retained-set Hessian 稳定性改造、汇总脚本与 eta sweep】
  本轮目标：
  1. 不扩展方法主线，不重构训练流程。
  2. 只给 retained-set Hessian unlearning 增加更新范数控制。
  3. 增加自动结果汇总脚本，方便后续看实验表。
  4. 跑 retained-set 小步长 sweep，判断 eta 和 clipping 对 global loss 崩坏的影响。

  本轮代码改动：
  1. main.py：
     - 新增参数 `--unlearn_max_update_norm`
     - 类型为 float，默认 0.0
     - 约定：0 或负数表示不启用 update clipping

  2. server.py：
     - 在 `apply_fedhds_unlearning(...)` 中，在真正 `param.data.add_` 前增加 global L2 norm clipping
     - clipping 作用对象是 `inverse_hvp * scale` 形成的整体 unlearning update
     - 如果 `unlearn_max_update_norm > 0` 且 scaled update L2 norm 超过该阈值，就把 scale 乘以
       `max_update_norm / scaled_update_l2_norm`
     - 保留原有 `actual_param_delta_l2_norm` 等指标
     - 新增记录到 `experiment_metrics` 的字段包括：
       - `unlearn_eta`
       - `unlearn_max_update_norm`
       - `update_clipping_enabled`
       - `update_clip_applied`
       - `update_clip_coefficient`
       - `scaled_update_l2_norm_before_clipping`
       - `scaled_update_l2_norm_after_clipping`
       - `scaled_update_max_abs_before_clipping`
       - `scaled_update_max_abs_after_clipping`

  3. scripts/summarize_experiments.py：
     - 新增轻量汇总脚本
     - 递归扫描指定实验根目录下所有 `final_results.json`
     - 输出 `summary.csv` 到实验根目录
     - 主要字段包括：
       - group / path
       - round2_global_loss / final_global_loss
       - global_loss_before_unlearning / global_loss_after_unlearning
       - forget_client_loss_before_unlearning / forget_client_loss_after_unlearning
       - unlearning_time_sec
       - hessian_mode
       - retained_reference_set_size
       - last_hvp_l2_norm
       - actual_param_delta_l2_norm
       - unlearn_eta
       - unlearn_max_update_norm
       - clipping 相关字段

  本轮验证：
  1. 语法验证通过：
     `python -m py_compile main.py server.py client.py scripts/summarize_experiments.py`
  2. 已对旧结果生成汇总：
     `formal_local_compare_20260429/final_eta001/summary.csv`
  3. 已对本轮 sweep 结果生成汇总：
     `formal_local_compare_20260429/retained_eta_sweep/summary.csv`

  retained-set 小步长 sweep 配置：
  - dataset: dolly
  - model: Qwen/Qwen2-0.5B
  - num_clients: 4
  - k: 1.0
  - rounds: 2
  - batch_or_epoch: batch
  - local_step: 2
  - batch_size: 1
  - max_length: 64
  - data_sample: 0.05
  - iid: 0
  - filtering: True
  - compound_dim: 8
  - n_cluster: 5
  - filtering_sample_limit: 20
  - use_fedhds_unlearn: True
  - use_retained_hessian: True
  - hessian_ref_per_client: 4
  - forget_client_idx: 0
  - lissa_depth: 1
  - lissa_damping: 0.01
  - unlearn_grad_sample_size: 8

  本轮 retained-set sweep 结果：

  | 设置 | 训练后 Global Loss | Forget Loss | Unlearn 后 Global Loss | Update L2 | Clipping |
  |---|---:|---:|---:|---:|---|
  | eta=0.001 | 0.3310 | 2.4274 -> 2.7134 | 0.9103 | 0.2928 | 否 |
  | eta=0.003 | 0.3310 | 2.4274 -> 3.3970 | 1.5150 | 0.8783 | 否 |
  | eta=0.005 | 0.3310 | 2.4274 -> 4.7942 | 2.1302 | 1.4639 | 否 |
  | eta=0.01, max_update_norm=1.0 | 0.3310 | 2.4274 -> 3.6385 | 1.5112 | 1.0000 | 是，coef=0.3416 |

  本轮结论：
  1. retained-set Hessian 的方向确实能提高 forget client loss。
  2. 但 update norm 增大时，global loss 仍会明显变差。
  3. `eta=0.01` 未裁剪时 retained-set global loss 曾崩到 9.9319；加入 `max_update_norm=1.0` 后降到 1.5112，说明 clipping 生效。
  4. 当前还没有找到“forget loss 明显升高，同时 global loss 接近 0.3310”的理想设置。
  5. 下一步更值得继续扫：
     - `max_update_norm=0.3 / 0.5 / 0.8`
     - 或更小 eta：`0.0003 / 0.0005`

  维护约定：
  以后所有代码改动、实验配置、结果目录、关键数值和阶段性结论，都继续追加到 readme3.txt，避免后续理解断层。


【本轮进度 7：继续 retained-set 稳定性实验，clip sweep 与 tiny eta sweep】
  本轮目标：
  1. 不再改方法代码。
  2. 继续围绕 retained-set Hessian 的更新幅度做稳定性实验。
  3. 重点判断两件事：
     - 固定 eta=0.01 时，global L2 clipping 能不能形成可控折中。
     - 进一步降低 eta 后，global loss 能不能接近训练后原始值 0.3310。

  本轮新增实验目录：
  1. `formal_local_compare_20260429/retained_clip_sweep/`
     - 固定 `unlearn_eta=0.01`
     - 扫 `unlearn_max_update_norm=0.3 / 0.5 / 0.8`
     - 已生成 `summary.csv`

  2. `formal_local_compare_20260429/retained_tiny_eta_sweep/`
     - 不启用 clipping
     - 扫 `unlearn_eta=0.0003 / 0.0005`
     - 已生成 `summary.csv`

  本轮验证：
  1. 语法验证仍通过：
     `python -m py_compile main.py server.py client.py scripts/summarize_experiments.py`
  2. 汇总脚本正常输出：
     - `formal_local_compare_20260429/retained_clip_sweep/summary.csv`
     - `formal_local_compare_20260429/retained_tiny_eta_sweep/summary.csv`

  固定 eta=0.01 的 clipping sweep 结果：

  | 设置 | 训练后 Global Loss | Forget Loss | Unlearn 后 Global Loss | Update L2 | Clip Coef |
  |---|---:|---:|---:|---:|---:|
  | max_update_norm=0.3 | 0.3310 | 2.4274 -> 2.7217 | 0.9171 | 0.3000 | 0.1025 |
  | max_update_norm=0.5 | 0.3310 | 2.4274 -> 2.9165 | 1.1605 | 0.5000 | 0.1708 |
  | max_update_norm=0.8 | 0.3310 | 2.4274 -> 3.2662 | 1.4980 | 0.8000 | 0.2732 |

  tiny eta sweep 结果：

  | 设置 | 训练后 Global Loss | Forget Loss | Unlearn 后 Global Loss | Update L2 | Clipping |
  |---|---:|---:|---:|---:|---|
  | eta=0.0003 | 0.3310 | 2.4274 -> 2.4626 | 0.5646 | 0.0878 | 否 |
  | eta=0.0005 | 0.3310 | 2.4274 -> 2.5339 | 0.7083 | 0.1464 | 否 |

  本轮观察：
  1. clipping 的行为和预期一致：`actual_param_delta_l2_norm` 基本等于设置的 `max_update_norm`。
  2. retained-set Hessian 的主要敏感因素仍然是 update L2 norm，而不是 eta 或 clipping 本身。
  3. update L2 越大，forget loss 越高，但 global loss 也越差。
  4. update L2 约 0.09 到 0.15 时，global loss 已经降到 0.56 到 0.71，但 forget loss 提升很弱。
  5. update L2 约 0.30 时，forget loss 有一定提升到 2.72，但 global loss 已经到 0.91。
  6. 这说明当前 retained-set Hessian 单步更新存在明显 trade-off：要明显遗忘就会付出较大全局效用代价。

  当前阶段结论：
  1. 代码链路、指标记录、clipping 和结果汇总都已经稳定。
  2. retained-set Hessian 作为“方向”是有效的，但单步强更新不够稳。
  3. 如果要写小论文，当前结果更适合表述为：
     - 提出了 retained-set curvature approximation；
     - 工程链路可运行；
     - clipping 能显著避免 eta=0.01 的灾难性崩溃；
     - 但在当前 dolly/Qwen2-0.5B 小规模设置下，遗忘强度和全局效用之间存在明显 trade-off。

  下一步建议：
  1. 如果继续找本地可用结果，优先扫更细的 update norm 区间：
     - `max_update_norm=0.15 / 0.2 / 0.25`
     - 这些点介于 tiny eta 和 max_norm=0.3 之间，可能找到 global loss < 0.8 且 forget loss 有轻微提升的折中点。
  2. 如果想提高遗忘效果但减少 global loss 损伤，下一轮可能需要方法层的小改造，而不是继续盲扫 eta：
     - 例如多步小更新，每步后重新计算 reference loss；
     - 或对 inverse HVP 做更强 damping；
     - 或只更新 LoRA 的部分层/限制 per-layer norm。
  3. 暂时不要宣称 retained-set Hessian 已经优于 baseline，只能说当前发现了可控但有代价的稳定区间。


【本轮进度 8：retained-set fine clipping sweep，max_update_norm=0.15/0.2/0.25】
  本轮目标：
  1. 不再改方法代码。
  2. 固定 retained-set Hessian 的 `unlearn_eta=0.01`。
  3. 只继续细扫 global update L2 clipping 阈值：
     - `unlearn_max_update_norm=0.15`
     - `unlearn_max_update_norm=0.2`
     - `unlearn_max_update_norm=0.25`
  4. 判断在 `max_update_norm=0.3` 之前是否存在 global loss 更低、forget loss 又有可见提升的折中点。

  本轮新增实验目录：
  - `formal_local_compare_20260429/retained_fine_clip_sweep/`
  - 已生成汇总文件：
    `formal_local_compare_20260429/retained_fine_clip_sweep/summary.csv`

  本轮验证：
  1. 汇总脚本正常输出：
     `python scripts/summarize_experiments.py formal_local_compare_20260429\retained_fine_clip_sweep`
  2. 语法验证通过：
     `python -m py_compile main.py server.py client.py scripts/summarize_experiments.py`

  固定 eta=0.01 的 fine clipping sweep 结果：

  | 设置 | 训练后 Global Loss | Forget Loss | Unlearn 后 Global Loss | Update L2 | Clip Coef | Unlearn Time |
  |---|---:|---:|---:|---:|---:|---:|
  | max_update_norm=0.15 | 0.3310 | 2.4274 -> 2.5388 | 0.7162 | 0.1500 | 0.0512 | 24.89s |
  | max_update_norm=0.2 | 0.3310 | 2.4274 -> 2.6036 | 0.8090 | 0.2000 | 0.0683 | 26.42s |
  | max_update_norm=0.25 | 0.3310 | 2.4274 -> 2.6652 | 0.8731 | 0.2500 | 0.0854 | 20.19s |

  本轮观察：
  1. 三组实验都触发了 clipping，且 `actual_param_delta_l2_norm` 基本等于设定的 `max_update_norm`。
  2. `max_update_norm=0.15` 是这三组里 global loss 最稳的点：global loss 为 0.7162，forget loss 从 2.4274 提到 2.5388。
  3. `max_update_norm=0.2` 和 `0.25` 带来更高 forget loss，但 global loss 分别升到 0.8090 和 0.8731，已经超过 0.8 附近。
  4. 和 tiny eta sweep 对比，`eta=0.0005` 的 update L2 为 0.1464，结果是 global loss 0.7083、forget loss 2.5339；本轮 `max_update_norm=0.15` 的表现非常接近。这进一步说明当前主要控制变量是实际 update L2 norm，而不是 eta 或 clipping 形式本身。
  5. 目前 retained-set 单步 Hessian 更新的折中区间大约在 update L2 0.15 到 0.25：可以得到轻到中等的 forget loss 提升，但还不能让 global loss 接近训练后原始值 0.3310。

  当前阶段结论：
  1. 继续单纯扫 `max_update_norm` 已经能描出稳定性曲线，但很难同时得到强遗忘和低 global loss。
  2. 如果小论文要追求更好的方法结果，下一步应进入方法层小改造，而不是继续只做参数/数据层 sweep。
  3. 候选最小方法改造方向：
     - 多步小更新，并在每步后监控 global/reference loss，超过阈值就停止；
     - 在 retained-set reference loss 上做 line search 或 trust-region 接受准则；
     - 增大 LiSSA damping 或对 inverse HVP 做额外 shrink；
     - 限制 per-layer update norm，避免少数层承担过大更新。


【本轮进度 9：方法层最小改造，多步小更新 + reference/global loss guard】
  本轮目标：
  1. 不继续只扫 eta 或 max_update_norm。
  2. 在现有 retained-set Hessian unlearning 链路上做最小方法层改造。
  3. 默认行为保持旧逻辑：`unlearn_num_steps=1` 且 guard 参数为 0 时，仍等价于单步更新。
  4. 开启新参数后，将一次 unlearning update 切成多步小更新，并允许用 retained/reference loss 或 global eval loss 作为接受准则，失败则回滚当前小步并停止。

  本轮代码改动：
  1. main.py 新增 CLI 参数：
     - `--unlearn_num_steps`，int，默认 1。
       - 表示把最终 clipped update 均分成多少个小步。
     - `--unlearn_ref_loss_guard_ratio`，float，默认 0.0。
       - 0 表示关闭。
       - 大于 0 时，使用 reference loss guard。
       - 判断规则：小步后 reference loss 不能超过小步前 baseline 的该倍数。
     - `--unlearn_global_loss_guard_max`，float，默认 0.0。
       - 0 表示关闭。
       - 大于 0 时，使用 global eval loss guard。
       - 判断规则：小步后 global eval loss 不能超过该绝对阈值。
     - 新增合法性检查：
       - `unlearn_num_steps >= 1`
       - guard 参数必须非负。

  2. server.py 改造 `apply_fedhds_unlearning(...)`：
     - 保留原有 forget gradient、retained-set Hessian、LiSSA HVP、global L2 clipping 逻辑。
     - 原来的总 update 仍先由 `inverse_hvp * scale` 形成，并按 `unlearn_max_update_norm` 做整体 clipping。
     - 新增 step loop：
       - `step_scale = clipped_scale / unlearn_num_steps`
       - 每步应用 `inverse_hvp * step_scale`
       - 每步应用前保存参数快照
       - guard 通过则接受该步
       - guard 失败则恢复该步前参数，记录停止原因并退出
     - 新增 no-grad loss 计算辅助函数：
       - `_compute_batch_loss_value(...)`
       - `_compute_average_loss_value(...)`
       - `_compute_guard_reference_loss_value(...)`
     - 这些辅助函数只用于 guard，不影响 HVP 所需的二阶计算图。

  3. server.py 新增记录到 `experiment_metrics` 的字段：
     - `unlearn_num_steps`
     - `unlearn_ref_loss_guard_ratio`
     - `unlearn_global_loss_guard_max`
     - `unlearn_ref_loss_guard_enabled`
     - `unlearn_global_loss_guard_enabled`
     - `unlearn_reference_loss_before_guard`
     - `unlearn_reference_loss_after_last_accepted`
     - `unlearn_global_loss_after_last_accepted`
     - `unlearn_steps_attempted`
     - `unlearn_steps_accepted`
     - `unlearn_guard_stop_reason`
     - `unlearn_guard_step_records`
     - `unlearn_requested_scale_after_clipping`
     - `unlearn_step_scale`
     - `unlearn_step_update_l2_norm`
     - `unlearn_applied_scale`
     - `unlearn_applied_update_l2_norm_estimate`

  4. scripts/summarize_experiments.py 新增 CSV 字段：
     - `unlearn_num_steps`
     - `unlearn_steps_attempted`
     - `unlearn_steps_accepted`
     - `unlearn_guard_stop_reason`
     - `unlearn_ref_loss_guard_ratio`
     - `unlearn_global_loss_guard_max`
     - `unlearn_reference_loss_before_guard`
     - `unlearn_reference_loss_after_last_accepted`
     - `unlearn_global_loss_after_last_accepted`
     - `unlearn_applied_update_l2_norm_estimate`

  本轮验证：
  - 语法验证通过：
    `python -m py_compile main.py server.py client.py scripts/summarize_experiments.py`

  本轮尚未运行新实验。
  建议下一组最小验证实验：
  1. retained-set Hessian，固定：
     - `unlearn_eta=0.01`
     - `unlearn_max_update_norm=0.3`
     - `unlearn_num_steps=3`
     - `unlearn_global_loss_guard_max=0.8`
  2. 预期行为：
     - 如果第 1 或第 2 小步后 global loss 已超过 0.8，则回滚该步并停止。
     - 最终 `actual_param_delta_l2_norm` 应小于或等于 0.3。
     - `unlearn_steps_accepted` 会显示实际接受了几个小步。

  当前阶段判断：
  这次改造不是单纯数据/参数 sweep，而是方法层最小改造。它把 retained-set Hessian 的单步强更新变成可截停的小步 trust-region 更新，为后续寻找“更强遗忘但不让 global loss 崩掉”的设置提供了机制。


【本轮进度 10：第一组 stepped guard 验证实验】
  本轮目标：
  1. 验证“多步小更新 + global loss guard”是否真的能截停并回滚超阈值小步。
  2. 不再改代码，只跑第一组最小验证实验。

  实验目录：
  - `formal_local_compare_20260429/retained_step_guard_validation/max03_steps3_global08/`
  - 汇总文件：
    `formal_local_compare_20260429/retained_step_guard_validation/summary.csv`

  实验配置：
  - retained-set Hessian
  - `unlearn_eta=0.01`
  - `unlearn_max_update_norm=0.3`
  - `unlearn_num_steps=3`
  - `unlearn_global_loss_guard_max=0.8`
  - `unlearn_ref_loss_guard_ratio=0.0`，即本次不启用 reference loss guard
  - 其余训练配置沿用前面 dolly/Qwen2-0.5B 小规模正式实验设置。

  本轮验证：
  - 语法验证仍通过：
    `python -m py_compile main.py server.py client.py scripts/summarize_experiments.py`
  - 汇总脚本正常输出：
    `python scripts/summarize_experiments.py formal_local_compare_20260429\retained_step_guard_validation`

  关键结果：

  | 设置 | 训练后 Global Loss | Forget Loss | 最终 Global Loss | 实际 Update L2 | Step Attempted | Step Accepted | Stop Reason |
  |---|---:|---:|---:|---:|---:|---:|---|
  | max_norm=0.3, steps=3, global_guard=0.8 | 0.3310 | 2.4274 -> 2.4766 | 0.5955 | 0.1000 | 2 | 1 | global_loss_guard |

  guard step 细节：

  | Step | 临时/接受后 Global Loss | 阈值 | 结果 |
  |---:|---:|---:|---|
  | 1 | 0.5955 | 0.8 | accepted |
  | 2 | 0.8087 | 0.8 | rejected and rollback |

  本轮观察：
  1. guard 机制按预期生效：第 2 小步超过 global loss 阈值后被拒绝，并恢复到第 1 小步后的参数。
  2. 最终 `actual_param_delta_l2_norm=0.1000`，等于每步 update L2，而不是原计划的总 update L2 0.3000。
  3. 最终 global loss 0.5955，比之前 `max_update_norm=0.15` 的 0.7162 更稳，但 forget loss 只升到 2.4766，遗忘强度较弱。
  4. 这说明 stepped guard 不是直接提升遗忘效果，而是提供了“自动保守截停”的机制，能防止超过设定 global loss 上限。

  当前阶段结论：
  1. 多步小更新 + global loss guard 的工程实现是有效的。
  2. 在 `max_norm=0.3, steps=3, global_guard=0.8` 下，guard 实际只接受了 1/3 更新，因此最终更接近 tiny eta 结果。
  3. 下一步如果要尝试更强遗忘且仍守住 global loss，可以继续调：
     - 放宽 global guard 到 `0.9`
     - 或增加步数到 `5`，让每步更小
     - 或在同样 global guard 下增大总 `max_update_norm`，看是否能接受更多细粒度小步
  4. 本轮仍不能宣称 retained-set Hessian 已优于 baseline，只能说方法层 guard 已经能把灾难性更新变成可控的截停过程。


【本轮进度 11：stepped guard 小对照，steps=5 与 global_guard=0.9】
  本轮目标：
  1. 在第一组 stepped guard 验证成功后，继续做两个最小对照。
  2. 分别观察：
     - 更细步长是否能在同样 global guard 下接受更多有效更新。
     - 放宽 global guard 是否能换取更强 forget loss。

  实验目录：
  - `formal_local_compare_20260429/retained_step_guard_sweep/`
  - 汇总文件：
    `formal_local_compare_20260429/retained_step_guard_sweep/summary.csv`

  实验配置共同点：
  - retained-set Hessian
  - `unlearn_eta=0.01`
  - `unlearn_max_update_norm=0.3`
  - `unlearn_ref_loss_guard_ratio=0.0`
  - 其余训练配置沿用前面 dolly/Qwen2-0.5B 小规模正式实验设置。

  本轮两组对照：
  1. `max03_steps5_global08`
     - `unlearn_num_steps=5`
     - `unlearn_global_loss_guard_max=0.8`
  2. `max03_steps3_global09`
     - `unlearn_num_steps=3`
     - `unlearn_global_loss_guard_max=0.9`

  本轮验证：
  - 语法验证仍通过：
    `python -m py_compile main.py server.py client.py scripts/summarize_experiments.py`
  - 汇总脚本正常输出：
    `python scripts/summarize_experiments.py formal_local_compare_20260429\retained_step_guard_sweep`

  汇总结果：

  | 设置 | 训练后 Global Loss | Forget Loss | 最终 Global Loss | 实际 Update L2 | Step Attempted | Step Accepted | Stop Reason |
  |---|---:|---:|---:|---:|---:|---:|---|
  | max_norm=0.3, steps=5, global_guard=0.8 | 0.3310 | 2.4274 -> 2.5783 | 0.7750 | 0.1800 | 4 | 3 | global_loss_guard |
  | max_norm=0.3, steps=3, global_guard=0.9 | 0.3310 | 2.4274 -> 2.6038 | 0.8087 | 0.2000 | 3 | 2 | global_loss_guard |

  step 细节：

  | 设置 | Step | 临时/接受后 Global Loss | 阈值 | 结果 |
  |---|---:|---:|---:|---|
  | steps=5, guard=0.8 | 1 | 0.4849 | 0.8 | accepted |
  | steps=5, guard=0.8 | 2 | 0.6451 | 0.8 | accepted |
  | steps=5, guard=0.8 | 3 | 0.7750 | 0.8 | accepted |
  | steps=5, guard=0.8 | 4 | 0.8646 | 0.8 | rejected and rollback |
  | steps=3, guard=0.9 | 1 | 0.5955 | 0.9 | accepted |
  | steps=3, guard=0.9 | 2 | 0.8087 | 0.9 | accepted |
  | steps=3, guard=0.9 | 3 | 0.9167 | 0.9 | rejected and rollback |

  本轮观察：
  1. 增加步数到 5 后，同样 `global_guard=0.8` 可以接受 3 个小步，实际 update L2 达到 0.18；第一组验证的 `steps=3, guard=0.8` 只能接受 1 个小步，update L2 为 0.10。
  2. `steps=5, guard=0.8` 的最终 global loss 为 0.7750，仍守住 0.8；forget loss 提升到 2.5783，比第一组验证的 2.4766 明显更强。
  3. 放宽到 `global_guard=0.9` 后，`steps=3` 能接受 2 个小步，实际 update L2 为 0.20；forget loss 到 2.6038，但 global loss 到 0.8087。
  4. 这说明更细步长比单纯放宽 guard 更适合做可控实验，因为它能更接近阈值边界，而不是一次跨过太多。

  当前阶段结论：
  1. 当前最有价值的本地设置是：
     - `max_norm=0.3`
     - `steps=5`
     - `global_guard=0.8`
  2. 它在 global loss < 0.8 的约束下，比之前单步/粗步设置取得了更强 forget loss。
  3. 下一步若继续找更好点，优先：
     - `steps=5, global_guard=0.85`
     - 或 `steps=10, global_guard=0.8`
     - 暂时不建议继续只放宽到 0.9 以上，否则 global utility 损伤会更明显。


【本轮进度 12：stepped guard 最后三组验证，限定 sweep 收尾】
  本轮目标：
  1. 按预先限定，只再跑 3 组，不继续无边界 sweep。
  2. 验证三件事：
     - `steps=10, global_guard=0.8` 是否能在同样 global 约束下比 `steps=5` 接受更多有效更新。
     - `steps=5, global_guard=0.87` 是否能接受第 4 步，换取更强 forget loss。
     - `max_norm=0.4, steps=10, global_guard=0.8` 是否能靠更大总 trust region 和细步长自动截停到更优位置。

  实验目录：
  - `formal_local_compare_20260429/retained_step_guard_final_sweep/`
  - 汇总文件：
    `formal_local_compare_20260429/retained_step_guard_final_sweep/summary.csv`

  实验配置共同点：
  - retained-set Hessian
  - `unlearn_eta=0.01`
  - `unlearn_ref_loss_guard_ratio=0.0`
  - 其余训练配置沿用前面 dolly/Qwen2-0.5B 小规模正式实验设置。

  本轮三组实验：
  1. `max03_steps10_global08`
     - `unlearn_max_update_norm=0.3`
     - `unlearn_num_steps=10`
     - `unlearn_global_loss_guard_max=0.8`
  2. `max03_steps5_global087`
     - `unlearn_max_update_norm=0.3`
     - `unlearn_num_steps=5`
     - `unlearn_global_loss_guard_max=0.87`
  3. `max04_steps10_global08`
     - `unlearn_max_update_norm=0.4`
     - `unlearn_num_steps=10`
     - `unlearn_global_loss_guard_max=0.8`

  本轮验证：
  - 语法验证仍通过：
    `python -m py_compile main.py server.py client.py scripts/summarize_experiments.py`
  - 汇总脚本正常输出：
    `python scripts/summarize_experiments.py formal_local_compare_20260429\retained_step_guard_final_sweep`

  汇总结果：

  | 设置 | 训练后 Global Loss | Forget Loss | 最终 Global Loss | 实际 Update L2 | Step Attempted | Step Accepted | Stop Reason |
  |---|---:|---:|---:|---:|---:|---:|---|
  | max_norm=0.3, steps=10, global_guard=0.8 | 0.3310 | 2.4274 -> 2.5782 | 0.7776 | 0.1800 | 7 | 6 | global_loss_guard |
  | max_norm=0.3, steps=5, global_guard=0.87 | 0.3310 | 2.4274 -> 2.6534 | 0.8646 | 0.2400 | 5 | 4 | global_loss_guard |
  | max_norm=0.4, steps=10, global_guard=0.8 | 0.3310 | 2.4274 -> 2.5516 | 0.7365 | 0.1600 | 5 | 4 | global_loss_guard |

  step 细节摘要：
  1. `max_norm=0.3, steps=10, guard=0.8`
     - 每步 update L2 约 0.03。
     - 第 6 步后 global loss 为 0.7776，接受。
     - 第 7 步临时 global loss 为 0.8223，超过 0.8，回滚。
  2. `max_norm=0.3, steps=5, guard=0.87`
     - 每步 update L2 约 0.06。
     - 第 4 步后 global loss 为 0.8646，低于 0.87，接受。
     - 第 5 步临时 global loss 为 0.9134，超过 0.87，回滚。
  3. `max_norm=0.4, steps=10, guard=0.8`
     - 每步 update L2 约 0.04。
     - 第 4 步后 global loss 为 0.7365，接受。
     - 第 5 步临时 global loss 为 0.8065，超过 0.8，回滚。

  本轮观察：
  1. `steps=10, guard=0.8` 没有明显优于 `steps=5, guard=0.8`：
     - 二者最终实际 update L2 都约 0.18。
     - forget loss 都约 2.578。
     - 说明在 global loss 0.8 约束下，当前方向可接受的总更新幅度大约就是 0.18。
  2. `guard=0.87` 能接受更大实际 update L2 0.24，并把 forget loss 提升到 2.6534，但 global loss 也升到 0.8646。
  3. 增大总 `max_update_norm` 到 0.4 并没有带来更好结果；由于每步变大，guard 在实际 update L2 0.16 就截停，forget loss 反而低于 max_norm=0.3 的 steps=5/10 设置。
  4. 这说明当前关键不是给更大的总 trust region，而是控制每步粒度和 global loss 上限。

  当前阶段结论：
  1. 如果要求 global loss < 0.8，当前最好的本地候选仍是：
     - `max_norm=0.3`
     - `steps=5` 或 `steps=10`
     - `global_guard=0.8`
     - 实际 update L2 约 0.18
     - forget loss 约 2.578
     - global loss 约 0.775 到 0.778
  2. 如果允许 global loss 到 0.87 附近，`max_norm=0.3, steps=5, global_guard=0.87` 可以得到更强 forget loss 2.6534，但效用代价更明显。
  3. 本轮已经足够支撑“小论文最小方法方案”的实验叙述：stepped guard 能把 retained-set Hessian 的单步不稳定更新转化为可控的、按 global utility 约束截停的更新过程。
  4. 暂时不建议继续做更多相近 sweep；后续应整理结果表和方法描述，而不是继续找小数点级别的折中。


【本轮进度 13：论文实验结果整理、方法段落草稿与最小论文叙述】
  本轮目标：
  1. 不继续跑新的相近参数实验。
  2. 读取现有所有 `summary.csv`，整理成论文写作可直接引用的总表。
  3. 梳理“小论文最小方法方案”的方法描述、消融逻辑和实验结论边界。

  本轮代码与实验状态：
  1. 未修改 `main.py`、`server.py`、`client.py` 或 `scripts/summarize_experiments.py`。
  2. 未启动新的训练或 unlearning 实验。
  3. 本轮只做已有结果整理和论文叙述草稿。

  本轮新增结果整理文件：
  1. `formal_local_compare_20260429/paper_results_summary.csv`
     - 汇总来源覆盖：
       - `final_eta001/summary.csv`
       - `retained_eta_sweep/summary.csv`
       - `retained_clip_sweep/summary.csv`
       - `retained_tiny_eta_sweep/summary.csv`
       - `retained_fine_clip_sweep/summary.csv`
       - `retained_step_guard_validation/summary.csv`
       - `retained_step_guard_sweep/summary.csv`
       - `retained_step_guard_final_sweep/summary.csv`
     - 字段包含 source suite、group、paper role、setting、Hessian mode、eta、max norm、steps、global guard、forget/global loss、update L2、accepted steps、stop reason、unlearning time 和简短 take-away。

  2. `formal_local_compare_20260429/paper_results_summary.md`
     - 提供论文阅读版压缩表。
     - 数值按论文表格习惯四舍五入。
     - 原始高精度数值仍保留在各实验目录的 `summary.csv`。

  论文主表候选结果：

  | 角色 | 设置 | Forget Loss | Final Global Loss | Update L2 | Step Accepted | 用途 |
  |---|---|---:|---:|---:|---:|---|
  | FL baseline | FL | - | 0.2983 | - | - | 普通联邦训练效用基线 |
  | FedHDS baseline | FedHDS | - | 0.3310 | - | - | 本文训练底座 |
  | Forget-batch Hessian baseline | eta=0.01 | 2.4274 -> 3.1713 | 1.1036 | 1.7409 | - | 当前 unlearning baseline，遗忘强但效用损伤明显 |
  | Retained-set Hessian unguarded | eta=0.01 | 2.4274 -> 9.7236 | 9.9319 | 2.9278 | - | 说明单步 retained-set 更新不稳定 |
  | Single-step clipping | max_norm=0.15 | 2.4274 -> 2.5388 | 0.7162 | 0.1500 | - | global loss < 0.8 下最稳的单步 clipping 点 |
  | Stepped guard candidate | max_norm=0.3, steps=5, guard=0.8 | 2.4274 -> 2.5783 | 0.7750 | 0.1800 | 3/4 | global loss < 0.8 下当前最佳本地候选之一 |
  | Stepped guard candidate | max_norm=0.3, steps=10, guard=0.8 | 2.4274 -> 2.5782 | 0.7776 | 0.1800 | 6/7 | 与 steps=5 基本等价，说明 0.8 约束下可接受更新约为 0.18 |
  | Relaxed guard candidate | max_norm=0.3, steps=5, guard=0.87 | 2.4274 -> 2.6534 | 0.8646 | 0.2400 | 4/5 | 如果允许 global loss 到 0.87 附近，可换取更强遗忘 |

  论文方法段落草稿：

  本文在 FedHDS 训练得到的全局模型基础上研究客户端级 federated unlearning。给定需要遗忘的客户端 `j`，我们首先使用该客户端数据估计遗忘梯度
  `g_j = grad L_j(theta)`，用于刻画“需要从模型中移除的方向”。与直接在 forget client batch 上估计 Hessian 不同，本文使用非遗忘客户端构成的小规模 retained reference set `D_ref` 来近似系统保留任务的曲率：
  `H_ref = grad^2 L_ref(theta)`。随后通过 LiSSA/HVP 近似求解 `H_ref^{-1} g_j`，形成 retained-set curvature approximation 下的 unlearning update：
  `Delta theta = eta / (K - 1) * H_ref^{-1} g_j`。

  为避免 retained-set Hessian 单步更新过大导致 global utility 崩溃，本文引入 global update norm clipping。具体而言，在真正写回参数前，先计算整体现实更新 `Delta theta` 的 global L2 norm；若其超过阈值 `tau`，则按 `tau / ||Delta theta||_2` 对整次更新缩放。该步骤不改变 Hessian 参考来源，只限制 unlearning update 的实际幅度，使 retained-set 方向从不可控的大步变成有界更新。

  进一步地，本文将 clipped update 均分为 `S` 个小步，并在每个小步后执行 trust-region guard。每一步先临时应用小更新，再评估 global loss 或 retained/reference loss；若超过预设阈值，则回滚当前小步并终止后续更新。这样，unlearning 不再依赖一次性接受完整 Hessian update，而是变成可截停、可回滚的受约束更新过程。

  论文实验结论草稿：
  1. 工程链路已经闭环：FedHDS 训练、forget-batch Hessian unlearning、retained-set Hessian unlearning、clipping、stepped guard 和结果汇总脚本都可运行。
  2. retained-set Hessian 能提供明确的遗忘方向：随着实际 update L2 增大，forget client loss 会升高。
  3. 单步 retained-set Hessian 更新不稳定：eta=0.01 未加控制时 global loss 从 0.3310 崩到 9.9319。
  4. global L2 clipping 能把灾难性更新变成有界更新，但单步 clipping 仍然存在明显 forget/global utility trade-off。
  5. stepped guard 的主要贡献不是无代价提高 forget loss，而是把 retained-set Hessian 的强更新转化为受 global utility 约束的 trust-region 过程；在 global loss < 0.8 的约束下，当前最合理候选是 `max_norm=0.3, steps=5/10, guard=0.8`，forget loss 约 2.578，global loss 约 0.775-0.778。
  6. 如果论文需要展示“更强遗忘但效用代价更高”的点，可以报告 `max_norm=0.3, steps=5, guard=0.87`，forget loss 2.6534，global loss 0.8646。
  7. 当前结果不能宣称 retained-set Hessian 全面优于 forget-batch baseline。更稳妥的表述是：retained-set curvature approximation 提供了可解释的遗忘方向，而 clipping/stepped guard 将原本不稳定的 retained-set 更新变成可控、可回滚、可按效用阈值截停的 unlearning procedure。

  建议的小论文消融逻辑：
  1. FL vs FedHDS：说明训练底座和基础 global utility。
  2. FedHDS + forget-batch Hessian：作为已有 unlearning baseline，展示遗忘强度和效用代价。
  3. FedHDS + retained-set Hessian unguarded：展示 retained-set 曲率方向如果不控制会不稳定。
  4. + global L2 clipping：展示 clipping 可以避免 eta=0.01 的灾难性崩溃，并形成 update L2 与 forget/global loss 的稳定 trade-off 曲线。
  5. + stepped trust-region guard：展示 guard 能按 global loss 阈值自动接受/回滚小步，并给出 global loss < 0.8 下的当前最佳折中点。

  当前阶段判断：
  1. 现有结果已经足够支撑“小论文最小方法方案”的方法和实验叙述。
  2. 短期不建议继续做大量相近参数 sweep。
  3. 下一步更应该进入论文写作层面：
     - 固化主表和消融表；
     - 写方法小节；
     - 写实验设置小节；
     - 写结果分析和 limitation；
     - 明确避免“全面优于 baseline”的过度表述。


【本轮进度 14：2026-04-30 最小 smoke 复核】
  本轮目标：
  1. 重新读取 readme1/readme2 后，确认当前代码仍满足“小论文最小方法方案”和“按文件拆分的最小改造清单”。
  2. 不改算法代码，只复核 baseline 与 retained-set Hessian 两条 unlearning 路径是否还能跑通。

  本轮环境说明：
  1. Windows 默认 `python` 缺少 `torch`，不能用于项目运行。
  2. 使用 WSL 中的 HG 环境：
     `/mnt/f/ANACONDA/envs/HG/bin/python`
  3. Hugging Face 网络不可达，因此改用本地缓存 snapshot 离线加载模型：
     `/home/fred/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8`

  本轮验证：
  1. 语法验证通过：
     `python -m py_compile client.py server.py main.py scripts/summarize_experiments.py`
  2. baseline smoke 跑通：
     - 路径：`formal_local_compare_20260429/smoke_20260430/baseline/`
     - Hessian mode：`forget-batch`
     - Forget gradient sample size：2
     - LiSSA depth / damping：1 / 0.01
     - Forget loss：2.6000 -> 10.0000
     - Unlearning time：1.49s
  3. retained-set Hessian smoke 跑通：
     - 路径：`formal_local_compare_20260429/smoke_20260430/retained/`
     - Hessian mode：`retained-set`
     - Retained reference set size：6
     - Forget gradient sample size：2
     - LiSSA depth / damping：1 / 0.01
     - Forget loss：2.6000 -> 16.8029
     - Unlearning time：17.20s
  4. 已生成汇总文件：
     `formal_local_compare_20260429/smoke_20260430/summary.csv`

  本轮结论：
  1. readme2 的三个完成标准仍然满足：
     - `--use_fedhds_unlearn` 且不加 `--use_retained_hessian` 时，老路径能跑。
     - 加上 `--use_retained_hessian` 后，新路径能跑。
     - 日志能打印 forget gradient sample size、retained reference set size、LiSSA depth / damping。
  2. 本轮没有发现需要修代码的问题。
  3. 当前项目进度已经超过 readme1/readme2 的最小改造阶段；下一步应继续从 readme3 的进度 13 往后做论文写作整理，而不是重复实现 retained Hessian。


【本轮进度 15：实验工具链完善，出图/论文表/云端批跑模板】
  本轮目标：
  1. 不再改 unlearning 核心算法。
  2. 补齐云服务器正式实验前需要的工具链：
     - 自动出图
     - 自动生成论文压缩表
     - 云端批量运行模板
     - 工具使用说明

  本轮新增文件：
  1. `scripts/plot_experiments.py`
     - 输入 `summary.csv`
     - 默认输出到 `<summary.csv 所在目录>/figures/`
     - 支持 `--formats png,pdf`
     - 当前生成图包括：
       - `forget_loss_before_after`
       - `global_loss_final`
       - `update_norm_vs_global_loss`
       - `update_norm_vs_forget_gain`
       - `guard_steps`
       - `unlearning_time`

  2. `scripts/make_paper_table.py`
     - 输入 `summary.csv`
     - 输出：
       - `paper_table.csv`
       - `paper_table.md`
     - 字段压缩为论文表格常用内容：
       - setting
       - Hessian mode
       - forget loss before/after
       - final global loss
       - update L2
       - accepted/attempted steps
       - unlearning time
       - note

  3. `scripts/run_cloud_experiments.sh`
     - 云服务器批跑模板
     - 支持 `RUN_SMOKE=1` 先做小规模 smoke
     - 默认实验组包括：
       - `fl`
       - `fedhds`
       - `forget_batch_hessian`
       - `retained_guard`
     - 跑完后会自动执行：
       - `summarize_experiments.py`
       - `make_paper_table.py`
       - `plot_experiments.py`

  4. `scripts/README_EXPERIMENT_TOOLS.md`
     - 记录汇总、出表、出图和云端运行命令。

  本轮验证：
  1. 语法验证通过：
     `python -m py_compile scripts/plot_experiments.py scripts/make_paper_table.py scripts/summarize_experiments.py`
  2. 使用本轮 smoke 结果验证出表成功：
     - `formal_local_compare_20260429/smoke_20260430/paper_table.csv`
     - `formal_local_compare_20260429/smoke_20260430/paper_table.md`
  3. 使用本轮 smoke 结果验证出图成功：
     - `formal_local_compare_20260429/smoke_20260430/figures/forget_loss_before_after.png`
     - `formal_local_compare_20260429/smoke_20260430/figures/global_loss_final.png`
     - `formal_local_compare_20260429/smoke_20260430/figures/update_norm_vs_global_loss.png`
     - `formal_local_compare_20260429/smoke_20260430/figures/update_norm_vs_forget_gain.png`
     - `formal_local_compare_20260429/smoke_20260430/figures/guard_steps.png`
     - `formal_local_compare_20260429/smoke_20260430/figures/unlearning_time.png`
  4. 使用已有 `retained_step_guard_final_sweep/summary.csv` 验证多行 sweep 结果也能出表和出图。

  当前阶段结论：
  1. 代码核心链路、结果表格汇总、论文压缩表、图片展示、云端批跑模板都已经具备。
  2. 现在缺的不是本地代码能力，而是云服务器上的正式大规模实验结果。
  3. 等云服务器准备好后，建议先执行 `RUN_SMOKE=1 bash scripts/run_cloud_experiments.sh`，确认服务器环境、数据、模型和 GPU 都正常，再跑正式配置。


【给下一次 Codex 的交接提示词：AutoDL 云端实验继续】
  请先阅读项目根目录下的 `readme1.txt`、`readme2.txt`、`readme3.txt`，尤其是 readme3 的“进度 14”和“进度 15”。

  当前项目状态：
  1. unlearning 核心代码已经完成，不要重复重写 retained-set Hessian 主逻辑。
  2. 当前核心代码路径包括：
     - `client.py`
     - `server.py`
     - `main.py`
  3. 实验工具链已经补齐：
     - `scripts/summarize_experiments.py`
     - `scripts/make_paper_table.py`
     - `scripts/plot_experiments.py`
     - `scripts/run_cloud_experiments.sh`
     - `scripts/README_EXPERIMENT_TOOLS.md`
  4. 本地已经验证过：
     - forget-batch baseline smoke 能跑
     - retained-set Hessian smoke 能跑
     - `summary.csv` 能生成
     - `paper_table.csv/md` 能生成
     - figures 图片能生成
  5. 当前缺的是 AutoDL 云服务器上的正式实验结果，不是本地代码能力。

  用户下一次可能会提供：
  1. AutoDL 实例截图或环境截图，可能放在 `图库.txt` 或其它文件里。
  2. 服务器 GPU 型号、镜像、终端输出或 SSH 信息。
  3. 用户没有完整空闲时间，因此优先做短任务，不要直接启动 8 小时正式实验。

  下一次建议执行顺序：
  1. 先读取用户提供的 `图库.txt` 或截图，确认云服务器配置和当前登录方式。
  2. 让用户先在 AutoDL 终端执行并提供这些输出：
     ```bash
     nvidia-smi
     python -V
     conda --version || true
     df -h
     free -h
     pwd
     ```
  3. 如果服务器可用，先做环境准备：
     - 上传或拉取项目
     - 确认 `data/databricks-dolly-15k.jsonl` 存在
     - 安装/检查依赖
     - 确认 PyTorch CUDA 可用
  4. 不要立刻跑正式 8 小时实验。先跑云端 smoke：
     ```bash
     RUN_SMOKE=1 \
     PYTHON_BIN=python \
     MODEL=Qwen/Qwen2-0.5B \
     DATA_PATH=/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl \
     DEVICE=0 \
     RESULT_ROOT=formal_cloud_results \
     bash scripts/run_cloud_experiments.sh
     ```
     注意：`DATA_PATH` 和项目路径要按服务器实际路径调整。
  5. smoke 成功后，检查：
     - `formal_cloud_results/smoke/summary.csv`
     - `formal_cloud_results/smoke/paper_table.md`
     - `formal_cloud_results/smoke/figures/`
  6. 如果 smoke 成功但用户没时间，就停在这里，不要继续跑正式实验。
  7. 等用户有较长空闲时间，再跑正式 0.5B 四组实验。

  云服务器选择建议：
  1. 首选 AutoDL 单卡 `RTX 5090 / 32GB`，性价比更适合当前 LoRA + Qwen2-0.5B。
  2. `RTX PRO 6000 / 96GB` 更稳但更贵，当前小论文第一阶段不是必须。
  3. 当前不要租多卡，因为代码主要是单卡流程。

  第一阶段正式实验规划：
  1. 先用 LoRA + Qwen2-0.5B 跑完整四组：
     - FL
     - FedHDS
     - FedHDS + forget-batch Hessian
     - FedHDS + retained-set Hessian / retained guard
  2. 第一阶段产出：
     - `final_results.json`
     - `summary.csv`
     - `paper_table.csv`
     - `paper_table.md`
     - `figures/*.png` 或 `figures/*.pdf`
  3. 第一阶段完整正式实验预计可能需要数小时，用户今天下午有课时不要强行启动。

  重要边界：
  1. 现在的小论文路线是先完成 LoRA + Qwen2-0.5B 的完整云端正式结果。
  2. 第二阶段再扩大 data_sample、rounds、num_clients、seed、forget client。
  3. 第三阶段再考虑 Qwen2.5-1.5B、Qwen2-1.5B 或 Qwen2.5-3B。
  4. 不要一开始就上 7B。


【本轮进度 16：AutoDL 购买前确认与云端最小实验边界】
  本轮目标：
  1. 复核 readme1/readme2/readme3 后，确认当前任务是否已经进入 AutoDL 云端最小实验阶段。
  2. 明确购买云服务器的推荐规格、计费方式和第一步运行边界。
  3. 不修改算法代码，不启动新的本地或云端训练。

  当前判断：
  1. 当前核心算法、结果汇总、论文表格和出图工具链已经具备。
  2. 现在确实可以进入 AutoDL 云服务器购买阶段。
  3. 云端第一目标不是做大量 sweep，而是跑通 LoRA + Qwen2-0.5B 的最小完整结果。
  4. 云端执行顺序必须是：
     - 先购买/启动单卡实例；
     - 先做环境检查；
     - 先跑 `RUN_SMOKE=1 bash scripts/run_cloud_experiments.sh`；
     - smoke 成功后，再根据时间决定是否跑正式四组最小实验。

  AutoDL 实例建议：
  1. 优先选择单卡、显存 >= 32GB 的 GPU。
  2. 首选仍是单卡 `RTX 5090 / 32GB`，如果平台当时无货，可以选择单卡 48GB 或更高显存型号。
  3. `RTX 4090 / 24GB` 可作为预算型 smoke 选择，但 retained-set Hessian / HVP 阶段更容易受显存影响，因此不作为首选正式实验配置。
  4. 暂时不要租多卡；当前代码和实验脚本主要按单卡流程设计。
  5. 暂时不要租 7B 级别所需的大显存高价机器；第一阶段只做 Qwen2-0.5B。

  计费建议：
  1. 第一次购买建议选择按量计费，便于 smoke 失败或环境不合适时及时关机止损。
  2. AutoDL 官方文档说明实例运行中才开始按实例时长计费，但计费按实例开关机时间计算，不按 GPU 是否实际计算判断；因此不用时必须关机。
  3. 调试、上传、整理文件等不需要 GPU 的阶段，可考虑关机后使用无卡模式降低成本。

  SSH 公钥说明：
  1. 用户提供的 `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICSWljVIG1U5mG1Ayi2bsq9zLMNGDLjiLsStWZQzHH4Q huudungdang629@gmail.com` 是公钥格式，可填入 AutoDL 的 SSH 公钥/密钥登录配置中。
  2. 不应泄露私钥文件；只复制 `.pub` 公钥内容到平台。

  购买后第一步让服务器输出：
  ```bash
  nvidia-smi
  python -V
  conda --version || true
  df -h
  free -h
  pwd
  ```

  云端 smoke 命令模板：
  ```bash
  RUN_SMOKE=1 \
  PYTHON_BIN=python \
  MODEL=Qwen/Qwen2-0.5B \
  DATA_PATH=/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl \
  DEVICE=0 \
  RESULT_ROOT=formal_cloud_results \
  bash scripts/run_cloud_experiments.sh
  ```

  注意：
  1. `DATA_PATH` 和项目路径需要按 AutoDL 实例实际目录调整。
  2. smoke 成功后先检查：
     - `formal_cloud_results/smoke/summary.csv`
     - `formal_cloud_results/smoke/paper_table.md`
     - `formal_cloud_results/smoke/figures/`
  3. 如果 smoke 成功但用户没有连续数小时空闲时间，就先停止，不要直接跑正式四组。

  阶段性结论：
  1. 用户现在可以购买 AutoDL 云服务器。
  2. 推荐购买单卡、>=32GB 显存、按量计费实例。
  3. 第一阶段只做最小云端闭环，不做大规模 sweep，不直接上更大模型。


【本轮进度 17：AutoDL 购买协作流程约定】
  本轮目标：
  1. 用户准备首次购买 AutoDL 云服务器，需要逐步辅助，不默认用户熟悉云服务器操作。
  2. 用户会把购买和配置过程截图放到“图库”相关文件或目录中，后续根据截图判断下一步。
  3. 继续保持云端最小实验边界：先环境检查和 smoke，不直接启动长时间正式实验。

  协作方式：
  1. 用户每完成一个 AutoDL 页面步骤后，可以把截图放到项目目录下的 `图库` 文件夹，或保存为 `图库.txt` 记录截图路径/页面信息。
  2. 如果是图片文件，建议文件名包含步骤含义，例如：
     - `01_租用实例页面.png`
     - `02_GPU和镜像选择.png`
     - `03_实例开机后SSH信息.png`
     - `04_终端环境检查输出.png`
  3. 用户放好截图后，只需要说明“图库已更新”，后续读取截图或文件内容后再给下一步建议。

  购买页面优先检查项：
  1. 计费方式：优先按量计费。
  2. GPU 数量：单卡。
  3. GPU 显存：首选 >=32GB。
  4. GPU 型号：优先 RTX 5090 / 32GB；没有则选择其他单卡 >=32GB 显存实例。
  5. 镜像：选择 PyTorch + CUDA 的深度学习镜像，避免选择纯 Python 或无 CUDA 镜像。
  6. 磁盘：需要能容纳项目、数据、模型缓存和实验输出；第一阶段 Qwen2-0.5B 通常不需要特别大的磁盘，但不能太小。
  7. SSH 公钥：可以使用用户提供的 ed25519 公钥；不要上传或泄露私钥。

  购买后不要立即做的事：
  1. 不要直接跑正式四组实验。
  2. 不要直接切换到 1.5B/3B/7B。
  3. 不要租多卡。
  4. 不要继续做大量相近参数 sweep。

  购买后第一步：
  1. 获取实例 SSH 登录信息。
  2. 登录后先执行环境检查命令：
     ```bash
     nvidia-smi
     python -V
     conda --version || true
     df -h
     free -h
     pwd
     ```
  3. 把输出截图或文字放入“图库”相关文件/目录，再根据结果决定项目上传、依赖安装和 smoke 命令。

  阶段性结论：
  1. 可以开始购买，但每一步先截图确认。
  2. 第一目标仍是云端 smoke 成功，而不是立刻拿最终论文结果。


【本轮进度 18：AutoDL 创建开机前确认】
  本轮问题：
  1. 用户询问是否可以直接点击创建开机。

  答复条件：
  1. 如果当前 AutoDL 页面已经满足以下条件，可以点击创建开机：
     - 按量计费；
     - 单卡；
     - GPU 显存 >= 32GB，优先 RTX 5090 / 32GB；
     - 镜像为 PyTorch + CUDA 深度学习环境；
     - 磁盘空间至少 50GB，推荐 100GB；
     - SSH 公钥已正确填写，且没有上传私钥；
     - 没有选择多卡、7B、大模型专用高价配置。
  2. 如果页面上有任何不确定项，应先截图放入 `图库`，不要急着开机。

  创建后边界：
  1. 创建并开机后只做环境检查，不直接跑正式实验。
  2. 首次登录后先执行：
     ```bash
     nvidia-smi
     python -V
     conda --version || true
     df -h
     free -h
     pwd
     ```
  3. 把输出截图或文字放入 `图库` 后，再决定下一步项目上传和 smoke。


【本轮进度 19：图库截图确认 SSH 公钥与镜像选择】
  本轮输入：
  1. 用户将 AutoDL 页面截图保存到 `图库.docx`。
  2. 读取截图后确认当前页面是“添加 SSH 公钥”弹窗，不是实例创建时的镜像选择页面。

  当前截图判断：
  1. SSH 公钥输入框已经填入 ed25519 公钥。
  2. 备注填写为 `Malenia`，可用作该公钥的备注名。
  3. 页面提示“正在运行中的实例需重启实例后生效”，这对后续新建实例没有问题；如果已有运行实例，要重启才会应用新公钥。

  操作建议：
  1. 当前页面先点击 `确定` 保存 SSH 公钥。
  2. 随后回到算力市场/创建实例页面。
  3. 创建实例时必须选择镜像。
  4. 镜像优先选择 AutoDL 官方或平台预置的 PyTorch + CUDA 深度学习镜像。
  5. 不要选择纯 Python、CPU、无 CUDA 或明显面向其他框架的镜像。

  镜像选择偏好：
  1. 优先：PyTorch 2.x + CUDA 12.x + Python 3.10/3.11。
  2. 可接受：PyTorch 2.x + CUDA 11.8 + Python 3.10。
  3. 避免：TensorFlow-only、Paddle-only、CPU-only、过旧 CUDA/Python 镜像。

  下一步：
  1. 用户保存公钥后，进入创建实例页。
  2. 将 GPU、计费、磁盘、镜像选择区域截图继续放入图库。
  3. 再根据截图确认是否可以点击创建开机。


【本轮进度 20：AutoDL 创建实例页截图确认与镜像选择建议】
  本轮输入：
  1. 用户更新 `图库.docx`，截图为 AutoDL 创建实例页面。
  2. 本轮通过只读共享复制方式成功提取最新截图，避免 docx 被 Word/WPS 占用导致读取失败。

  当前截图确认到的配置：
  1. 计费方式：按量计费。
  2. GPU：RTX 5090，32GB。
  3. GPU 数量：1 卡。
  4. CPU：25 核。
  5. 内存：90GB。
  6. 驱动/CUDA：驱动 580.76.05，CUDA 13.0。
  7. 数据盘：免费 50GB。
  8. 价格：约 2.78 元/小时。
  9. 当前镜像区域尚未选择具体镜像，下拉框显示“请选择框架名称/框架版本/Python版本/CUDA版本”。

  判断：
  1. 当前 GPU、单卡、按量计费配置符合第一阶段 LoRA + Qwen2-0.5B 云端最小实验需求。
  2. 数据盘 50GB 对 smoke 和第一阶段 0.5B 最小实验通常够用；如果后续缓存多个模型或保存更多结果，再考虑扩容。
  3. 现在必须选择镜像；不建议在未选择 PyTorch/CUDA 镜像时点击创建开机。

  镜像选择建议：
  1. 优先选择“基础镜像”里的 PyTorch 镜像。
  2. 首选组合：
     - PyTorch 2.8 或 2.7；
     - Python 3.10 或 3.11；
     - CUDA 12.x。
  3. 如果没有 PyTorch 2.8/2.7，可选 PyTorch 2.5/2.6 + CUDA 12.x + Python 3.10。
  4. 避免选择：
     - CPU-only；
     - 纯 Python；
     - TensorFlow-only；
     - Paddle-only；
     - CUDA 过旧或 Python 过旧的镜像。
  5. 暂时不需要优先选社区镜像，除非基础镜像里没有合适的 PyTorch + CUDA。

  下一步：
  1. 用户点击镜像下拉框，选择 PyTorch + CUDA 镜像。
  2. 选择后再把镜像区域截图放入图库。
  3. 若镜像为合适的 PyTorch/CUDA 组合，即可点击“创建并开机”。


【本轮进度 21：镜像下拉框第二张截图确认】
  本轮输入：
  1. 用户更新 `图库.docx`，并说明需要查看第二张图片。
  2. 本轮成功提取到 `image1.png` 和 `image2.png`，其中第二张为镜像下拉框截图。

  第二张截图内容：
  1. 框架名称已停留在 `PyTorch`。
  2. 框架版本已停留在 `2.8.0`。
  3. Python 版本列显示 `3.12(ubuntu22.04)`。
  4. 当前下拉框底部仍显示“请选择框架名称/框架版本/Python版本/CUDA版本”，并提示“请选择镜像”。

  判断：
  1. 选择 `PyTorch 2.8.0` 的方向是正确的。
  2. 但当前还没有完成最终镜像选择，需要继续点击右侧 Python 版本，并选择后续 CUDA 版本。
  3. Python 3.12 对当前项目依赖存在一定兼容风险；项目本地更推荐 Python 3.10 或 3.11。
  4. 如果该镜像列表只给 PyTorch 2.8.0 + Python 3.12，也可以运行 smoke，但后续安装依赖时可能更容易遇到包兼容问题。

  操作建议：
  1. 优先查找：
     - PyTorch 2.8.0 + Python 3.10/3.11 + CUDA 12.x；
     - 或 PyTorch 2.7.0/2.6.0/2.5.1 + Python 3.10/3.11 + CUDA 12.x。
  2. 如果只有 `PyTorch 2.8.0 + Python 3.12 + CUDA 12.x/13.0`，可以继续选择，但后续要先做环境检查和依赖 smoke。
  3. 选择流程应继续点击 `3.12(ubuntu22.04)` 右侧箭头，直到出现 CUDA 版本，再选择 CUDA 版本完成镜像选择。

  下一步：
  1. 用户继续点右侧 Python 版本和 CUDA 版本。
  2. 选择完成后，确认输入框不再提示“请选择镜像”。
  3. 再截图确认是否可以创建开机。


【本轮进度 22：镜像 CUDA 版本出现但尚未最终选中】
  本轮输入：
  1. 用户再次更新 `图库.docx` 并询问当前是否可以。
  2. 读取第二张截图后，确认镜像下拉框已经展开到 CUDA 版本列。

  当前截图内容：
  1. 框架名称：PyTorch。
  2. 框架版本：2.8.0。
  3. Python 版本：3.12(ubuntu22.04)。
  4. CUDA 版本：12.8。
  5. 底部输入框仍显示“请选择框架名称/框架版本/Python版本/CUDA版本”。
  6. 页面仍有红字“请选择镜像”。

  判断：
  1. 当前选择方向正确，`PyTorch 2.8.0 + CUDA 12.8` 符合项目第一阶段云端实验需求。
  2. Python 3.12 不是最理想版本，但 PyTorch 2.8.0 与 CUDA 12.8 的组合可以先用于 smoke；后续若依赖安装失败，再在实例内创建 Python 3.10/3.11 conda 环境。
  3. 当前还不能直接点击“创建并开机”，因为镜像尚未最终选中。

  操作建议：
  1. 用户需要点击最右侧的 `12.8` 这一项，完成镜像选择。
  2. 点击后应确认底部镜像输入框不再显示占位文字，且红字“请选择镜像”消失。
  3. 如果镜像框显示类似 `PyTorch 2.8.0 / Python 3.12 / CUDA 12.8`，即可点击“创建并开机”。

  后续边界：
  1. 创建开机后先做环境检查。
  2. 不直接跑正式四组实验。


【本轮进度 23：Python 3.10 与 PyTorch 新版本镜像取舍】
  本轮问题：
  1. 用户询问如果选择 Python 3.10，框架版本需要落后一档，应该选择哪个。
  2. 用户提到第三张图片，但本轮从 `图库.docx` 中仍只提取到两张图片，可能第三张尚未成功保存进 docx。

  项目依赖背景：
  1. `requirement.txt` 当前记录了 `torch==2.8.0`、`transformers==4.57.6`、`peft==0.17.1` 等依赖。
  2. 项目实际更关心 PyTorch/CUDA 可用、transformers/peft 能安装、Qwen2-0.5B 能加载和训练 smoke 能跑通。
  3. Python 3.12 对部分机器学习依赖包更容易出现兼容或 wheel 缺失问题；Python 3.10/3.11 通常更稳。

  取舍判断：
  1. 如果可选 `PyTorch 2.5/2.6/2.7 + Python 3.10 + CUDA 12.x`，优先选该类组合。
  2. 不必为了 PyTorch 2.8.0 强行选择 Python 3.12。
  3. 对当前 Qwen2-0.5B + LoRA + HVP smoke 来说，PyTorch 2.5 及以上通常已经足够；Python 版本稳定性更重要。
  4. 最推荐选择顺序：
     - PyTorch 2.7/2.6 + Python 3.10 + CUDA 12.x；
     - PyTorch 2.5.1 + Python 3.10 + CUDA 12.x；
     - PyTorch 2.8.0 + Python 3.12 + CUDA 12.8；
     - 更旧或非 PyTorch 镜像不推荐。

  操作建议：
  1. 若页面存在 `PyTorch 2.5.1 + Python 3.10 + CUDA 12.x`，可以直接选它。
  2. 如果只有 `PyTorch 2.8.0 + Python 3.12 + CUDA 12.8`，也可以先选，但后续如果依赖安装失败，要在实例内创建 Python 3.10 conda 环境。
  3. 选择完成后仍需确认镜像输入框显示完整镜像名称，红字“请选择镜像”消失，再点击创建开机。


【本轮进度 24：第三张镜像截图确认，2.1.2/Python3.10/CUDA11.8 不推荐】
  本轮输入：
  1. 用户再次保存 `图库.docx`，本轮成功提取到第三张图片。
  2. 第三张截图显示镜像下拉框中选择到：
     - PyTorch；
     - 框架版本 2.1.2；
     - Python 3.10(ubuntu22.04)；
     - CUDA 11.8。

  判断：
  1. 该组合的 Python 3.10 是稳定选择。
  2. 但 PyTorch 2.1.2 + CUDA 11.8 对 RTX 5090 这类新 GPU 不理想，可能存在架构支持、性能或兼容风险。
  3. 当前不建议为了 Python 3.10 降到 PyTorch 2.1.2。

  更新后的镜像优先级：
  1. 如果存在：PyTorch 2.5.1/2.6/2.7 + Python 3.10 + CUDA 12.x，优先选择。
  2. 如果只有两个选择：
     - PyTorch 2.8.0 + Python 3.12 + CUDA 12.8；
     - PyTorch 2.1.2 + Python 3.10 + CUDA 11.8；
     则优先选择 PyTorch 2.8.0 + Python 3.12 + CUDA 12.8。
  3. 理由是 RTX 5090 更需要较新的 PyTorch/CUDA 栈；Python 3.12 的包兼容问题可以后续通过 conda 环境处理，但旧 CUDA/PyTorch 对新 GPU 的风险更关键。

  操作建议：
  1. 先尝试在 PyTorch 2.5.1、2.7.0 等版本下查看是否有 Python 3.10/3.11 + CUDA 12.x。
  2. 如果没有，回到 PyTorch 2.8.0 + Python 3.12 + CUDA 12.8，点击最右侧 CUDA 12.8 完成选择。
  3. 不建议选择 PyTorch 2.1.2 + CUDA 11.8 作为 RTX 5090 实例的基础镜像。


【本轮进度 25：AutoDL 创建开机确认】
  本轮输入：
  1. 用户更新 `图库.docx`，询问是否可以创建开机。
  2. 最新截图显示创建实例页面已完成关键配置。

  当前最终配置：
  1. 计费方式：按量计费。
  2. 主机：243机。
  3. GPU：RTX 5090 32GB。
  4. GPU 数量：1 卡。
  5. CPU：25 核。
  6. 内存：90GB。
  7. 系统盘：30GB。
  8. 数据盘：免费 50GB SSD。
  9. 驱动/CUDA：驱动 580.76.05，CUDA 13.0。
  10. 镜像：PyTorch / 2.8.0 / 3.12(ubuntu22.04) / 12.8。
  11. 价格：约 2.78 元/小时。

  判断：
  1. 镜像已经完整选中，页面不再显示“请选择镜像”红字。
  2. 当前配置可以点击“创建并开机”。
  3. Python 3.12 不是最稳版本，但该镜像的新 PyTorch/CUDA 栈更适合 RTX 5090；后续依赖若有问题，再建 Python 3.10/3.11 conda 环境。

  创建后立即执行：
  ```bash
  nvidia-smi
  python -V
  conda --version || true
  df -h
  free -h
  pwd
  ```

  创建后边界：
  1. 不直接跑正式四组实验。
  2. 不直接下载大模型或换 1.5B/3B/7B。
  3. 先把环境检查输出放入图库，再决定项目上传、依赖安装和 cloud smoke。


【本轮进度 26：AutoDL 实例已创建，进入首次登录与环境检查】
  本轮输入：
  1. 用户已成功创建 AutoDL 实例。
  2. 用户提供了 SSH 登录地址：
     - host: `connect.westc.seetacloud.com`
     - port: `56117`
     - user: `root`
  3. 用户同时提供了登录密码；密码属于敏感信息，不写入 readme3，不应继续在项目文件中保存。

  当前任务边界：
  1. 现在只做首次登录和环境检查。
  2. 暂时不上传项目、不安装依赖、不运行 smoke、不跑正式实验。
  3. 环境确认无误后，下一步再决定项目同步方式和 cloud smoke 命令。

  建议用户本地执行登录命令：
  ```bash
  ssh -p 56117 root@connect.westc.seetacloud.com
  ```
  首次连接如提示是否继续连接，输入 `yes`。
  随后输入用户自己的实例密码。

  登录成功后先执行：
  ```bash
  nvidia-smi
  python -V
  conda --version || true
  df -h
  free -h
  pwd
  ```

  下一步：
  1. 用户把上述输出截图或文字放入 `图库.docx`。
  2. 根据输出判断：
     - GPU/驱动/CUDA 是否正常；
     - Python/conda 是否可用；
     - 磁盘是否够用；
     - 当前工作目录在 `/root` 还是 `/root/autodl-tmp`。
  3. 再决定项目上传、依赖安装和 `RUN_SMOKE=1` 云端 smoke。


【本轮进度 27：面向首次使用者的 SSH 登录操作步骤】
  本轮问题：
  1. 用户表示不会执行 SSH 登录，需要一步一步指导。

  操作边界：
  1. 不在 readme3 中记录实例密码。
  2. 先只登录和检查环境，不上传项目、不安装依赖、不跑实验。

  Windows 本地推荐操作：
  1. 按 `Win` 键，搜索并打开 `PowerShell` 或 `Windows Terminal`。
  2. 在终端中执行：
     ```bash
     ssh -p 56117 root@connect.westc.seetacloud.com
     ```
  3. 如果首次连接提示是否继续连接，输入：
     ```bash
     yes
     ```
  4. 出现密码提示后，输入 AutoDL 实例密码并回车。输入密码时终端不显示字符是正常现象。
  5. 登录成功后，提示符通常会变成类似 `root@...:~#`。
  6. 然后执行环境检查：
     ```bash
     nvidia-smi
     python -V
     conda --version || true
     df -h
     free -h
     pwd
     ```

  如果 Windows 提示找不到 `ssh`：
  1. 可先使用 AutoDL 网页里的终端入口完成环境检查。
  2. 或再处理 Windows OpenSSH 客户端问题。


【本轮进度 28：首次 SSH 登录成功】
  本轮输入：
  1. 用户在 Windows PowerShell 中执行：
     ```bash
     ssh -p 56117 root@connect.westc.seetacloud.com
     ```
  2. 首次连接时用户输入 `yes` 后，SSH 已添加 host key 到 known_hosts。
  3. 当前终端提示符已变为：
     ```bash
     root@autodl-container-65f24387ad-4efb5c17:~#
     ```

  当前判断：
  1. SSH 登录已经成功。
  2. 用户没有操作错误。
  3. 当前已经处于 AutoDL Ubuntu 22.04 容器内。

  AutoDL 登录欢迎信息确认到的初始资源：
  1. CPU：25 核。
  2. 内存：90GB。
  3. GPU：NVIDIA GeForce RTX 5090，1 卡。
  4. 系统盘 `/`：30GB。
  5. 数据盘 `/root/autodl-tmp`：50GB。

  下一步：
  1. 在当前 `root@...#` 提示符后继续执行环境检查命令。
  2. 先不上传项目、不安装依赖、不跑 smoke。


【本轮进度 29：AutoDL 首次环境检查结果】
  本轮输入：
  1. 用户已在 AutoDL SSH 会话中执行环境检查命令。
  2. 输出确认用户当前已经登录到：
     ```bash
     root@autodl-container-65f24387ad-4efb5c17:~#
     ```

  环境检查关键结果：
  1. `nvidia-smi`：
     - NVIDIA-SMI：580.76.05
     - Driver Version：580.76.05
     - CUDA Version：13.0
     - GPU：NVIDIA GeForce RTX 5090
     - 显存：32607MiB
     - 进程：No running processes found
  2. Python：
     - `Python 3.12.3`
  3. conda：
     - `conda 24.4.0`
  4. 磁盘：
     - 系统盘 `/`：30G，已用 53M
     - 另有 `/autodl-pub` 和 `/autodl-pub/data`
     - 本次 `df -h` 输出中尚未直接看到 `/root/autodl-tmp` 行，需要下一步单独确认。
  5. 内存：
     - `free -h` 显示总内存 754Gi，可用 697Gi。
     - 这比购买页的 90GB 显示更大，可能是 AutoDL 容器/宿主显示口径差异；当前不是问题。
  6. 当前路径：
     - `/root`

  当前判断：
  1. SSH 登录成功。
  2. GPU 驱动可见，RTX 5090 可见。
  3. Python 和 conda 可用。
  4. 还需要确认 PyTorch 是否已经安装、是否能 `torch.cuda.is_available()`。
  5. 还需要确认 `/root/autodl-tmp` 数据盘路径是否存在，后续项目和数据应优先放在那里。

  下一步命令：
  ```bash
  ls -ld /root/autodl-tmp || true
  df -h /root/autodl-tmp || true
  python - <<'PY'
  import torch
  print("torch", torch.__version__)
  print("cuda available", torch.cuda.is_available())
  print("cuda version", torch.version.cuda)
  print("device count", torch.cuda.device_count())
  if torch.cuda.is_available():
      print("device name", torch.cuda.get_device_name(0))
  PY
  ```

  继续边界：
  1. 当前仍不上传项目。
  2. 当前仍不安装依赖。
  3. 当前仍不跑 smoke 或正式实验。


【本轮进度 30：数据盘确认成功，PyTorch 检查命令进入 heredoc 等待态】
  本轮输入：
  1. 用户执行了数据盘和 PyTorch 检查命令。
  2. 数据盘输出正常：
     - `/root/autodl-tmp` 存在；
     - 挂载点 `/dev/md0`；
     - 容量 50G；
     - 当前使用约 20K。
  3. 执行 Python heredoc 时终端出现 `>` 提示符，说明 shell 仍在等待 heredoc 结束。

  原因判断：
  1. 用户复制命令时 `PY` 结束标记前可能带了空格。
  2. Bash 的 heredoc 结束标记必须顶格输入，不能带空格。
  3. 这不是服务器错误，也不是配置错误。

  处理建议：
  1. 让用户按 `Ctrl+C` 退出当前 `>` 多行输入状态。
  2. 改用单行 Python 命令检查 PyTorch，避免 heredoc 对新手不友好：
     ```bash
     python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda version', torch.version.cuda); print('device count', torch.cuda.device_count()); print('device name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
     ```

  当前已确认：
  1. AutoDL 数据盘正常。
  2. 后续项目应优先放到 `/root/autodl-tmp/FedHDS-zhaoge`。


【本轮进度 31：PyTorch 单行检查因换行缩进失败，改用短命令】
  本轮输入：
  1. 用户执行单行 Python 检查命令时，命令被终端/复制过程拆成多行。
  2. 第二行前带空格，Python 报错：
     ```text
     IndentationError: unexpected indent
     ```

  判断：
  1. 这不是服务器环境错误。
  2. 这不是 PyTorch/CUDA 错误。
  3. 只是长命令复制后换行导致 Python 解析失败。

  处理方式：
  1. 改用多条短命令分别检查，降低复制出错概率：
     ```bash
     python -c "import torch; print(torch.__version__)"
     python -c "import torch; print(torch.cuda.is_available())"
     python -c "import torch; print(torch.version.cuda)"
     python -c "import torch; print(torch.cuda.device_count())"
     python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
     ```
  2. 每条命令独立执行，输出后再执行下一条。


【本轮进度 32：AutoDL PyTorch/CUDA 检查通过】
  本轮输入：
  1. 用户在 AutoDL 实例上分条执行 PyTorch/CUDA 检查命令。

  检查结果：
  1. PyTorch 版本：
     - `2.8.0+cu128`
  2. CUDA 可用：
     - `torch.cuda.is_available()` 返回 `True`
  3. PyTorch CUDA 版本：
     - `12.8`
  4. GPU 数量：
     - `1`
  5. GPU 名称：
     - `NVIDIA GeForce RTX 5090`

  当前判断：
  1. AutoDL 镜像可用。
  2. PyTorch 能正确识别 CUDA 和 RTX 5090。
  3. 云端基础环境已经满足进入项目上传和 smoke 准备阶段的条件。

  下一阶段建议：
  1. 将本地项目上传到 `/root/autodl-tmp/FedHDS-zhaoge`。
  2. 上传后先检查项目文件和数据文件是否存在。
  3. 再检查 Python 依赖，不直接跑正式实验。


【本轮进度 33：首次 scp 上传命令少空格导致失败】
  本轮输入：
  1. 用户在本地 PowerShell 执行 `scp` 上传核心文件。
  2. 命令末尾误写成：
     ```text
     F:\FedHDS-zhaoge\readme3.txtroot@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/
     ```
  3. 因为 `readme3.txt` 和远端地址之间缺少空格，scp 将其当成本地路径，报错：
     ```text
     No such file or directory
     ```

  判断：
  1. 这是本地命令拼写问题，不是服务器问题。
  2. 需要重新执行命令，确保最后一个本地文件与 `root@...` 之间有空格。

  建议：
  1. 为降低复制错误，改用 `Set-Location` 进入项目目录后再执行相对路径 scp：
     ```powershell
     Set-Location F:\FedHDS-zhaoge
     scp -P 56117 client.py server.py main.py m_utils.py evaluations.py requirement.txt README.md readme1.txt readme2.txt readme3.txt root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/
     ```
  2. 上传文件夹时也使用相对路径：
     ```powershell
     scp -P 56117 -r data scripts utils_data root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/
     ```


【本轮进度 34：核心代码文件上传成功】
  本轮输入：
  1. 用户重新执行修正后的 `scp` 核心文件上传命令。
  2. 上传成功的文件包括：
     - `client.py`
     - `server.py`
     - `main.py`
     - `m_utils.py`
     - `evaluations.py`
     - `requirement.txt`
     - `README.md`
     - `readme1.txt`
     - `readme2.txt`
     - `readme3.txt`

  当前判断：
  1. 核心 Python 代码和 README 已成功上传到云端 `/root/autodl-tmp/FedHDS-zhaoge/`。
  2. 还需要继续上传运行所需目录：
     - `data`
     - `scripts`
     - `utils_data`

  下一步：
  1. 在本地 PowerShell 中继续执行：
     ```powershell
     scp -P 56117 -r F:\FedHDS-zhaoge\data F:\FedHDS-zhaoge\scripts F:\FedHDS-zhaoge\utils_data root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/
     ```
  2. 上传目录后，在云端执行文件检查。


【本轮进度 35：目录上传命令漏掉远端目标，误复制到本地 utils_data】
  本轮输入：
  1. 用户在本地 PowerShell 执行目录上传命令时，实际输入为：
     ```powershell
     scp -P 56117 -r F:\FedHDS-zhaoge\data F:\FedHDS-zhaoge\scripts F:\FedHDS-zhaoge\utils_data
     ```
  2. 命令末尾缺少远端目标：
     ```text
     root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/
     ```
  3. 因此 `scp` 将最后一个参数 `F:\FedHDS-zhaoge\utils_data` 当成本地目标目录，把 `data` 和 `scripts` 复制到了本地 `utils_data` 下。

  判断：
  1. 本次目录没有上传到云服务器。
  2. 服务器环境没有被破坏。
  3. 本地 `utils_data\data` 和 `utils_data\scripts` 属于误复制副本，暂时不影响云端流程；后续可以再清理。

  纠正方式：
  1. 为避免路径太长和再次漏掉远端目标，建议先进入本地项目目录：
     ```powershell
     Set-Location F:\FedHDS-zhaoge
     ```
  2. 然后执行短命令：
     ```powershell
     scp -P 56117 -r data scripts utils_data root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/
     ```
  3. 注意 `utils_data` 和 `root@...` 之间必须有空格；`root@...:/root/...` 必须在同一条命令末尾，不要单独另起一行。


【本轮进度 36：PowerShell 当前目录错误，改为逐目录绝对路径上传】
  本轮输入：
  1. 用户误将远端地址和 `Set-Location` 拼在同一行，PowerShell 报 `CommandNotFoundException`。
  2. 随后用户在 `C:\Users\Fred` 下执行：
     ```powershell
     scp -P 56117 -r data scripts utils_data root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/
     ```
  3. 因当前目录不是 `F:\FedHDS-zhaoge`，报错：
     ```text
     stat local "data": No such file or directory
     ```

  判断：
  1. 本次仍未上传目录到云端。
  2. 服务器端没有受影响。
  3. 对新手而言，先 `Set-Location` 再 scp 容易因复制残留出错。

  调整后的上传策略：
  1. 不再依赖当前目录。
  2. 使用三条独立的绝对路径 `scp` 命令分别上传：
     ```powershell
     scp -P 56117 -r "F:\FedHDS-zhaoge\data" "root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/"
     scp -P 56117 -r "F:\FedHDS-zhaoge\scripts" "root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/"
     scp -P 56117 -r "F:\FedHDS-zhaoge\utils_data" "root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/"
     ```
  3. 每条命令单独复制、单独执行，避免一条长命令中漏掉远端目标。


【本轮进度 37：data/scripts/utils_data 目录上传成功】
  本轮输入：
  1. 用户按三条独立 `scp` 命令上传：
     - `data`
     - `scripts`
     - `utils_data`
  2. 输出显示 `databricks-dolly-15k.jsonl`、脚本文件和 `utils_data` 代码文件均已传输完成。

  当前判断：
  1. 云端 `/root/autodl-tmp/FedHDS-zhaoge/` 已具备核心代码、数据、脚本和数据工具目录。
  2. 由于此前本地误复制，`utils_data` 中可能包含额外的 `data/`、`scripts/` 和 `__pycache__/` 副本。
  3. 这些额外副本一般不影响运行；暂时不做清理，避免新手阶段引入误删风险。

  下一步：
  1. 回到服务器 SSH 窗口。
  2. 检查目录结构和关键文件。
  3. 先做语法编译检查，再检查 Python 依赖。
  4. 仍不直接跑 smoke 或正式实验。


【本轮进度 38：云端项目结构确认与语法检查通过】
  本轮输入：
  1. 用户在云端执行：
     ```bash
     cd /root/autodl-tmp/FedHDS-zhaoge
     pwd
     ls
     ls data
     ls scripts
     ls utils_data
     ```
  2. 输出确认当前路径：
     ```bash
     /root/autodl-tmp/FedHDS-zhaoge
     ```
  3. 项目根目录包含：
     - `client.py`
     - `server.py`
     - `main.py`
     - `m_utils.py`
     - `evaluations.py`
     - `requirement.txt`
     - `data`
     - `scripts`
     - `utils_data`
  4. `data` 中包含：
     - `databricks-dolly-15k.jsonl`
  5. `scripts` 中包含：
     - `run_cloud_experiments.sh`
     - `summarize_experiments.py`
     - `make_paper_table.py`
     - `plot_experiments.py`
  6. `utils_data` 中包含核心数据工具文件，同时也有此前本地误复制带来的 `data/`、`scripts/` 副本。

  语法检查：
  1. 用户第一次执行 `py_compile` 时因换行把 `scripts/` 当作文件，并将 `plot_experiments.py` 单独作为 shell 命令执行，出现小错误。
  2. 随后用户正确执行：
     ```bash
     python -m py_compile main.py server.py client.py scripts/summarize_experiments.py scripts/make_paper_table.py scripts/plot_experiments.py
     ```
  3. 该命令无输出，表示语法检查通过。

  当前判断：
  1. 云端项目上传成功。
  2. 核心代码和脚本语法检查通过。
  3. 下一步应检查依赖包是否齐全，仍不直接跑 smoke。

  下一步建议命令：
  ```bash
  python -c "import transformers; print('transformers', transformers.__version__)"
  python -c "import peft; print('peft', peft.__version__)"
  python -c "import sklearn; print('sklearn', sklearn.__version__)"
  python -c "import hdbscan; print('hdbscan ok')"
  python -c "import pandas, matplotlib; print('pandas/matplotlib ok')"
  ```


【本轮进度 39：云端依赖检查缺少项目包】
  本轮输入：
  1. 用户在云端执行依赖检查命令。
  2. 当前环境缺少：
     - `transformers`
     - `peft`
     - `sklearn`
     - `hdbscan`
     - `pandas`

  当前判断：
  1. AutoDL 基础镜像只保证 PyTorch/CUDA 可用，不包含本项目所需全部 Python 包。
  2. 这是正常现象，不是上传问题。
  3. 当前 PyTorch 2.8.0+cu128 已经可用，依赖安装时应避免覆盖已有 torch/torchvision/torchaudio。

  安装策略：
  1. 先升级 pip 基础工具。
  2. 安装项目所需非 torch 包。
  3. 暂不执行 `pip install -r requirement.txt`，因为其中固定了 torch/torchaudio/torchvision 等包，可能覆盖当前已验证可用的 CUDA PyTorch。

  建议安装命令：
  ```bash
  python -m pip install -U pip setuptools wheel
  python -m pip install accelerate transformers peft bitsandbytes sentencepiece hdbscan scikit-learn scipy pandas matplotlib seaborn rouge tqdm requests pyyaml psutil
  ```

  安装后重新检查：
  ```bash
  python -c "import transformers; print('transformers', transformers.__version__)"
  python -c "import peft; print('peft', peft.__version__)"
  python -c "import sklearn; print('sklearn', sklearn.__version__)"
  python -c "import hdbscan; print('hdbscan ok')"
  python -c "import pandas, matplotlib; print('pandas/matplotlib ok')"
  python -c "import bitsandbytes as bnb; print('bitsandbytes ok')"
  ```


【本轮进度 40：云端依赖安装完成】
  本轮输入：
  1. 用户执行：
     ```bash
     python -m pip install -U pip setuptools wheel
     python -m pip install accelerate transformers peft bitsandbytes sentencepiece hdbscan scikit-learn scipy pandas matplotlib seaborn rouge tqdm requests pyyaml psutil
     ```
  2. 安装过程使用阿里云 PyPI 镜像。

  安装后检查结果：
  1. `transformers 5.7.0`
  2. `peft 0.19.1`
  3. `sklearn 1.8.0`
  4. `hdbscan ok`
  5. `pandas/matplotlib ok`
  6. `bitsandbytes ok`

  当前判断：
  1. 项目运行所需主要依赖已安装。
  2. 当前仍未运行训练或 smoke。
  3. 下一步应先做项目级 import 检查，以及 Hugging Face/Qwen2-0.5B 访问检查。

  下一步建议：
  1. 检查 transformers/peft/bitsandbytes 项目级导入：
     ```bash
     python -c "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig; from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training; print('model imports ok')"
     ```
  2. 检查 Qwen2-0.5B tokenizer 是否可下载/加载：
     ```bash
     python -c "from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B', use_fast=True); print('tokenizer ok', tok.__class__.__name__)"
     ```
  3. tokenizer 检查通过后，再进入 cloud smoke。


【本轮进度 41：模型依赖导入通过，Qwen tokenizer 下载疑似卡住】
  本轮输入：
  1. 用户在云端执行模型相关 import 检查：
     ```bash
     python -c "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig; from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training; print('model imports ok')"
     ```
  2. 输出：
     ```text
     model imports ok
     ```
  3. 用户随后执行 Qwen2-0.5B tokenizer 加载命令：
     ```bash
     python -c "from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B',use_fast=True); print('tokenizer ok', tok.__class__.__name__)"
     ```
  4. 该命令长时间无输出。

  当前判断：
  1. 项目级模型依赖导入已经通过。
  2. tokenizer 加载长时间无输出，最可能原因是 Hugging Face 网络访问慢、连接等待或下载无进度。
  3. 这不是用户命令输入错误。
  4. 需要先中断当前命令，再测试 Hugging Face 网络；若网络不稳定，使用镜像端点或改用本地缓存策略。

  建议处理：
  1. 如果等待超过 3-5 分钟仍无输出，按 `Ctrl+C` 中断。
  2. 先执行网络探测：
     ```bash
     python -c "import requests; r=requests.get('https://huggingface.co', timeout=10); print(r.status_code)"
     ```
  3. 如 Hugging Face 访问失败或超时，尝试使用镜像端点：
     ```bash
     export HF_ENDPOINT=https://hf-mirror.com
     export HF_HUB_DISABLE_XET=1
     ```
  4. 然后重新加载 tokenizer。


【本轮进度 42：Hugging Face 直连失败，hf-mirror tokenizer 加载成功】
  本轮输入：
  1. 用户执行 Hugging Face 直连测试：
     ```bash
     python -c "import requests; r=requests.get('https://huggingface.co', timeout=10); print(r.status_code)"
     ```
  2. 结果报错：
     ```text
     OSError: [Errno 101] Network is unreachable
     ```
  3. 用户随后执行：
     ```bash
     export HF_ENDPOINT=https://hf-mirror.com
     export HF_HUB_DISABLE_XET=1
     ```
  4. 再执行 Qwen2-0.5B tokenizer 加载后成功下载并输出：
     ```text
     tokenizer ok Qwen2Tokenizer
     ```

  当前判断：
  1. AutoDL 实例无法直连 `huggingface.co`。
  2. `hf-mirror.com` 可用。
  3. tokenizer 已经成功下载到 Hugging Face 缓存。
  4. 后续所有模型下载、smoke 和正式运行都需要保留：
     - `HF_ENDPOINT=https://hf-mirror.com`
     - `HF_HUB_DISABLE_XET=1`

  下一步：
  1. 先下载/验证 Qwen2-0.5B 模型能加载，不直接跑训练。
  2. 建议执行轻量模型加载检查：
     ```bash
     python -c "from transformers import AutoModelForCausalLM; import torch; m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', torch_dtype=torch.float16, trust_remote_code=True); print('model ok', type(m).__name__)"
     ```
  3. 如果模型下载和加载成功，再进入 `RUN_SMOKE=1` 云端 smoke。


【本轮进度 43：Qwen2-0.5B 模型下载速度过慢，改用加速与数据盘缓存策略】
  本轮输入：
  1. 用户执行 Qwen2-0.5B 模型加载检查：
     ```bash
     python -c "from transformers import AutoModelForCausalLM; import torch; m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', torch_dtype=torch.float16, trust_remote_code=True); print('model ok', type(m).__name__)"
     ```
  2. 输出显示：
     - `torch_dtype` deprecated warning；
     - `model.safetensors` 约 988MB；
     - 当前下载进度约 10.5MB / 988MB；
     - 速度约 193kB/s；
     - 预计剩余约 1小时24分钟。

  当前判断：
  1. 命令本身没有错。
  2. 下载速度过慢，不建议直接等待。
  3. 后续应使用 AutoDL 学术资源加速、HF mirror 和数据盘缓存/本地模型目录。
  4. 模型缓存不应继续默认写到系统盘 `/root/.cache`，应迁移到 `/root/autodl-tmp/cache` 或显式下载到 `/root/autodl-tmp/models/Qwen2-0.5B`。

  建议操作：
  1. 中断当前下载：
     ```text
     Ctrl+C
     ```
  2. 启用 AutoDL 学术资源加速和 HF mirror：
     ```bash
     source /etc/network_turbo
     export HF_ENDPOINT=https://hf-mirror.com
     export HF_HUB_DISABLE_XET=1
     export HF_HOME=/root/autodl-tmp/cache
     mkdir -p /root/autodl-tmp/cache /root/autodl-tmp/models
     ```
  3. 用 `huggingface-cli` 下载到数据盘本地模型目录：
     ```bash
     huggingface-cli download Qwen/Qwen2-0.5B --local-dir /root/autodl-tmp/models/Qwen2-0.5B
     ```
  4. 下载完成后使用本地模型路径验证加载：
     ```bash
     python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; import torch; p='/root/autodl-tmp/models/Qwen2-0.5B'; tok=AutoTokenizer.from_pretrained(p,use_fast=True); m=AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, trust_remote_code=True); print('local model ok', type(m).__name__, tok.__class__.__name__)"
     ```

  后续 smoke 注意：
  1. smoke 运行时建议设置：
     - `MODEL=/root/autodl-tmp/models/Qwen2-0.5B`
     - `HF_HOME=/root/autodl-tmp/cache`
  2. 这样可以避免正式运行时重复从网络下载模型。


【本轮进度 44：huggingface-cli 已废弃，需改用 hf download】
  本轮输入：
  1. 用户执行：
     ```bash
     source /etc/network_turbo
     export HF_ENDPOINT=https://hf-mirror.com
     export HF_HUB_DISABLE_XET=1
     export HF_HOME=/root/autodl-tmp/cache
     mkdir -p /root/autodl-tmp/cache /root/autodl-tmp/models
     ```
  2. AutoDL 输出“设置成功”，学术加速已启用。
  3. 用户执行：
     ```bash
     huggingface-cli download Qwen/Qwen2-0.5B --local-dir /root/autodl-tmp/models/Qwen2-0.5B
     ```
  4. 输出提示：
     ```text
     Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead.
     ```
  5. 因此模型实际没有下载。
  6. 用户随后尝试从 `/root/autodl-tmp/models/Qwen2-0.5B` 加载模型，报 `HFValidationError`。

  当前判断：
  1. 问题不是模型本身，也不是 transformers import。
  2. 原因是新版本 Hugging Face CLI 已废弃 `huggingface-cli download`，必须改用 `hf download`。
  3. 本地目录 `/root/autodl-tmp/models/Qwen2-0.5B` 尚未包含有效模型文件，transformers 因此没有按本地模型目录处理。

  纠正命令：
  ```bash
  hf download Qwen/Qwen2-0.5B --local-dir /root/autodl-tmp/models/Qwen2-0.5B
  ```

  下载完成后检查：
  ```bash
  ls -lh /root/autodl-tmp/models/Qwen2-0.5B
  ```

  再进行本地模型加载检查：
  ```bash
  python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; import torch; p='/root/autodl-tmp/models/Qwen2-0.5B'; tok=AutoTokenizer.from_pretrained(p,use_fast=True); m=AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, trust_remote_code=True); print('local model ok', type(m).__name__, tok.__class__.__name__)"
  ```


【本轮进度 45：hf download 仍然过慢，准备改试 ModelScope 国内源】
  本轮输入：
  1. 用户先误执行本地模型加载检查，因本地模型目录尚无完整模型文件，继续报 `HFValidationError`。
  2. 用户随后执行：
     ```bash
     hf download Qwen/Qwen2-0.5B --local-dir /root/autodl-tmp/models/Qwen2-0.5B
     ```
  3. 下载启动后显示：
     - `Downloading (incomplete total...)`
     - `211M/988M`
     - 速度约 `167kB/s`
     - 预计剩余约 `1:17:32`

  当前判断：
  1. `hf download` 命令本身已正确。
  2. 但通过 hf-mirror 下载大文件仍然过慢。
  3. 不建议在前台硬等一小时以上；优先尝试 ModelScope 国内源。

  建议操作：
  1. 按 `Ctrl+C` 中断当前 `hf download`。
  2. 安装 ModelScope：
     ```bash
     python -m pip install -U modelscope
     ```
  3. 使用独立目录下载，避免与未完成的 HF 目录混合：
     ```bash
     mkdir -p /root/autodl-tmp/models/Qwen2-0.5B-ms
     modelscope download --model Qwen/Qwen2-0.5B --local_dir /root/autodl-tmp/models/Qwen2-0.5B-ms
     ```
  4. 如果 `Qwen/Qwen2-0.5B` 在 ModelScope 上找不到，可再试：
     ```bash
     modelscope download --model qwen/Qwen2-0.5B --local_dir /root/autodl-tmp/models/Qwen2-0.5B-ms
     ```
  5. 下载完成后用 `/root/autodl-tmp/models/Qwen2-0.5B-ms` 作为本地模型路径进行加载检查。

  备选：
  1. 如果 ModelScope 也失败，再考虑使用 `aria2c` 多线程直接下载 `hf-mirror` 的大文件。
  2. 如果用户不急，继续等待 `hf download` 完成也可以，但预计会额外消耗约 1 小时以上租用时间。


【本轮进度 46：ModelScope 下载 Qwen2-0.5B 基本完成】
  本轮输入：
  1. 用户第一次误将 `mkdir` 和 `modelscope download` 写在同一行，导致：
     ```text
     mkdir: unrecognized option '--model'
     ```
  2. 随后用户正确执行：
     ```bash
     modelscope download --model qwen/Qwen2-0.5B --local_dir /root/autodl-tmp/models/Qwen2-0.5B-ms
     ```
  3. ModelScope 输出显示下载目录：
     ```text
     /root/autodl-tmp/models/Qwen2-0.5B-ms
     ```
  4. 关键文件下载情况：
     - `config.json`
     - `generation_config.json`
     - `tokenizer_config.json`
     - `vocab.json`
     - `merges.txt`
     - `tokenizer.json`
     - `model.safetensors`，约 942M
  5. 输出显示：
     ```text
     Successfully Downloaded from model qwen/Qwen2-0.5B.
     ```

  当前判断：
  1. ModelScope 下载速度明显快于 hf-mirror，约 1分14秒完成 942M 权重。
  2. 终端后续出现的进度条可能是下载器残留输出或并发进度刷新，不一定代表仍在下载。
  3. 下一步应检查文件是否存在和大小，而不是继续等待。

  建议检查：
  ```bash
  ls -lh /root/autodl-tmp/models/Qwen2-0.5B-ms
  du -sh /root/autodl-tmp/models/Qwen2-0.5B-ms
  ```

  如果提示符没有返回：
  1. 等 1-2 分钟仍无变化，可按 `Ctrl+C`。
  2. 然后执行上述 `ls` 和 `du` 检查。

  下载确认后：
  1. 使用本地 ModelScope 目录作为模型路径：
     - `/root/autodl-tmp/models/Qwen2-0.5B-ms`
  2. 后续 smoke 命令中的 `MODEL` 应改为该本地路径。


【本轮进度 47：ModelScope 本地模型文件确认完整】
  本轮输入：
  1. 用户执行模型目录检查。
  2. `/root/autodl-tmp/models/Qwen2-0.5B-ms` 中关键文件存在：
     - `config.json`
     - `generation_config.json`
     - `tokenizer_config.json`
     - `vocab.json`
     - `merges.txt`
     - `tokenizer.json`
     - `model.safetensors`
  3. `model.safetensors` 大小：
     - `943M`
  4. 目录总大小：
     - `954M`

  当前判断：
  1. Qwen2-0.5B 已经通过 ModelScope 成功下载到数据盘。
  2. 后续不应再从 `Qwen/Qwen2-0.5B` 网络路径加载，而应使用本地路径：
     ```text
     /root/autodl-tmp/models/Qwen2-0.5B-ms
     ```
  3. 下一步应验证 transformers 从该本地路径加载 tokenizer 和模型。

  下一步命令：
  ```bash
  python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; import torch; p='/root/autodl-tmp/models/Qwen2-0.5B-ms'; tok=AutoTokenizer.from_pretrained(p,use_fast=True); m=AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, trust_remote_code=True); print('local model ok', type(m).__name__, tok.__class__.__name__)"
  ```


【本轮进度 48：本地 Qwen2-0.5B 模型加载验证通过】
  本轮输入：
  1. 用户执行本地模型加载检查：
     ```bash
     python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; import torch; p='/root/autodl-tmp/models/Qwen2-0.5B-ms'; tok=AutoTokenizer.from_pretrained(p,use_fast=True); m=AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, trust_remote_code=True); print('local model ok', type(m).__name__, tok.__class__.__name__)"
     ```
  2. 输出：
     - `Loading weights: 100%`
     - `local model ok Qwen2ForCausalLM Qwen2Tokenizer`

  当前判断：
  1. Qwen2-0.5B 本地模型目录可被 transformers 正常加载。
  2. 云端已经具备运行最小 smoke 的条件：
     - GPU 可用；
     - PyTorch/CUDA 可用；
     - 项目文件上传完成；
     - 依赖安装完成；
     - 数据文件存在；
     - 本地模型存在且可加载。

  下一步：
  1. 运行 cloud smoke：
     ```bash
     RUN_SMOKE=1 \
     PYTHON_BIN=python \
     MODEL=/root/autodl-tmp/models/Qwen2-0.5B-ms \
     DATA_PATH=/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl \
     DEVICE=0 \
     RESULT_ROOT=formal_cloud_results \
     bash scripts/run_cloud_experiments.sh
     ```
  2. smoke 目的只是验证云端完整链路，不作为论文正式结果。
  3. smoke 成功后检查：
     - `formal_cloud_results/smoke/summary.csv`
     - `formal_cloud_results/smoke/paper_table.md`
     - `formal_cloud_results/smoke/figures/`


【本轮进度 49：AutoDL cloud smoke 完整跑通】
  本轮输入：
  1. 用户在 AutoDL 云端执行 cloud smoke：
     ```bash
     RUN_SMOKE=1 \
     PYTHON_BIN=python \
     MODEL=/root/autodl-tmp/models/Qwen2-0.5B-ms \
     DATA_PATH=/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl \
     DEVICE=0 \
     RESULT_ROOT=formal_cloud_results \
     bash scripts/run_cloud_experiments.sh
     ```
  2. smoke 运行完成。

  smoke 配置：
  1. 模型：本地 ModelScope 下载的 Qwen2-0.5B：
     `/root/autodl-tmp/models/Qwen2-0.5B-ms`
  2. 数据：Dolly：
     `/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl`
  3. RUN_SMOKE=1，脚本自动使用小规模配置：
     - 4 clients
     - 1 round
     - local_step=1
     - data_sample=0.02
     - unlearn_grad_sample_size=2
     - hessian_ref_per_client=2

  smoke 四组结果状态：
  1. `fl`：
     - 成功完成；
     - round 1 global loss: 10.0000；
     - 已保存 `final_results.json`。
  2. `fedhds`：
     - 成功完成；
     - round 1 global loss: 10.0000；
     - 已保存 `final_results.json`。
  3. `forget_batch_hessian`：
     - 成功完成；
     - forget client loss: 2.6189 -> 17.2269；
     - post-unlearning global loss: 10.0000；
     - unlearning time: 0.74s；
     - actual parameter delta L2: 36.38225；
     - 已保存 `final_results.json`。
  4. `retained_guard`：
     - 成功完成；
     - retained reference set size: 6；
     - before clipping update L2: 2.136734；
     - after clipping update L2 target: 0.3；
     - global loss guard max: 0.8；
     - step 1/5 rejected by global loss guard；
     - accepted=0/1；
     - actual parameter delta L2: 0.0；
     - forget client loss: 2.6189 -> 2.6189；
     - post-unlearning global loss: 10.0000；
     - unlearning time: 4.19s；
     - 已保存 `final_results.json`。

  smoke 输出文件：
  1. `formal_cloud_results/smoke/summary.csv`
  2. `formal_cloud_results/smoke/paper_table.csv`
  3. `formal_cloud_results/smoke/paper_table.md`
  4. `formal_cloud_results/smoke/figures/` 下生成 12 个 png/pdf 图：
     - `forget_loss_before_after`
     - `global_loss_final`
     - `update_norm_vs_global_loss`
     - `update_norm_vs_forget_gain`
     - `guard_steps`
     - `unlearning_time`

  当前判断：
  1. 云端项目环境、数据读取、模型加载、FL、FedHDS、forget-batch unlearning、retained guard、结果汇总、论文表格和出图全链路均已跑通。
  2. 该 smoke 结果只用于云端链路验证，不作为论文正式结果。
  3. smoke 中 global loss 为 10.0000，说明配置极小且训练不足，不能用于方法优劣判断。
  4. retained guard 在 smoke 中因 global loss guard 被第 1 步拒绝，这是 guard 机制可运行的证据，不是正式效果结论。

  下一步建议：
  1. 先把 `formal_cloud_results/smoke` 下载回本地保存。
  2. 如果用户没有连续数小时空闲时间，应先暂停并关机止损。
  3. 如要跑正式最小四组实验，需确认用户有较长空闲时间，并建议使用 `screen` 后台运行。


【本轮进度 50：cloud smoke 已下载回本地，准备进入第一阶段正式最小四组实验】
  本轮输入：
  1. 用户已将云端 smoke 结果下载回本地：
     ```powershell
     scp -P 56117 -r "root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results/smoke" "F:\FedHDS-zhaoge\formal_cloud_results_smoke"
     ```
  2. 下载内容包括：
     - 四组日志：`fl.log`、`fedhds.log`、`forget_batch_hessian.log`、`retained_guard.log`
     - 四组 `final_results.json`
     - `summary.csv`
     - `paper_table.csv`
     - `paper_table.md`
     - `figures/*.png`
     - `figures/*.pdf`
  3. 用户说明未来两天时间充裕，云服务器已充值，可以运行约 24 小时。

  当前判断：
  1. 可以进入第一阶段正式云端最小四组实验。
  2. 仍然不做大量参数 sweep。
  3. 正式第一阶段只跑 `scripts/run_cloud_experiments.sh` 默认非 smoke 四组：
     - `fl`
     - `fedhds`
     - `forget_batch_hessian`
     - `retained_guard`
  4. 推荐使用 `screen` 后台运行，避免 SSH 断开导致实验中断。

  第一阶段正式云端配置：
  1. 模型：`/root/autodl-tmp/models/Qwen2-0.5B-ms`
  2. 数据：`/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl`
  3. 脚本默认非 smoke 参数：
     - `NUM_CLIENTS=20`
     - `CLIENT_FRACTION=0.2`
     - `ROUNDS=10`
     - `LOCAL_STEP=2`
     - `BATCH_SIZE=1`
     - `MAX_LENGTH=64`
     - `DATA_SAMPLE=0.2`
     - `IID=0`
     - `FORGET_CLIENT_IDX=0`
     - `HESSIAN_REF_PER_CLIENT=4`
     - `UNLEARN_GRAD_SAMPLE_SIZE=8`
     - `LISSA_DEPTH=1`
     - `LISSA_DAMPING=0.01`
     - retained guard 使用 `max_norm=0.3, steps=5, global_guard=0.8`

  建议正式运行命令：
  ```bash
  cd /root/autodl-tmp/FedHDS-zhaoge
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_DISABLE_XET=1
  export HF_HOME=/root/autodl-tmp/cache
  RUN_SMOKE=0 \
  PYTHON_BIN=python \
  MODEL=/root/autodl-tmp/models/Qwen2-0.5B-ms \
  DATA_PATH=/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl \
  DEVICE=0 \
  RESULT_ROOT=formal_cloud_results \
  bash scripts/run_cloud_experiments.sh
  ```

  运行管理建议：
  1. 使用 `screen -S fedhds_formal` 创建后台会话。
  2. 在 screen 中运行正式命令。
  3. 需要离开时按 `Ctrl+A`，再按 `D` 分离会话。
  4. 之后用 `screen -r fedhds_formal` 回到实验窗口。
  5. 正式实验结束后，下载 `formal_cloud_results/summary.csv`、`paper_table.md`、`figures/` 和全部 `final_results.json`。


【本轮进度 51：正式实验目录需与 smoke 隔离】
  本轮补充判断：
  1. `scripts/summarize_experiments.py` 会递归扫描指定结果根目录下的 `final_results.json`。
  2. 如果正式运行继续使用 `RESULT_ROOT=formal_cloud_results`，可能把已存在的 `formal_cloud_results/smoke` 结果也汇总进正式 `summary.csv`。
  3. 因此正式第一阶段结果必须使用独立子目录。

  推荐正式目录：
  ```text
  formal_cloud_results/formal_minimal_20260503
  ```

  推荐后台运行方式：
  1. 创建 screen 会话：
     ```bash
     screen -S fedhds_formal
     ```
  2. 在 screen 中运行：
     ```bash
     cd /root/autodl-tmp/FedHDS-zhaoge
     export HF_ENDPOINT=https://hf-mirror.com
     export HF_HUB_DISABLE_XET=1
     export HF_HOME=/root/autodl-tmp/cache
     export PYTHONUNBUFFERED=1
     RUN_SMOKE=0 \
     PYTHON_BIN=python \
     MODEL=/root/autodl-tmp/models/Qwen2-0.5B-ms \
     DATA_PATH=/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl \
     DEVICE=0 \
     RESULT_ROOT=formal_cloud_results/formal_minimal_20260503 \
     bash scripts/run_cloud_experiments.sh 2>&1 | tee formal_cloud_results/formal_minimal_20260503/top_level_run.log
     ```
  3. 若需要离开 SSH：按 `Ctrl+A`，再按 `D`。
  4. 查看会话：
     ```bash
     screen -ls
     ```
  5. 回到会话：
     ```bash
     screen -r fedhds_formal
     ```

  正式实验边界：
  1. 本轮只跑四组最小正式实验，不做 sweep。
  2. 不换 1.5B/3B/7B。
  3. 不改代码。
  4. 跑完后先下载正式目录，再分析结果表。


【本轮进度 52：screen 启动后终端欢迎信息乱码】
  本轮输入：
  1. 用户执行正式实验前进入 `screen` 后，终端出现 AutoDL 欢迎信息乱码。
  2. 乱码内容包含 AutoDL 目录说明、系统盘/数据盘提示等。

  当前判断：
  1. 这通常是终端编码、screen 刷屏或中文欢迎信息显示问题。
  2. 不代表实验已经出错。
  3. 如果仍能输入命令，直接执行 `clear` 清屏即可。

  建议操作：
  1. 在当前 screen 窗口按回车。
  2. 输入：
     ```bash
     clear
     ```
  3. 如果能看到正常提示符，再执行：
     ```bash
     cd /root/autodl-tmp/FedHDS-zhaoge
     pwd
     ```
  4. 若提示符正常，再继续正式实验命令。

  如果 screen 内完全无法正常显示：
  1. 先按 `Ctrl+A`，再按 `D` 分离。
  2. 回到普通 SSH 后执行：
     ```bash
     screen -ls
     ```
  3. 再决定是否重连或新建 screen。


【本轮进度 53：screen 内提示符恢复正常，正式实验可启动】
  本轮输入：
  1. 用户在 screen 中执行：
     ```bash
     cd /root/autodl-tmp/FedHDS-zhaoge
     pwd
     ```
  2. 输出：
     ```bash
     /root/autodl-tmp/FedHDS-zhaoge
     ```
  3. 用户说明字体变成红色。

  当前判断：
  1. screen 会话可正常输入命令。
  2. 当前路径正确。
  3. 字体变红通常是 root 提示符/终端主题显示，不影响实验。

  下一步：
  1. 可以继续在当前 screen 中执行正式四组实验命令。
  2. 正式目录继续使用：
     ```text
     formal_cloud_results/formal_minimal_20260503
     ```


【本轮进度 54：正式实验运行中，screen/PowerShell 显示不舒服但实验正常】
  本轮输入：
  1. 用户将当前终端界面截图放入 `图库.docx`。
  2. 截图显示正式实验正在运行，输出包含：
     - `Evaluating Round 3 Loss: 1.7392`
     - `Start Training Round 4`
     - `Extracting features on cuda:0`
     - 多个 tqdm 进度条
  3. 用户反馈：
     - 界面看着不舒服；
     - 不能上下滑动。

  当前判断：
  1. 正式实验正在正常运行，并非卡死或报错。
  2. 红色文字通常是终端主题/root/screen 显示效果，不影响程序。
  3. `����` 等乱码来自 tqdm 进度条特殊字符在 Windows PowerShell/screen 组合中显示异常。
  4. screen 默认不使用普通终端滚轮历史，因此用户感觉不能上下滑动是正常现象。

  建议操作：
  1. 不要中断实验。
  2. 如果只是想离开难看的界面，按 `Ctrl+A` 后按 `D` 分离 screen，让实验后台继续运行。
  3. 回到普通 SSH 后可用日志文件查看进度：
     ```bash
     tail -n 40 /root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results/formal_minimal_20260503/logs/fl.log
     ```
     或根据当前组查看：
     ```bash
     tail -n 40 /root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results/formal_minimal_20260503/logs/fedhds.log
     tail -n 40 /root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results/formal_minimal_20260503/top_level_run.log
     ```
  4. 如需回到实验画面：
     ```bash
     screen -r fedhds_formal
     ```

  screen 滚动说明：
  1. 在 screen 内查看历史可按 `Ctrl+A` 后按 `[` 进入 copy/scrollback 模式。
  2. 进入后可用方向键、PageUp/PageDown 滚动。
  3. 按 `q` 或 `Esc` 退出滚动模式。
  4. 对新手更推荐直接分离 screen，用 `tail` 看日志。


【本轮进度 55：正式四组实验第 4 组 retained guard 因 CUDA OOM 中断】
  本轮输入：
  1. 用户反馈正式运行结束并贴出错误日志。
  2. 日志显示当前已经进入第 4 组 `retained_guard`：
     - Hessian reference mode: retained-set
     - Retained samples per client: 4
     - Forget gradient sample size: 8
     - Forget gradient norm: L2=6.247183e+00
     - LiSSA depth=1, damping=0.01
  3. 在 `server.py` 的 `compute_hvp(...)` 中报错：
     ```text
     torch.OutOfMemoryError: CUDA out of memory.
     Tried to allocate 20.00 MiB.
     GPU total capacity: 31.36 GiB
     only 5.06 MiB free
     process has 31.34 GiB memory in use
     allocated by PyTorch: 30.69 GiB
     ```

  当前判断：
  1. 第一阶段正式四组实验没有全部成功完成。
  2. 前三组 `fl`、`fedhds`、`forget_batch_hessian` 大概率已完成并保存 `final_results.json`。
  3. 第四组 `retained_guard` 在 retained-set HVP 阶段 OOM 中断。
  4. 这不是 screen/终端显示问题，而是正式配置下 retained reference set 过大导致二阶 autograd 显存峰值超过 RTX 5090 32GB。
  5. 当前不应直接重复跑原正式脚本，否则仍会在第 4 组 OOM。

  终端恢复建议：
  1. 在当前终端执行：
     ```bash
     printf '\033[0m'
     clear
     ```
  2. 如果还在 screen 中且想回普通可滚动终端：
     - 按 `Ctrl+A`，再按 `D` 分离 screen；
     - 或输入 `exit` 退出当前 screen shell。
  3. 回到普通 SSH 后，Windows PowerShell 的滚轮通常可以正常滚动。

  结果处理建议：
  1. 先对当前正式目录生成部分汇总，确认前三组结果是否完整：
     ```bash
     cd /root/autodl-tmp/FedHDS-zhaoge
     python scripts/summarize_experiments.py formal_cloud_results/formal_minimal_20260503
     python scripts/make_paper_table.py formal_cloud_results/formal_minimal_20260503/summary.csv
     ```
  2. 查看部分汇总：
     ```bash
     cat formal_cloud_results/formal_minimal_20260503/summary.csv
     ```

  后续补救方向：
  1. 不建议重跑前三组。
  2. 建议单独补跑一个更省显存的 retained guard：
     - 降低 `hessian_ref_per_client`，例如从 4 降到 1；
     - 保留 `unlearn_grad_sample_size=8` 或必要时降到 4；
     - 保留 `max_norm=0.3, steps=5, global_guard=0.8`；
     - 单独运行 retained case，写入新目录 `retained_guard_ref1`，避免覆盖已有结果。
  3. 该补跑属于 OOM 修复，不是参数 sweep。


【本轮进度 56：正式前三组结果汇总完成，准备单独补跑 retained guard】
  本轮输入：
  1. 用户执行：
     ```bash
     python scripts/summarize_experiments.py formal_cloud_results/formal_minimal_20260503
     python scripts/make_paper_table.py formal_cloud_results/formal_minimal_20260503/summary.csv
     cat formal_cloud_results/formal_minimal_20260503/summary.csv
     ```
  2. 汇总脚本输出 3 行，说明前三组正式结果完整，第 4 组 retained_guard 未生成 final_results。

  当前前三组正式结果：
  1. `fl`：
     - round2_global_loss: 1.783530279994011
     - final_global_loss: 1.6442784418662388
  2. `fedhds`：
     - round2_global_loss: 1.7811286449432373
     - final_global_loss: 1.6656732633709908
  3. `forget_batch_hessian`：
     - round2_global_loss: 1.7811286449432373
     - final_global_loss: 1.6867005775372188
     - global loss before unlearning: 1.6656732633709908
     - global loss after unlearning: 1.6867005775372188
     - forget client loss: 2.338137240240387 -> 2.3488887650709525
     - unlearning time: 1.4364721775054932s
     - Hessian mode: forget-batch
     - last_hvp_l2_norm: 166.2401686318689
     - actual_param_delta_l2_norm: 0.08718537669219428
     - unlearn_eta: 0.01
     - unlearn_steps_accepted: 1/1

  当前判断：
  1. 正式前三组可作为第一阶段云端正式结果的一部分。
  2. 第四组 retained guard 需要单独补跑。
  3. 原正式 retained_guard 的 `hessian_ref_per_client=4` 在 RTX 5090 32GB 上 OOM。
  4. 单独补跑应使用 `hessian_ref_per_client=1` 作为 OOM 修复；不重跑前三组。

  建议单独补跑命令：
  ```bash
  cd /root/autodl-tmp/FedHDS-zhaoge
  mkdir -p formal_cloud_results/formal_minimal_20260503_retained_ref1/logs
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_DISABLE_XET=1
  export HF_HOME=/root/autodl-tmp/cache
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python main.py \
    --dataset dolly \
    --data_path /root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl \
    --model /root/autodl-tmp/models/Qwen2-0.5B-ms \
    --num_clients 20 \
    -k 0.2 \
    --rounds 10 \
    --batch_or_epoch batch \
    --local_step 2 \
    --batch_size 1 \
    --max_length 64 \
    --data_sample 0.2 \
    --iid 0 \
    --device 0 \
    --log \
    --log_root formal_cloud_results/formal_minimal_20260503_retained_ref1/retained_guard_ref1 \
    --zeroshot \
    --filtering \
    --compound_dim 8 \
    --n_cluster 5 \
    --filtering_sample_limit 20 \
    --use_fedhds_unlearn \
    --forget_client_idx 0 \
    --lissa_depth 1 \
    --lissa_damping 0.01 \
    --unlearn_grad_sample_size 8 \
    --use_retained_hessian \
    --hessian_ref_per_client 1 \
    --unlearn_eta 0.01 \
    --unlearn_max_update_norm 0.3 \
    --unlearn_num_steps 5 \
    --unlearn_global_loss_guard_max 0.8 \
    2>&1 | tee formal_cloud_results/formal_minimal_20260503_retained_ref1/logs/retained_guard_ref1.log
  ```

  如果 ref1 仍 OOM：
  1. 再降 `unlearn_grad_sample_size=4`。
  2. 不优先改其他方法参数。


【本轮进度 57：retained guard ref1 补跑完成，准备汇总正式四组】
  本轮输入：
  1. 用户单独补跑 `retained_guard_ref1`。
  2. 输出显示：
     ```text
     [SAVE] Results saved to formal_cloud_results/formal_minimal_20260503_retained_ref1/retained_guard_ref1/.../final_results.json
     [FINISH] All Rounds and Unlearning Procedure Completed.
     ```
  3. 输出中可见：
     - `Global Loss after Newton-step: 1.6657`
     - `Forget Client 0 Post-Unlearn Eval Loss` 约 2.31-2.32 区间
  4. 终端后续仍显示一些 tqdm 进度条残留，这是输出刷新问题，不代表程序仍在运行。

  当前判断：
  1. 第四组 retained guard 的低显存版本已完成。
  2. 该补跑配置使用：
     - `hessian_ref_per_client=1`
     - `unlearn_grad_sample_size=8`
     - `max_norm=0.3`
     - `steps=5`
     - `global_guard=0.8`
  3. 当前已有第一阶段正式云端四组结果：
     - 前三组在 `formal_cloud_results/formal_minimal_20260503`
     - 第四组在 `formal_cloud_results/formal_minimal_20260503_retained_ref1`
  4. 下一步应先分别汇总第四组，再合并成正式四组总表。

  下一步命令：
  ```bash
  cd /root/autodl-tmp/FedHDS-zhaoge
  python scripts/summarize_experiments.py formal_cloud_results/formal_minimal_20260503_retained_ref1
  python scripts/make_paper_table.py formal_cloud_results/formal_minimal_20260503_retained_ref1/summary.csv
  cat formal_cloud_results/formal_minimal_20260503_retained_ref1/summary.csv
  ```

  合并正式四组总表建议：
  1. 等看到 ref1 summary 的精确数值后，再合并：
     - `formal_cloud_results/formal_minimal_20260503/summary.csv` 的 3 行；
     - `formal_cloud_results/formal_minimal_20260503_retained_ref1/summary.csv` 的 1 行。
  2. 合并输出目录建议：
     - `formal_cloud_results/formal_minimal_20260503_combined`


【本轮进度 58：retained ref1 汇总完成，但发现 global guard 阈值尺度不匹配】
  本轮输入：
  1. 用户执行 retained ref1 汇总：
     ```bash
     python scripts/summarize_experiments.py formal_cloud_results/formal_minimal_20260503_retained_ref1
     python scripts/make_paper_table.py formal_cloud_results/formal_minimal_20260503_retained_ref1/summary.csv
     cat formal_cloud_results/formal_minimal_20260503_retained_ref1/summary.csv
     ```
  2. 输出 1 行 retained ref1 结果。

  retained ref1 精确结果：
  1. group: `retained_guard_ref1`
  2. round2_global_loss: `1.7811286449432373`
  3. final_global_loss: `1.6656732633709908`
  4. global loss before unlearning: `1.6656732633709908`
  5. global loss after unlearning: `1.6656732633709908`
  6. forget client loss: `2.338137240240387 -> 2.338137240240387`
  7. unlearning time: `20.11501145362854s`
  8. hessian mode: `retained-set`
  9. retained reference set size: `19`
  10. last_hvp_l2_norm: `81.22970454183088`
  11. actual_param_delta_l2_norm: `0.0`
  12. unlearn_eta: `0.01`
  13. max_update_norm: `0.3`
  14. unlearn_num_steps: `5`
  15. steps attempted/accepted: `1/0`
  16. guard stop reason: `global_loss_guard`
  17. global guard max: `0.8`
  18. scaled update L2 before/after clipping: `0.04267378679091458`

  关键判断：
  1. 这次 ref1 低显存补跑解决了 OOM。
  2. 但 retained update 没有被接受，实际参数更新为 0。
  3. 原因不是 retained-set Hessian 本身没有方向，而是 `unlearn_global_loss_guard_max=0.8` 与云端正式实验 loss 尺度不匹配。
  4. 云端正式 FedHDS 的 unlearning 前 global loss 已经是 `1.6657`，而 guard 阈值设置为 `0.8`，低于起点，因此任何小步都会因 global loss guard 被拒绝。
  5. 这是从本地小实验阈值迁移到云端正式配置时的阈值尺度错误，需要做一次“配置修正”，不属于参数 sweep。

  建议修正：
  1. 保持低显存设置 `hessian_ref_per_client=1`。
  2. 保持 `max_update_norm=0.3`、`steps=5`、`unlearn_grad_sample_size=8`。
  3. 只修正 global guard 阈值到云端正式 loss 尺度：
     - 建议 `unlearn_global_loss_guard_max=1.70`
  4. 选择 1.70 的理由：
     - FedHDS unlearning 前 global loss 为 1.6657；
     - forget-batch baseline unlearning 后 global loss 为 1.6867；
     - 1.70 允许很小的 utility 损伤窗口，但不放任 global loss 大幅恶化。

  建议单独补跑目录：
  ```text
  formal_cloud_results/formal_minimal_20260503_retained_ref1_guard170
  ```

  这一步是修复绝对 guard 阈值尺度，不是继续 sweep。


【本轮进度 59：retained ref1 guard170 补跑完成】
  本轮输入：
  1. 用户执行 retained ref1 guard170 补跑。
  2. 输出显示：
     ```text
     [UNLEARN CHECK] Global Loss after Newton-step: 1.6615
     [SAVE] Results saved to formal_cloud_results/formal_minimal_20260503_retained_ref1_guard170/retained_guard_ref1_guard170/.../final_results.json
     [FINISH] All Rounds and Unlearning Procedure Completed.
     ```
  3. 输出还显示 forget client post-unlearn loss 在约 `2.3178`、`2.3273`、`2.3298` 的进度刷新中。

  当前判断：
  1. guard170 补跑已完成。
  2. 从日志看 global loss after unlearning 约 `1.6615`，低于 guard=1.70，也低于 unlearning 前 `1.6657`。
  3. 需要用 `summarize_experiments.py` 读取 `final_results.json` 得到精确数值和 step accepted 情况。
  4. 终端残留的 tqdm 进度条不影响结果。

  下一步命令：
  ```bash
  cd /root/autodl-tmp/FedHDS-zhaoge
  python scripts/summarize_experiments.py formal_cloud_results/formal_minimal_20260503_retained_ref1_guard170
  python scripts/make_paper_table.py formal_cloud_results/formal_minimal_20260503_retained_ref1_guard170/summary.csv
  cat formal_cloud_results/formal_minimal_20260503_retained_ref1_guard170/summary.csv
  ```

  如果终端显示仍混乱：
  1. 不再使用 `reset`。
  2. 直接执行命令即可，或退出当前 screen/SSH 后重新登录。


【本轮进度 60：retained ref1 guard170 汇总完成，正式四组结果齐全】
  本轮输入：
  1. 用户执行 guard170 汇总和出表：
     ```bash
     python scripts/summarize_experiments.py formal_cloud_results/formal_minimal_20260503_retained_ref1_guard170
     python scripts/make_paper_table.py formal_cloud_results/formal_minimal_20260503_retained_ref1_guard170/summary.csv
     cat formal_cloud_results/formal_minimal_20260503_retained_ref1_guard170/summary.csv
     ```
  2. 输出 1 行 `retained_guard_ref1_guard170` 结果。

  retained ref1 guard170 精确结果：
  1. group: `retained_guard_ref1_guard170`
  2. round2_global_loss: `1.7811286449432373`
  3. final_global_loss: `1.6614613855878513`
  4. global loss before unlearning: `1.6656732633709908`
  5. global loss after unlearning: `1.6614613855878513`
  6. forget client loss: `2.338137240240387 -> 2.3460232077860366`
  7. unlearning time: `71.13468289375305s`
  8. hessian mode: `retained-set`
  9. retained reference set size: `19`
  10. last_hvp_l2_norm: `81.22970454183088`
  11. actual_param_delta_l2_norm: `0.04267378628769527`
  12. unlearn_eta: `0.01`
  13. max_update_norm: `0.3`
  14. unlearn_num_steps: `5`
  15. steps attempted/accepted: `5/5`
  16. guard stop reason: `completed`
  17. global guard max: `1.7`
  18. applied update L2 estimate: `0.04267378679091458`
  19. clipping enabled: True
  20. clipping applied: False

  当前正式四组结论：
  1. 第一阶段云端正式四组已齐全：
     - `fl`
     - `fedhds`
     - `forget_batch_hessian`
     - `retained_guard_ref1_guard170`
  2. retained-set guard170 没有 OOM。
  3. guard170 接受了 5/5 小步。
  4. retained-set guard170 的 forget loss 小幅上升：
     - `2.3381 -> 2.3460`
  5. global loss 没有恶化，反而略低于 unlearning 前：
     - `1.6657 -> 1.6615`
  6. 但遗忘强度很弱，不能宣称明显优于 forget-batch baseline。
  7. 论文表述应保持谨慎：
     - 云端正式结果验证了 retained-set + guard 的可运行和可控；
     - 当前正式配置下遗忘提升幅度较小；
     - 方法贡献重点是稳定性和可控 trust-region，而不是强遗忘效果全面领先。

  下一步：
  1. 合并前三组正式结果与 guard170 retained 结果。
  2. 生成正式四组 combined summary / paper table / figures。
  3. 下载 combined 结果回本地。


【本轮进度 61：combined summary 尚未生成，改用 Python 合并避免 shell 漏行】
  本轮输入：
  1. 用户尝试生成 combined 表，但先执行了不存在的路径：
     ```bash
     formal_minimal_20260503_combined/summary.csv
     ```
  2. 随后直接运行 `make_paper_table.py` 和 `plot_experiments.py`，但 combined summary 尚未生成，报错：
     ```text
     FileNotFoundError: Summary CSV does not exist
     ```
  3. `cat` 也显示：
     ```text
     No such file or directory
     ```

  当前判断：
  1. 前三组 summary 和 guard170 summary 都存在。
  2. 只是 combined `summary.csv` 还没有生成。
  3. 对新手而言，`head/tail` 多行命令容易漏执行，因此改用一条 Python 命令合并。

  建议命令：
  ```bash
  python -c "import pandas as pd, os; a='formal_cloud_results/formal_minimal_20260503/summary.csv'; b='formal_cloud_results/formal_minimal_20260503_retained_ref1_guard170/summary.csv'; out='formal_cloud_results/formal_minimal_20260503_combined'; os.makedirs(out, exist_ok=True); df=pd.concat([pd.read_csv(a), pd.read_csv(b)], ignore_index=True); df.to_csv(os.path.join(out,'summary.csv'), index=False); print(df[['group','final_global_loss','forget_client_loss_before_unlearning','forget_client_loss_after_unlearning','actual_param_delta_l2_norm','unlearn_steps_accepted','unlearn_guard_stop_reason']].to_string(index=False))"
  python scripts/make_paper_table.py formal_cloud_results/formal_minimal_20260503_combined/summary.csv
  python scripts/plot_experiments.py formal_cloud_results/formal_minimal_20260503_combined/summary.csv --formats png,pdf
  cat formal_cloud_results/formal_minimal_20260503_combined/summary.csv
  ```


【本轮进度 62：combined paper table 和 figures 生成成功】
  本轮输入：
  1. 用户执行：
     ```bash
     python scripts/make_paper_table.py formal_cloud_results/formal_minimal_20260503_combined/summary.csv
     python scripts/plot_experiments.py formal_cloud_results/formal_minimal_20260503_combined/summary.csv --formats png,pdf
     ```
  2. 输出显示：
     - `paper_table.csv` 4 行；
     - `paper_table.md` 4 行；
     - 生成 12 个 figure 文件。

  已生成 combined 文件：
  1. `formal_cloud_results/formal_minimal_20260503_combined/summary.csv`
  2. `formal_cloud_results/formal_minimal_20260503_combined/paper_table.csv`
  3. `formal_cloud_results/formal_minimal_20260503_combined/paper_table.md`
  4. `formal_cloud_results/formal_minimal_20260503_combined/figures/forget_loss_before_after.png`
  5. `formal_cloud_results/formal_minimal_20260503_combined/figures/forget_loss_before_after.pdf`
  6. `formal_cloud_results/formal_minimal_20260503_combined/figures/global_loss_final.png`
  7. `formal_cloud_results/formal_minimal_20260503_combined/figures/global_loss_final.pdf`
  8. `formal_cloud_results/formal_minimal_20260503_combined/figures/update_norm_vs_global_loss.png`
  9. `formal_cloud_results/formal_minimal_20260503_combined/figures/update_norm_vs_global_loss.pdf`
  10. `formal_cloud_results/formal_minimal_20260503_combined/figures/update_norm_vs_forget_gain.png`
  11. `formal_cloud_results/formal_minimal_20260503_combined/figures/update_norm_vs_forget_gain.pdf`
  12. `formal_cloud_results/formal_minimal_20260503_combined/figures/guard_steps.png`
  13. `formal_cloud_results/formal_minimal_20260503_combined/figures/guard_steps.pdf`
  14. `formal_cloud_results/formal_minimal_20260503_combined/figures/unlearning_time.png`
  15. `formal_cloud_results/formal_minimal_20260503_combined/figures/unlearning_time.pdf`

  当前判断：
  1. combined summary 已存在，否则 make_paper_table 和 plot 会报 FileNotFoundError。
  2. 还需要 `cat summary.csv` 确认四组数值。
  3. 然后下载 combined 目录和关键正式原始结果回本地。

  下一步：
  ```bash
  cat formal_cloud_results/formal_minimal_20260503_combined/summary.csv
  ```


【本轮进度 63：正式 combined 四组 summary 确认】
  本轮输入：
  1. 用户执行：
     ```bash
     cat formal_cloud_results/formal_minimal_20260503_combined/summary.csv
     ```
  2. 输出包含 4 组正式云端结果：
     - `fedhds`
     - `fl`
     - `forget_batch_hessian`
     - `retained_guard_ref1_guard170`

  正式云端 combined 结果目录：
  ```text
  /root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results/formal_minimal_20260503_combined
  ```

  正式四组关键结果：
  1. `FL`：
     - round2_global_loss: `1.783530279994011`
     - final_global_loss: `1.6442784418662388`
  2. `FedHDS`：
     - round2_global_loss: `1.7811286449432373`
     - final_global_loss: `1.6656732633709908`
  3. `FedHDS + forget-batch Hessian`：
     - global loss before unlearning: `1.6656732633709908`
     - global loss after unlearning: `1.6867005775372188`
     - forget client loss: `2.338137240240387 -> 2.3488887650709525`
     - unlearning time: `1.4364721775054932s`
     - hessian mode: `forget-batch`
     - last_hvp_l2_norm: `166.2401686318689`
     - actual_param_delta_l2_norm: `0.0871853766921942`
     - unlearn_eta: `0.01`
     - steps accepted: `1/1`
  4. `FedHDS + retained-set Hessian + stepped guard`：
     - group: `retained_guard_ref1_guard170`
     - global loss before unlearning: `1.6656732633709908`
     - global loss after unlearning: `1.6614613855878513`
     - forget client loss: `2.338137240240387 -> 2.346023207786037`
     - unlearning time: `71.13468289375305s`
     - hessian mode: `retained-set`
     - retained reference set size: `19`
     - last_hvp_l2_norm: `81.22970454183088`
     - actual_param_delta_l2_norm: `0.0426737862876952`
     - unlearn_eta: `0.01`
     - max_update_norm: `0.3`
     - unlearn_num_steps: `5`
     - steps accepted: `5/5`
     - guard stop reason: `completed`
     - global guard max: `1.7`
     - clipping enabled: True
     - clipping applied: False

  阶段性结论：
  1. 第一阶段正式云端最小四组实验已经完成。
  2. 云端正式结果比 smoke 正常，global loss 已降至 1.64-1.69 区间。
  3. forget-batch Hessian 带来轻微 forget loss 提升：
     - `+0.0108`
     同时 global loss 轻微变差：
     - `1.6657 -> 1.6867`
  4. retained-set guard170 带来轻微 forget loss 提升：
     - `+0.0079`
     同时 global loss 未恶化，略降：
     - `1.6657 -> 1.6615`
  5. retained-set guard170 的遗忘幅度弱于 forget-batch baseline，但 utility 更稳。
  6. retained-set guard170 的耗时明显更高：
     - `71.13s` vs forget-batch `1.44s`
  7. 当前不能宣称 retained-set 方法全面优于 baseline；更稳妥的论文结论是：
     - retained-set curvature + stepped guard 在云端正式配置下可运行且可控；
     - 在约束 global utility 的同时能产生小幅 forgetting；
     - 但存在遗忘强度弱和二阶计算耗时/显存成本高的问题。

  下一步：
  1. 下载 `formal_minimal_20260503_combined` 回本地。
  2. 下载完整正式原始目录或至少关键 final_results/log。
  3. 若短期不继续跑实验，关机止损。


【本轮进度 64：正式云端结果已下载回本地并确认】
  本轮输入：
  1. 用户在本地 PowerShell 下载 combined 目录：
     ```powershell
     scp -P 56117 -r "root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results/formal_minimal_20260503_combined" "F:\FedHDS-zhaoge\formal_cloud_results_formal_minimal_20260503_combined"
     ```
  2. 用户随后下载完整云端结果目录：
     ```powershell
     scp -P 56117 -r "root@connect.westc.seetacloud.com:/root/autodl-tmp/FedHDS-zhaoge/formal_cloud_results" "F:\FedHDS-zhaoge\formal_cloud_results_autodl_20260503"
     ```

  本地确认：
  1. combined 结果目录存在：
     ```text
     F:\FedHDS-zhaoge\formal_cloud_results_formal_minimal_20260503_combined
     ```
     包含：
     - `summary.csv`
     - `paper_table.csv`
     - `paper_table.md`
     - `figures/`
  2. 完整云端结果目录存在：
     ```text
     F:\FedHDS-zhaoge\formal_cloud_results_autodl_20260503
     ```
     包含：
     - `smoke`
     - `formal_minimal_20260503`
     - `formal_minimal_20260503_retained_ref1`
     - `formal_minimal_20260503_retained_ref1_guard170`
     - `formal_minimal_20260503_combined`
  3. 本地 combined `summary.csv` 已读取确认，4 组数据完整。

  当前最终正式云端结果文件：
  1. 主表：
     ```text
     F:\FedHDS-zhaoge\formal_cloud_results_formal_minimal_20260503_combined\summary.csv
     ```
  2. 论文压缩表：
     ```text
     F:\FedHDS-zhaoge\formal_cloud_results_formal_minimal_20260503_combined\paper_table.md
     ```
  3. 图目录：
     ```text
     F:\FedHDS-zhaoge\formal_cloud_results_formal_minimal_20260503_combined\figures
     ```
  4. 完整原始结果归档：
     ```text
     F:\FedHDS-zhaoge\formal_cloud_results_autodl_20260503
     ```

  关键阶段性结论：
  1. AutoDL 云端从购买、环境配置、项目上传、依赖安装、模型下载、smoke 到正式四组实验均已完成。
  2. 第一阶段正式最小四组已有可用结果。
  3. retained-set guard 的正式可用配置为：
     - `hessian_ref_per_client=1`
     - `unlearn_grad_sample_size=8`
     - `max_update_norm=0.3`
     - `steps=5`
     - `global_guard=1.7`
  4. 原 `hessian_ref_per_client=4` 在 RTX 5090 32GB 上 OOM，不建议直接作为正式配置重复运行。
  5. 原 `global_guard=0.8` 不适合云端正式 loss 尺度，会导致 0 步接受。
  6. 当前正式结论应谨慎写作：retained-set guard 可运行、可控、utility 稳，但 forgetting 提升弱、耗时高。

  成本建议：
  1. 结果已下载回本地。
  2. 如果短期不继续运行实验，应立即在 AutoDL 控制台关机止损。
  3. 若只是保留文件，确认 AutoDL 数据盘/实例策略后再决定是否释放实例或保留镜像。


【本轮进度 65：第一阶段结果文字版与 Word 报告版生成】
  本轮目标：
  1. 用户要求将第一阶段整理结果的文字部分写入 `readme_result1.txt`。
  2. 用户要求将主表、结论边界、retained guard 优缺点等全面整理内容写入 `readme_result_1.docx`。

  已生成/覆盖文件：
  1. 文本整理版：
     ```text
     F:\FedHDS-zhaoge\readme_result1.txt
     ```
  2. Word 报告版：
     ```text
     F:\FedHDS-zhaoge\readme_result_1.docx
     ```

  文件内容：
  1. `readme_result1.txt` 包含：
     - 第一阶段当前状态；
     - 实验设置；
     - 正式主结果摘要；
     - 结果解读；
     - retained guard 优点；
     - retained guard 缺点；
     - 结论边界；
     - 论文写法建议；
     - 下一步建议。
  2. `readme_result_1.docx` 包含完整结构化报告：
     - 阶段状态；
     - 实验设置表；
     - 正式主表；
     - 关键数值解读；
     - retained guard 优点；
     - retained guard 缺点；
     - 结论边界；
     - 论文写法建议；
     - 下一步建议。

  技术说明：
  1. 当前本地 Python 环境没有 `python-docx`。
  2. 因此 `readme_result_1.docx` 使用标准 OpenXML `.docx` 结构直接生成。
  3. 已检查 docx 包结构包含：
     - `[Content_Types].xml`
     - `_rels/.rels`
     - `word/document.xml`
     - `word/styles.xml`
     - `word/_rels/document.xml.rels`
  4. `readme_result1.txt` 已按 UTF-8 保存，Windows 下应可正常读取中文。

  写作边界：
  1. 文档明确说明第一阶段已经完成。
  2. 文档明确避免“retained-set 全面优于 baseline”的过度表述。
  3. 文档将 retained-set guard 定位为稳定性与可控性改进。
  4. 文档记录了当前正式结果的局限：
     - forgetting 提升弱；
     - retained guard 耗时高；
     - retained Hessian 显存敏感；
     - global guard 阈值依赖 loss 尺度。
