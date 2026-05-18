 可以。下面这版就是我建议的 “小论文最小方法方案”。它的目标很明确：

  不扩散题目，不堆太多机制，只在你现有 FedHDS + 联邦遗忘代码上，做一个小而完整、能讲清楚的改进。

  题目定位

  建议主线定成：

  FedHDS-assisted Federated Unlearning with Retained-Set Curvature Approximation

  中文可以写成：

  一种基于保留参考集曲率近似的联邦遗忘方法

  这里的关键词只有两个：

  - FedHDS：作为训练阶段的数据筛选/联邦训练底座
  - Retained-Set Curvature：作为你真正的方法改进点

  这样题目不会散。

  核心方法

  保留你现在已经跑通的框架：

  1. 先用 FedHDS 或普通 FL 训练全局模型
  2. 指定一个忘却客户端 j
  3. 用近似二阶更新做 unlearning

  但把你现在 server.py:182 里的遗忘步骤，改成下面这个版本：

  [
  M_{u}=M^{}+\eta \cdot \frac{1}{K-1} H_{\text{ref}}^{-1}\nabla L_{j}(M^{})
  ]

  其中：

  - (\nabla L_j(M^*))：仍然来自 忘却客户端完整数据 D_j
  - (H_{\text{ref}})：不再用 forget client 的 batch 近似
  - (H_{\text{ref}}) 改为来自 剩余客户端参考集 D_ref

  这就是你的小创新点。

  一句话解释这个创新

  现有粗糙做法把“要删掉谁”和“系统剩余曲率长什么样”混在一起了。
  你的改进是：

  - 用忘却客户端数据定义“删什么”
  - 用保留客户端数据定义“删完以后模型应该沿着什么曲率面恢复”

  这个逻辑是顺的，而且很好写进论文。

  D_ref 怎么构造

  为了控制工程量，不做复杂版，直接做最小实现：

  - 从所有非忘却客户端中采样少量样本，组成 D_ref
  - 如果 --filtering=True，优先从每个客户端本轮训练集 Ω_k 里取
  - 如果 --filtering=False，就从各客户端完整训练集 D_k 随机取少量样本

  这样有两个好处：

  - 和你现有 FedHDS 自然接上
  - 计算量可控

  建议新增参数

  放在 main.py:87 这一段参数区就够：

  - --hessian_ref_mode {random, filtered}
  - --hessian_ref_size
  - --hessian_ref_per_client
  - --use_retained_hessian

  已有的这几个继续保留：

  - --lissa_depth
  - --lissa_damping
  - --unlearn_grad_sample_size

  代码改动范围

  只动 3 个点，别扩散：

  1. server.py:151
      - 增加 build_retained_reference_loader(...)
      - LiSSA/HVP 不再围绕 forget batch，而是围绕 D_ref
  2. client.py:47
      - 已经有 get_full_train_loader()，基本够用
      - 最多补一个“小样本引用接口”，别大改
  3. main.py:221
      - unlearning 阶段把 client_list 传给 server
      - 加新参数，不改训练主流程

  明确不做的东西

  这篇小论文先不要碰：

  - 跨层特征融合
  - KD 恢复
  - FedUR 全量更新残差历史
  - 差分隐私
  - 模型剪枝
  - 去中心化联邦遗忘
  - 补偿集 R

  这些东西都容易把题目搞散。

  实验最小闭环

  只做 4 组就够：

  1. FL
  2. FedHDS
  3. FedHDS + Unlearning (forget-batch Hessian)
     这个作为你当前方法 baseline
  4. FedHDS + Unlearning (retained-set Hessian)
     这个作为你的方法

  指标也收紧

  只回答 3 个问题：

  1. 效用
      - 全局 eval loss / perplexity
  2. 遗忘效果
      - forget client 上的 loss 变化
      - 或 forget client 相关样本上的生成质量下降
  3. 遗忘代价
      - unlearning 额外耗时

  如果你精力够，再加一个可选指标：

  4. 与重训的接近程度
      - “移除该客户端后重训模型” vs “unlearn 模型”的参数距离或 eval 差距

  但这项是可选，不强求。

  论文贡献可以写成这样

  1. 构建了一个可运行的 FedHDS + federated unlearning 框架
  2. 提出一种 保留参考集曲率近似 的联邦遗忘方法
  3. 实验证明该方法在保持遗忘能力的同时，更好地维持剩余任务效用

  这个贡献强度，写小论文是够的。

  实施顺序

  1. 保持当前代码不动，作为 baseline
  2. 只实现 retained-set Hessian
  3. 先做 smoke test
  4. 再做小规模对比实验
  5. 最后上云调数值


  云服务器正式实验规划

  先不要一上来就换大模型。更稳的顺序是三阶段推进：

  第一阶段：LoRA + Qwen2-0.5B 完整跑通

  目标是证明完整实验链路可靠，而不是立刻追求最好结果。

  这一阶段先跑 4 组：

  1. FL
  2. FedHDS
  3. FedHDS + Unlearning (forget-batch Hessian)
  4. FedHDS + Unlearning (retained-set Hessian / retained guard)

  这一阶段要产出：

  - 每组的 final_results.json
  - 汇总表 summary.csv
  - 论文压缩表 paper_table.md / paper_table.csv
  - figures 里的结果图片

  只有当这一步在云服务器上完整跑通后，才进入下一阶段。

  第二阶段：仍然使用 Qwen2-0.5B，但扩大实验规模

  目标是让结果更可信，而不是急着换模型。

  可以逐步扩大：

  - data_sample
  - rounds
  - num_clients
  - 换 seed
  - 换 forget client

  这一阶段主要验证方法稳定性，观察 retained-set Hessian / guard 在更大数据量和更多客户端下是否仍然保持类似趋势。

  第三阶段：换稍大一些的模型做补充验证

  目标是证明方法不是只在 0.5B 模型上能跑。

  可选模型优先级：

  1. Qwen2.5-1.5B
  2. Qwen2-1.5B
  3. Qwen2.5-3B

  暂时不建议直接上 7B。7B 的成本、显存压力和调参风险都更高，不适合作为第一轮正式云端实验。

  总体原则

  先用 LoRA + Qwen2-0.5B 在云服务器上拿到一套完整、可复现、可出表和可出图的正式结果；确认趋势后，再扩大数据规模或换更大模型。
