可以，下面这版就是按文件拆分的最小改造清单。目标只有一个：

  不改训练主链路，只把 unlearning 里的 Hessian 参考目标，从 forget client batch 改成 retained reference set。

  范围
  只动 3 个文件：

  - main.py:87
  - server.py:151
  - client.py:47

  m_utils.py、load_data.py、FedHDS 聚类逻辑先不动。

  1. main.py:87
  要加的参数，尽量少：

  - --use_retained_hessian
      - action='store_true'
      - 默认关闭，保留当前 baseline 路径
  - --hessian_ref_per_client
      - type=int
      - 默认建议 2 或 4
      - 表示从每个非忘却客户端抽多少样本进 D_ref

  这文件只做两件事：

  - 加参数
  - 保持 server.apply_fedhds_unlearning(...) 调用不变，让 server.py 自己读参数分支

  不要在这里引入新流程，不要碰训练轮次逻辑。

  2. client.py:47
  这里加一个很小的工具接口，别改现有训练逻辑：

  新增一个方法，建议名字类似：

  - sample_reference_examples(sample_size, shuffle=False)

  它做的事很简单：

  - 数据来源必须是 self.full_train_dataset，不是 self.train_loader
  - 从完整客户端数据 D_k 里取前 sample_size 条，或随机取 sample_size 条
  - 返回 list[example]

  这里的原则：

  - 只给 unlearning 提供 retained reference samples
  - 不碰 FedHDS 训练集构建
  - 不碰 use_full_training_set() 和 build_training_set_with_precalculated_clusters()

  3. server.py:151
  这是唯一的核心改动区。

  建议新增 3 个 helper：

  1. build_retained_reference_loader(forget_client_idx, client_list)

  - 遍历所有 idx != forget_client_idx 的客户端
  - 每个客户端调用 client.sample_reference_examples(args.hessian_ref_per_client)
  - 拼成一个小的 reference_examples 列表
  - 用任意一个 retained client 的 full_collate_fn 构造 DataLoader
  - 这个 loader 就是 D_ref

  2. _compute_average_loss(data_loader)

  - 遍历 D_ref
  - 计算平均 loss
  - 返回一个用于 HVP 的标量 loss

  3. 复用现有 _compute_average_forget_gradients(...)

  - 这部分继续算 g_j
  - 数据来源还是 forget client 完整数据 D_j
  - 这部分不要改方向

  然后改 server.py:182：

  - 先照旧算 g_j
  - 如果 args.use_retained_hessian：
      - 构造 ref_loader = build_retained_reference_loader(...)
      - 用 _compute_average_loss(ref_loader) 得到 reference_loss
      - LiSSA/HVP 用这个 reference_loss
  - 否则：
      - 走你现在的 baseline 路径

  这样就形成两条可对比路径：

  - baseline: forget-data Hessian
  - ours: retained-set Hessian

  4. 明确不做
  这一轮先不要做：

  - 不引入 R 补偿集
  - 不做 KD 恢复
  - 不做 FedUR 历史残差存储
  - 不做跨层特征融合
  - 不改 FedHDS 聚类/筛样本逻辑
  - 不改数据规模和正式实验超参

  5. 实现顺序
  按这个顺序最稳：

  1. 先改 client.py，补采样接口
  2. 再改 server.py，把 D_ref 和 Hessian 分支接起来
  3. 最后改 main.py，补参数开关
  4. 先 smoke test
  5. 再做 baseline vs retained-Hessian 对比

  6. 完成标准
  做到下面这 3 条就够：

  - --use_fedhds_unlearn 且不加 --use_retained_hessian 时，老路径还能跑
  - 加上 --use_retained_hessian 后，新路径能跑
  - 日志里能明确打印：
      - forget gradient sample size
      - retained reference set size
      - LiSSA depth / damping