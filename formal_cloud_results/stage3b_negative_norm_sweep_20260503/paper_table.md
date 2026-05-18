| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| negative_norm005_guard05 | eta=0.0100, max_norm=0.050, steps=5, global_guard=1.575, sign=negative, forget_guard=nondecrease | retained-set | 2.2581 -> 2.4105 | 1.5561 | 0.0500 | 5/5 | 132.76 | forget loss increased |
| negative_norm010_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.625, sign=negative, forget_guard=nondecrease | retained-set | 2.2581 -> 2.6104 | 1.5935 | 0.1000 | 5/5 | 130.74 | forget loss increased |
| negative_norm015_guard10 | eta=0.0100, max_norm=0.150, steps=5, global_guard=1.625, sign=negative, forget_guard=nondecrease | retained-set | 2.2581 -> 2.6880 | 1.6094 | 0.1200 | 4/5 | 132.08 | guard stopped by global_loss_guard |
