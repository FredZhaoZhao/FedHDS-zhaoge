| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| fedhds | eta=1.0000 | none | - | 1.5250 | - | - | - | utility baseline |
| fl | eta=1.0000 | none | - | 1.5441 | - | - | - | utility baseline |
| forget_batch_hessian | eta=0.0100 | forget-batch | 2.2581 -> 2.2866 | 1.6257 | 0.5077 | 1/1 | 1.69 | forget loss increased |
| retained_auto_direction_guard05 | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.575 | retained-set | 2.2581 -> 2.2329 | 1.5482 | 0.2486 | 5/5 | 189.67 | best probe forget delta -0.0073 |
| retained_auto_forget_guard_tol005_guard10 | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.625, forget_guard=tol0.0050 | retained-set | 2.2581 -> 2.2581 | 1.5250 | 0.0000 | 0/1 | 86.58 | guard stopped by forget_loss_guard |
| retained_auto_forget_guard_tol0_guard05 | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.575, forget_guard=nondecrease | retained-set | 2.2581 -> 2.2581 | 1.5250 | 0.0000 | 0/1 | 88.01 | guard stopped by forget_loss_guard |
| retained_negative_norm005_guard05 | eta=0.0100, max_norm=0.050, steps=5, global_guard=1.575, sign=negative, forget_guard=nondecrease | retained-set | 2.2581 -> 2.4105 | 1.5561 | 0.0500 | 5/5 | 133.76 | forget loss increased |
| retained_negative_norm010_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.625, sign=negative, forget_guard=nondecrease | retained-set | 2.2581 -> 2.6104 | 1.5935 | 0.1000 | 5/5 | 140.23 | forget loss increased |
| retained_negative_norm015_guard10 | eta=0.0100, max_norm=0.150, steps=5, global_guard=1.625, sign=negative, forget_guard=nondecrease | retained-set | 2.2581 -> 2.6880 | 1.6094 | 0.1200 | 4/5 | 137.92 | guard stopped by global_loss_guard |
