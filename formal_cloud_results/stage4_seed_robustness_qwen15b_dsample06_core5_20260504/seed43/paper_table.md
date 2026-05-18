| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| fedhds | eta=1.0000 | none | - | 1.6377 | - | - | - | utility baseline |
| fl | eta=1.0000 | none | - | 1.6790 | - | - | - | utility baseline |
| forget_batch_hessian | eta=0.0100 | forget-batch | 1.8483 -> 1.8490 | 1.6368 | 0.0155 | 1/1 | 1.10 | forget loss increased |
| retained_auto_forget_guard_tol0_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.738, sign=negative, forget_guard=nondecrease | retained-set | 1.8483 -> 1.8472 | 1.6380 | 0.0307 | 5/5 | 272.95 | best probe sign negative, forget delta 0.0083 |
| retained_negative_norm010_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.738, sign=negative, forget_guard=nondecrease | retained-set | 1.8483 -> 1.8472 | 1.6380 | 0.0307 | 5/5 | 193.25 |  |
