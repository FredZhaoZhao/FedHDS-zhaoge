| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| fedhds | eta=1.0000 | none | - | 1.5423 | - | - | - | utility baseline |
| fl | eta=1.0000 | none | - | 1.5289 | - | - | - | utility baseline |
| forget_batch_hessian | eta=0.0100 | forget-batch | 2.4377 -> 2.8059 | 2.0040 | 1.1020 | 1/1 | 1.49 | forget loss increased |
| retained_auto_direction_guard05 | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.592, sign=negative | retained-set | 2.4377 -> 2.7601 | 1.5862 | 0.1483 | 5/5 | 283.13 | best probe sign negative, forget delta 1.1020 |
| retained_auto_forget_guard_tol005_guard10 | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.642, sign=negative, forget_guard=tol0.0050 | retained-set | 2.4377 -> 2.7601 | 1.5862 | 0.1483 | 5/5 | 286.25 | best probe sign negative, forget delta 1.1020 |
| retained_auto_forget_guard_tol0_guard05 | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.592, sign=negative, forget_guard=nondecrease | retained-set | 2.4377 -> 2.7601 | 1.5862 | 0.1483 | 5/5 | 281.33 | best probe sign negative, forget delta 1.1020 |
| retained_negative_norm005_guard05 | eta=0.0100, max_norm=0.050, steps=5, global_guard=1.592, sign=negative, forget_guard=nondecrease | retained-set | 2.4377 -> 2.5030 | 1.5554 | 0.0500 | 5/5 | 202.77 | forget loss increased |
| retained_negative_norm010_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.642, sign=negative, forget_guard=nondecrease | retained-set | 2.4377 -> 2.6147 | 1.5701 | 0.1000 | 5/5 | 206.28 | forget loss increased |
| retained_negative_norm015_guard10 | eta=0.0100, max_norm=0.150, steps=5, global_guard=1.642, sign=negative, forget_guard=nondecrease | retained-set | 2.4377 -> 2.7601 | 1.5862 | 0.1483 | 5/5 | 201.35 | forget loss increased |
