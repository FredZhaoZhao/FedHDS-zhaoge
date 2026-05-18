| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| fedhds | eta=1.0000 | none | - | 1.6438 | - | - | - | utility baseline |
| fl | eta=1.0000 | none | - | 1.6611 | - | - | - | utility baseline |
| forget_batch_hessian | eta=0.0100 | forget-batch | 1.8812 -> 1.8702 | 1.6408 | 0.0519 | 1/1 | 1.21 |  |
| retained_auto_direction_guard05 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.694, sign=negative | retained-set | 1.8812 -> 1.9206 | 1.6405 | 0.0282 | 5/5 | 183.74 | best probe sign negative, forget delta 0.0151 |
| retained_auto_forget_guard_tol005_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.744, sign=negative, forget_guard=tol0.0050 | retained-set | 1.8812 -> 1.9206 | 1.6405 | 0.0282 | 5/5 | 184.75 | best probe sign negative, forget delta 0.0151 |
| retained_auto_forget_guard_tol0_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.744, sign=negative, forget_guard=nondecrease | retained-set | 1.8812 -> 1.9206 | 1.6405 | 0.0282 | 5/5 | 183.97 | best probe sign negative, forget delta 0.0151 |
| retained_negative_norm005_guard05 | eta=0.0100, max_norm=0.050, steps=5, global_guard=1.694, sign=negative, forget_guard=nondecrease | retained-set | 1.8812 -> 1.9206 | 1.6405 | 0.0282 | 5/5 | 138.44 | forget loss increased |
| retained_negative_norm010_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.744, sign=negative, forget_guard=nondecrease | retained-set | 1.8812 -> 1.9206 | 1.6405 | 0.0282 | 5/5 | 130.41 | forget loss increased |
| retained_negative_norm015_guard10 | eta=0.0100, max_norm=0.150, steps=5, global_guard=1.744, sign=negative, forget_guard=nondecrease | retained-set | 1.8812 -> 1.9206 | 1.6405 | 0.0282 | 5/5 | 133.38 | forget loss increased |
