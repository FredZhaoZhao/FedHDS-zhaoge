| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| seed43/fedhds | eta=1.0000 | none | - | 1.6377 | - | - | - | utility baseline |
| seed43/fl | eta=1.0000 | none | - | 1.6790 | - | - | - | utility baseline |
| seed43/forget_batch_hessian | eta=0.0100 | forget-batch | 1.8483 -> 1.8490 | 1.6368 | 0.0155 | 1/1 | 1.10 | forget loss increased |
| seed43/retained_auto_forget_guard_tol0_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.738, sign=negative, forget_guard=nondecrease | retained-set | 1.8483 -> 1.8472 | 1.6380 | 0.0307 | 5/5 | 272.95 | best probe sign negative, forget delta 0.0083 |
| seed43/retained_negative_norm010_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.738, sign=negative, forget_guard=nondecrease | retained-set | 1.8483 -> 1.8472 | 1.6380 | 0.0307 | 5/5 | 193.25 |  |
| seed44/fedhds | eta=1.0000 | none | - | 2.0818 | - | - | - | utility baseline |
| seed44/fl | eta=1.0000 | none | - | 2.1781 | - | - | - | utility baseline |
| seed44/forget_batch_hessian | eta=0.0100 | forget-batch | 1.8732 -> 1.8677 | 2.0871 | 0.0496 | 1/1 | 1.05 |  |
| seed44/retained_auto_forget_guard_tol0_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=2.182, forget_guard=nondecrease | retained-set | 1.8732 -> 1.8725 | 2.0826 | 0.0058 | 5/5 | 283.76 | best probe forget delta 0.0108 |
| seed44/retained_negative_norm010_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=2.182, sign=negative, forget_guard=nondecrease | retained-set | 1.8732 -> 1.8732 | 2.0818 | 0.0000 | 0/1 | 44.32 | guard stopped by forget_loss_guard |
