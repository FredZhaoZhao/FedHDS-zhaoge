| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| fedhds | eta=1.0000 | none | - | 2.0818 | - | - | - | utility baseline |
| fl | eta=1.0000 | none | - | 2.1781 | - | - | - | utility baseline |
| forget_batch_hessian | eta=0.0100 | forget-batch | 1.8732 -> 1.8677 | 2.0871 | 0.0496 | 1/1 | 1.05 |  |
| retained_auto_forget_guard_tol0_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=2.182, forget_guard=nondecrease | retained-set | 1.8732 -> 1.8725 | 2.0826 | 0.0058 | 5/5 | 283.76 | best probe forget delta 0.0108 |
| retained_negative_norm010_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=2.182, sign=negative, forget_guard=nondecrease | retained-set | 1.8732 -> 1.8732 | 2.0818 | 0.0000 | 0/1 | 44.32 | guard stopped by forget_loss_guard |
