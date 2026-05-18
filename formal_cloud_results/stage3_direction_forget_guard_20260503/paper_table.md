| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| fedhds | eta=1.0000 | none | - | 1.5250 | - | - | - | utility baseline |
| retained_auto_direction_guard05 | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.575 | retained-set | 2.2581 -> 2.2329 | 1.5482 | 0.2486 | 5/5 | 180.68 | best probe forget delta -0.0073 |
| retained_auto_forget_guard_tol005_guard10 | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.625, forget_guard=tol0.0050 | retained-set | 2.2581 -> 2.2581 | 1.5250 | 0.0000 | 0/1 | 84.32 | guard stopped by forget_loss_guard |
| retained_auto_forget_guard_tol0_guard05 | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.575, forget_guard=nondecrease | retained-set | 2.2581 -> 2.2581 | 1.5250 | 0.0000 | 0/1 | 83.71 | guard stopped by forget_loss_guard |
