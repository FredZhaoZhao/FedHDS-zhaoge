| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| fedhds | eta=1.0000 | none | - | 1.5250 | - | - | - | utility baseline |
| fl | eta=1.0000 | none | - | 1.5441 | - | - | - | utility baseline |
| retrain_baseline | exclude client 0, lr=5e-5 | retrain-from-scratch | heldout client eval = 2.2168 | 1.5355 | - | - | - | gold-standard exact removal reference |
| forget_batch_hessian | eta=0.0100 | forget-batch | 2.2581 -> 2.2866 | 1.6257 | 0.5077 | 1/1 | 1.64 | forget loss increased |
| retained_guard_ref1_autoguard | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.575 | retained-set | 2.2581 -> 2.2329 | 1.5482 | 0.2486 | 5/5 | 131.13 | utility preserved, weak forgetting |
