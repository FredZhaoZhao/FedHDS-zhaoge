| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| fedhds | eta=1.0000 | none | - | 1.6201 | - | - | - | utility baseline |
| fl | eta=1.0000 | none | - | 1.6587 | - | - | - | utility baseline |
| forget_batch_hessian | eta=0.0100 | forget-batch | 2.0880 -> 2.0828 | 1.6291 | 0.2335 | 1/1 | 1.07 |  |
| retained_auto_direction_guard05 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.670, sign=negative | retained-set | 2.0880 -> 2.1078 | 1.6132 | 0.0305 | 5/5 | 269.13 | best probe sign negative, forget delta 0.0263 |
| retained_auto_forget_guard_tol005_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.720, sign=negative, forget_guard=tol0.0050 | retained-set | 2.0880 -> 2.1078 | 1.6132 | 0.0305 | 5/5 | 273.59 | best probe sign negative, forget delta 0.0263 |
| retained_auto_forget_guard_tol0_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.720, sign=negative, forget_guard=nondecrease | retained-set | 2.0880 -> 2.1078 | 1.6132 | 0.0305 | 5/5 | 286.24 | best probe sign negative, forget delta 0.0263 |
| retained_negative_norm005_guard05 | eta=0.0100, max_norm=0.050, steps=5, global_guard=1.670, sign=negative, forget_guard=nondecrease | retained-set | 2.0880 -> 2.1078 | 1.6132 | 0.0305 | 5/5 | 197.30 | forget loss increased |
| retained_negative_norm010_guard10 | eta=0.0100, max_norm=0.100, steps=5, global_guard=1.720, sign=negative, forget_guard=nondecrease | retained-set | 2.0880 -> 2.1078 | 1.6132 | 0.0305 | 5/5 | 207.45 | forget loss increased |
| retained_negative_norm015_guard10 | eta=0.0100, max_norm=0.150, steps=5, global_guard=1.720, sign=negative, forget_guard=nondecrease | retained-set | 2.0880 -> 2.1078 | 1.6132 | 0.0305 | 5/5 | 195.08 | forget loss increased |
