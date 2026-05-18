# Guarded Federated Unlearning via Retained-Set Curvature

## Abstract

Federated unlearning seeks to reduce the influence of a target client without fully retraining the federated model. In language-model federated instruction tuning, the challenge is to weaken the forget signal while preserving retained-client utility. We study a guarded post-training unlearning method on top of FedHDS: the forget-client gradient defines what should be removed, retained-client curvature defines the utility-preserving geometry, and global norm clipping, stepped trust-region checking, direction probing, and a forget-loss guard control the update. Experiments on Dolly with Qwen2-0.5B and Qwen2.5-1.5B show that forget-batch Hessian usually gives stronger forgetting but larger utility damage, whereas retained-set curvature helps only after direction correction and guard control. The exact gain remains scale- and seed-sensitive.

Keywords: federated unlearning; retained-set curvature; federated instruction tuning; large language models; utility-preserving unlearning

## 1. Introduction

Federated learning allows multiple clients to collaboratively train a global model without directly sharing their local data<sup>[1]</sup>. This training paradigm is useful when data are distributed across users, institutions, or devices, and when centralizing the raw data is undesirable. However, a trained federated model may later face an unlearning request: a client may withdraw consent, request data deletion, or become invalid due to data quality or compliance concerns. In such a case, the system should remove or reduce the influence of that client from the global model.

The most direct solution is to retrain the federated model after excluding the target client<sup>[2]</sup>. Although retraining provides a clean reference, it is often impractical. Federated training may involve many communication rounds, repeated local fine-tuning, and large model checkpoints. The cost becomes even more pronounced for language-model fine-tuning, where even one full run can already require substantial computing resources and time. Federated unlearning therefore studies how to approximate the effect of removing a client without performing full retraining.

Approximate federated unlearning is difficult because it has two competing requirements. On the one hand, the update should weaken the forgotten client's influence. In this work, that objective is evaluated operationally through forget-client loss: after unlearning, the loss on the forgotten client should not decrease and preferably should increase. On the other hand, the update should preserve the utility of the model on the retained data distribution. If the update increases forget loss but severely damages global performance, the method is not practically useful.

A central challenge in federated unlearning is the trade-off between forgetting and retained utility. An aggressive update may increase the loss on the forget client, but it can also damage performance on the retained clients. Conversely, an overly conservative update may preserve global utility but fail to meaningfully reduce the forgotten client's influence. This trade-off is especially important when second-order approximations are used: Hessian or inverse-Hessian-vector-product based updates can estimate parameter corrections for removing data influence<sup>[3][4]</sup>, but the choice of curvature reference directly affects both the direction and magnitude of the unlearning update.

Existing second-order unlearning approximations often use a gradient from the target data together with a Hessian or inverse-Hessian-vector product. A natural baseline is to compute both the gradient and curvature reference from the forget-client batch. This forget-batch Hessian approach is directly aligned with the forget objective, but it may move the model in a way that is harmful to retained utility. In a federated setting, where the post-unlearning model should still serve the remaining clients, this is a serious limitation.

We adopt FedHDS, the federated data-efficient instruction-tuning backbone introduced by Qin et al.<sup>[5]</sup>, and focus on post-training unlearning after the federated model has been obtained. Instead of estimating curvature only from the forget-client batch, the study uses retained clients to build a small reference set and approximate the curvature around the retained task distribution. The intuition is that the forget-client gradient defines what should be removed, while the retained-set curvature defines how the model should remain close to the utility-preserving region of the parameter space.

However, retained-set curvature alone is not sufficient. Our experiments show that the retained-set update can preserve global utility but may point in a direction that decreases the forget-client loss. This is an important empirical finding: a method can appear safe under a global-loss guard while still failing the forgetting objective. To address this, we introduce a guarded update procedure with four components. First, global update norm clipping limits the magnitude of the second-order correction. Second, a stepped trust-region guard divides the update into smaller steps and rolls back a step if the global-loss guard is violated. Third, direction probing evaluates positive and negative update signs. Fourth, forget-loss guard rejects a step if it decreases the forget-client loss below a specified tolerance.

The evaluation is intentionally narrow and focused. A feasibility study verifies the end-to-end pipeline, a main trade-off experiment compares forget-batch and retained-set curvature, a direction-and-guard ablation explains the sign problem, and brief scale and seed checks test whether the interpretation survives outside the core setting. The goal is not to exhaust every regime, but to determine whether retained-set curvature can support a controllable post-training unlearning update.

The main contributions are summarized as follows:

1. A post-training federated-unlearning protocol is built on top of a fixed FedHDS-trained model, so training quality and unlearning behavior can be evaluated separately and the full pipeline can be verified end to end.
2. A guarded retained-set update is introduced that separates the forget-client gradient from the retained-client curvature reference and validates candidate steps through direction probing, global-loss guard, and forget-loss guard.
3. Experiments support a bounded claim: the method is better understood as a utility-preserving and auditable approximate unlearning framework than as a universally strongest forgetting method, and its forgetting gain remains scale- and seed-sensitive.

## 2. Related Work

### 2.1 Federated Unlearning

Federated unlearning extends machine unlearning to decentralized training, where a removal request must be handled without reconstructing the full centralized dataset. Recent survey work emphasizes that successful unlearning should be judged not only by whether target influence is weakened, but also by whether the remaining model still preserves acceptable utility<sup>[6]</sup>.

Existing federated-unlearning methods largely follow two directions. The first direction targets client-level removal. FedEraser approximates client removal by calibrating stored historical updates rather than retraining the federated model from scratch<sup>[7]</sup>. Rapid-retraining methods further reduce the cost of honoring the right to be forgotten by retraining only the affected part of the learning process<sup>[8]</sup>, while system-oriented formulations treat unlearning as a client right that must be supported by the federated protocol itself<sup>[9]</sup>. These studies make client removal more practical, but they still leave open how strongly the model can be updated before retained utility deteriorates.

The second direction studies finer forgetting granularity. Class-discriminative pruning removes class-specific influence by pruning parameters associated with the target class<sup>[10]</sup>. Active-forgetting approaches further use teacher-student memory generation to support class-wise federated unlearning without relying entirely on direct access to forgotten data<sup>[11]</sup>. Together, these studies broaden federated unlearning from client deletion alone to a design space involving removal granularity, computational efficiency, and post-unlearning utility.

### 2.2 Utility-aware Federated Unlearning and Second-order Methods

Machine unlearning outside the federated setting provides a useful reference point. Early formulations defined unlearning as removing the influence of selected data without rebuilding the entire system from scratch<sup>[2]</sup>, while SISA-style training reduced removal cost through structured retraining<sup>[12]</sup>. In federated settings, recent survey work organizes existing methods into retraining-based, update-correction, historical-update, knowledge-distillation, and second-order approximation families, and emphasizes that evaluation should jointly consider forgetting effectiveness and retained utility<sup>[13]</sup>. This tension directly motivates the present study.

A complementary line comes from second-order approximation. Influence functions show that data influence can be approximated through inverse-Hessian-vector products<sup>[3]</sup>. LiSSA-style stochastic second-order optimization offers a practical way to approximate such inverse-HVP directions without explicitly forming the Hessian<sup>[4]</sup>. Trust-region methods add a related optimization principle: parameter updates should be accepted only within a controlled region so that overly aggressive steps can be rejected<sup>[14]</sup>. Together, these ideas motivate curvature-informed unlearning updates that are evaluated not only by forgetting strength, but also by their effect on retained utility.

### 2.3 Position of This Work

Taken together, the literature suggests three requirements for federated unlearning in language-model fine-tuning: weakening the target-client influence, preserving retained-client utility, and remaining computationally feasible after training. Prior work provides practical client-removal schemes, finer-grained forgetting mechanisms, and curvature-aware updates, but it rarely separates the source of the forgetting signal from the source of the curvature estimate.

The present study addresses that gap by separating the forgetting signal from the curvature reference and by screening candidate updates with explicit norm and loss guards. The table below positions this design relative to representative federated-unlearning categories. The comparison is qualitative and is intended to clarify the design space rather than to provide a performance ranking.

Table 1. Qualitative comparison of federated unlearning approaches.

| Category | Client-level removal | Post-training | Second-order | Utility guard | Direction validation | Retrain-free |
|---|---|---|---|---|---|---|
| Full or structured retraining [2][12] | Yes | No | No | Implicit | No | No |
| Update-history calibration [7] | Yes | Yes | No | Partial | No | Yes |
| Federated rapid retraining [8] | Yes | No | No | Implicit | No | No |
| Second-order influence-style updates [3][4] | Typically data-level | Yes | Yes | No | No | Yes |
| Proposed method (guarded retained-set) | Yes | Yes | Yes | Yes | Yes | Yes |

Compared with the representative categories summarized above, the proposed method is distinguished by a post-training update, a second-order direction defined by retained-set curvature, and explicit guards for utility and forgetting validity.



## 3. Retained-Set Curvature Based Guarded Federated Unlearning Method

FedHDS first produces the pre-unlearning model θ*. The proposed method then applies a post-training unlearning update to that same starting point. The procedure has three phases: first, compute the forget-client gradient; second, estimate a second-order direction from retained-client curvature; and third, validate the update through clipping, direction probing, and guards.

### 3.1 Problem Definition and Optimization Objectives

Consider a federated system with K clients, where client k owns local data Dₖ. After federated training, the server holds a pre-unlearning model θ*. When a target client j requests removal, the algorithm returns an updated model θᵤ. The objective is to reduce the influence of client j while preserving utility on the retained-client distribution.

Formula (1). Forget-client loss change

ΔLⱼ = Lⱼ(θᵤ) - Lⱼ(θ*)

Positive ΔLⱼ indicates that the post-unlearning model performs worse on the forget client than the pre-unlearning model, which is the operational sign of forgetting used throughout the paper.

The utility constraint requires the post-unlearning global loss to remain within the allowed boundary `γᴳ`.

### 3.2 FedHDS Federated Fine-tuning Backbone

The federated training phase follows the FedHDS fine-tuning backbone introduced by Qin et al.<sup>[5]</sup>. FedHDS is treated as a fixed training backbone rather than as a contribution of this paper: it provides the shared pre-unlearning model θ*, and the proposed method operates only after training has completed. FL and FedHDS therefore act as training references, whereas forget-batch and retained-set Hessians are compared as post-training unlearning updates from the same starting point.

### 3.3 Forget-client Gradient as the Unlearning Signal

The forget client provides the target signal for unlearning. Rather than retraining the entire model, the server extracts a gradient from that client's data and treats it as the signal that should be weakened.

Formula (2). Forget-client gradient

gⱼ = ∇θ Lⱼ(θ*)

This gradient is not applied directly as a first-order update. It is later combined with a curvature estimate to form a second-order candidate direction.

### 3.4 Retained-client Reference Set and Curvature Estimation

The main design choice is to estimate curvature from retained clients rather than from the forget-client batch. This separates the forgetting signal from the utility-preserving geometry.

The retained reference set `Dref` is formed by taking a small retained sample set `Sₖ` from each retained client and unioning them across `k ≠ j`. In the main experiments, each retained client contributes one example, so with 20 total clients and one forget client, `|Dref| = 19`.

Formula (3). Retained curvature direction

v = Href⁻¹ gⱼ

Here Href is the approximate Hessian estimated from Dref, and v is the resulting second-order update direction. Relative to forget-batch Hessian, this choice is deliberately more conservative with respect to retained utility.

### 3.5 LiSSA/HVP Approximation for Inverse Curvature Direction

For a language model of this scale, the Hessian is too large to construct or invert explicitly. The method therefore relies on Hessian-vector products together with a lightweight LiSSA-style approximation. In the reported experiments, the recursion depth is 1 and the damping coefficient is 0.01, so the second-order routine acts as a candidate-direction generator rather than a trusted final update.

### 3.6 Update Scaling and Global Norm Clipping

The second-order direction is scaled before application so that its magnitude remains comparable across different retained-client counts and does not immediately violate the utility constraint.

The raw update scales `v` by `η / (K - 1)` before clipping and guard checking.

Formula (4). Global norm clipping

Δθclip = Δθraw · min(1, τ / ||Δθraw||)

If the raw update is already small, it is left unchanged; otherwise it is scaled down to the allowed norm. Reporting both τ and the realized update norm helps separate direction quality from step magnitude.

### 3.7 Stepped Trust-region Guard for Global Utility Preservation

The clipped update is partitioned into several smaller steps so that utility can be checked incrementally rather than only after the full update is applied.

The clipped update is divided into `S` guarded substeps so that utility can be checked incrementally. A substep is accepted only if the post-update global loss stays within the guard.

### 3.8 Direction Probing and Forget-loss Guard for Valid Forgetting

The preceding trade-off comparison suggests that retained-set curvature can preserve global utility but may weaken forgetting. The method therefore evaluates both update signs along the same clipped direction.

Formula (5). Direction candidates

θ⁺ = θ* + Δθclip

θ⁻ = θ* - Δθclip

The two candidates θ⁺ and θ⁻ are obtained from the same clipped direction Δθclip. Direction probing selects the sign that better satisfies the forgetting objective before the stepped guard is applied.

Formula (6). Forget-loss guard

Lⱼ(θₜ + δₛ) ≥ Lⱼᵍᵘᵃʳᵈ - ε

Here ε is a small numerical tolerance such as 0 or 0.005. A step therefore cannot be accepted merely because global loss is safe; it must also avoid moving the forget-client loss in the wrong direction.

Algorithm 1. Guarded retained-set unlearning.

1. Train the federated model and select the FedHDS checkpoint θ*.
2. Compute the forget-client gradient gⱼ on the client to be removed.
3. Build Dref from retained clients and approximate v with retained-set HVP/LiSSA.
4. Scale and clip the candidate update to obtain Δθclip.
5. Probe both signs of Δθclip and choose the sign that is compatible with the forgetting objective.
6. Apply the chosen update in S guarded substeps; accept a substep only if both the global-loss guard and the forget-loss guard pass.
7. Return the updated model together with update norm, accepted steps, and guard outcome.



## 4. Experimental Design and Evaluation Protocol

### 4.1 Instruction-tuning Dataset and Base Language Models

We use Dolly as the federated instruction-tuning dataset<sup>[15]</sup>. Qwen2-0.5B<sup>[16]</sup> is used for the feasibility validation, main trade-off comparison, and direction ablation because it is large enough to represent a language-model fine-tuning setting while still supporting repeated Hessian-vector product experiments under the available single-device compute budget. Qwen2.5-1.5B<sup>[17]</sup> is additionally used in the model-scale generalization study.

The federated setting uses 20 clients, a client fraction of 0.2, 2 local steps, batch size 1, maximum sequence length 64, and client 0 as the designated forget client. For all unlearning experiments, the forget-client gradient is computed from eight local training examples of the forget client. Retained-set Hessian experiments use one reference example per retained client, giving 19 retained reference examples when one of the 20 clients is forgotten. The reported second-order updates use LiSSA recursion depth 1, damping 0.01, and unlearning step size 0.01.

### 4.2 Experiment Groups and Robustness Check

For reproducibility, the archived experiment folders keep their original stage labels, but the paper organizes the evidence by experimental purpose rather than by chronology. The evaluation is divided into five groups, each designed to answer a specific question.

A feasibility group uses a data sampling ratio of 0.2 and 10 rounds. It checks whether FL, the FedHDS baseline, forget-batch Hessian unlearning, and retained-set Hessian with guard can all be evaluated consistently in a small-scale setting. Its purpose is to confirm that the full pipeline is executable and measurable before stronger comparative claims are made.

The main trade-off group increases the data sampling ratio to 0.4 and the number of rounds to 20. It provides the central comparison between forgetting effectiveness and retained utility. In this group, the retained-set method uses automatically calibrated global-loss guard thresholds relative to the FedHDS baseline pre-unlearning loss.

Direction-and-guard ablation keeps the main trade-off configuration and varies update sign, forget-loss guard, and negative-direction update norm. It diagnoses why retained-set curvature can preserve global utility while still failing the forgetting objective, and it tests whether a corrected direction can satisfy both constraints.

A model/data scale group forms a 2 x 2 matrix over Qwen2-0.5B versus Qwen2.5-1.5B and data sampling ratios of 0.4 versus 0.6. It tests whether the retained-set negative-direction update and guard behavior remain meaningful as model size and data volume increase.

Finally, the seed-robustness group repeats the large-model/large-data configuration with seeds 43 and 44. It identifies which conclusions are stable across seeds and which remain sensitive to the local training trajectory.



### 4.3 Compared Unlearning Settings

This study compares five settings: FL, the FedHDS baseline, FedHDS followed by forget-batch Hessian unlearning, FedHDS followed by retained-set Hessian with global-loss guard, and FedHDS followed by retained-set Hessian with negative-direction correction and forget-loss guard. Together, these settings isolate baseline training quality, the effect of the curvature reference, and the impact of sign correction under the same forgetting signal.

The forget-batch Hessian setting serves as the direct unlearning baseline because it uses the forget-client batch as both the gradient source and the Hessian reference. By contrast, the retained-set Hessian setting keeps the same forget-client gradient but replaces the Hessian reference with retained-client data, thereby isolating the curvature effect. The negative-direction update is introduced only after direction probing shows that the default sign can be misaligned with the forgetting objective.

### 4.4 Evaluation Metrics and Decision Criteria

Reported metrics include final global loss, forget-client loss before and after unlearning, update L2 norm, numbers of accepted and attempted guard steps, guard outcome, and unlearning time. Utility preservation is assessed by whether the post-unlearning global loss remains close to the FedHDS baseline pre-unlearning loss or stays below the specified global-loss guard. Directional forgetting effectiveness is assessed by whether the forget-client loss is prevented from decreasing and, ideally, is increased.

For the direction-and-guard ablation, interpretation focuses on the interaction among update sign, forget-loss guard, forget-client loss change, and global-loss guard behavior rather than on another FL-versus-baseline comparison. The scale study then evaluates the same retained-set negative-direction update under larger model and data settings, while the seed-robustness study separates conclusions that remain stable across seeds from those that are sensitive to the local training trajectory.

## 5. Experimental Results and Mechanism Analysis

This section presents the experimental evidence in the order of increasing complexity. It begins with a feasibility check of the complete training-and-unlearning pipeline, then examines the utility-forgetting trade-off induced by different curvature references, and finally studies how direction correction, guard design, model/data scale, and random seed affect the observed behavior.

### 5.1 Pipeline Feasibility and Training-Backbone Context

A small-scale setting is first used to verify that the complete training-and-unlearning pipeline can be executed and evaluated consistently. The goal is not to establish a definitive performance ranking, but to confirm that the proposed post-training unlearning procedure behaves measurably before moving to the main comparison setting.

Table 2. Minimal end-to-end feasibility results.

| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) |
|---|---|---|---:|---:|---:|---:|---:|
| FedHDS baseline | step size 1.0000 | none | N/A | 1.6657 | N/A | N/A | N/A |
| FL baseline | step size 1.0000 | none | N/A | 1.6443 | N/A | N/A | N/A |
| Forget-batch Hessian | step size 0.0100 | forget-batch | 2.3381 -> 2.3489 | 1.6867 | 0.0872 | 1/1 | 1.44 |
| Retained-set guarded update | step size 0.0100, maximum norm 0.300, 5 substeps, global-loss guard 1.700 | retained-set | 2.3381 -> 2.3460 | 1.6615 | 0.0427 | 5/5 | 71.13 |

The feasibility setting is not meant for strong performance claims. Its purpose is to show that both unlearning paths can be executed end to end under the same pipeline. In this small regime, forget-batch Hessian gives a slightly larger forgetting gain (`+0.0108`) but worse global loss (`1.6867`), whereas the retained-set guarded update produces a smaller update and keeps global loss lower (`1.6615`). This is enough to justify moving to the main trade-off setting.

### 5.2 Curvature Reference Trade-off

The main trade-off setting increases the data sampling ratio from 0.2 to 0.4 and the number of training rounds from 10 to 20, and serves as the primary testbed for the utility-forgetting trade-off.

Table 3. Main utility-forgetting trade-off results with retrain reference.

| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) |
|---|---|---|---:|---:|---:|---:|---:|
| FedHDS baseline | step size 1.0000 | none | N/A | 1.5250 | N/A | N/A | N/A |
| FL baseline | step size 1.0000 | none | N/A | 1.5441 | N/A | N/A | N/A |
| Retrain baseline | exclude client 0, learning rate 5e-5 | retrain-from-scratch | held-out client-0 loss = 2.2168 | 1.5355 | N/A | N/A | N/A |
| Forget-batch Hessian | step size 0.0100 | forget-batch | 2.2581 -> 2.2866 | 1.6257 | 0.5077 | 1/1 | 1.64 |
| Retained-set guarded update | step size 0.0100, maximum norm 0.300, 5 substeps, global-loss guard 1.575 | retained-set | 2.2581 -> 2.2329 | 1.5482 | 0.2486 | 5/5 | 131.13 |

Figure 1 plots the per-round convergence of FL and FedHDS.

![Figure 1. FL vs FedHDS convergence curves.](formal_cloud_results/stage2_dsample04_round20_20260503/figures/convergence_fl_vs_fedhds.png)

Figure 1. FL vs FedHDS training convergence (data sampling ratio 0.4, 20 rounds). FedHDS converges to a lower final global loss (1.5250) than FL (1.5441), making it the stronger utility baseline for subsequent unlearning experiments.

Under this stronger training configuration, FedHDS achieves final global loss `1.5250`, below FL's `1.5441`, and is therefore used as the utility baseline for the unlearning experiments. The retrain-from-scratch reference reaches global loss `1.5355` and held-out client-0 loss `2.2168`, showing that exact client exclusion on this IID split does not necessarily produce a higher held-out forget-client loss.

![Figure 2. Final global loss comparison in the main trade-off experiment.](formal_cloud_results/stage2_dsample04_round20_20260503/figures/global_loss_final.png)

Figure 2. Final global loss comparison in the main trade-off experiment.

The main result of Table 3 is the curvature-reference trade-off. Forget-batch Hessian increases forget-client loss from `2.2581` to `2.2866` (`+0.0285`), but raises global loss from `1.5250` to `1.6257` (`+0.1007`). Its update norm (`0.5077`) is also much larger than that of the retained-set guarded update.

Figure 3 shows the same trade-off from the forget-client side: the retained-set guarded update moves in the opposite direction.

![Figure 3. Forget-client loss before and after unlearning in the main trade-off experiment.](formal_cloud_results/stage2_dsample04_round20_20260503/figures/forget_loss_before_after.png)

Figure 3. Forget-client loss before and after unlearning in the main trade-off experiment.



By contrast, the retained-set guarded update keeps global loss close to the FedHDS baseline (`1.5482`, only `+0.0232`) with all `5/5` steps accepted, but decreases forget-client loss to `2.2329` (`-0.0252`). The main conclusion is therefore not a simple win: forget-batch is aligned with forgetting but utility-costly, whereas retained-set is utility-preserving but direction-misaligned. Section 5.3 focuses on that sign problem.

### 5.3 Direction Misalignment and Guard Validation

The direction-and-guard ablation explains the retained-set failure mode observed in the main trade-off experiment. The key issue is sign sensitivity: retained-set curvature contains a usable forgetting signal, but the default positive direction is not the direction that should be applied under a utility-preserving constraint.

Table 4. Direction-probing diagnosis: positive versus negative candidate under the same retained-set curvature.

| Candidate direction | Forget delta | Final global loss | Passes global-loss guard? |
|---|---|---|---|
| Positive (+Δθ) | -0.0073 | 1.5482 | Yes |
| Negative (-Δθ) | +0.5014 | 1.7310 | No |

Table 4 isolates the core mismatch. The positive-direction candidate stays within the global-loss guard but moves the forget-client loss in the wrong direction. The negative-direction candidate carries a strong forgetting signal, but its unconstrained utility cost is too large. This is the main mechanism finding of the paper: retained-set curvature is not useless, but it is sign-sensitive and must be validated before application.

When forget-loss guard is enabled, the positive-direction candidate is rejected immediately under both tolerances tested (`0` and `0.05`), leaving the model at the FedHDS baseline with `0/1` accepted steps. The guard therefore prevents a utility-preserving but forgetting-invalid update from being accepted.

Table 5. Negative-direction update-norm sweep.

| Retained-set negative-direction update | Maximum norm | Forget loss | Final global loss | Update L2 | Steps | Guard outcome |
| --- | --- | --- | --- | --- | --- | --- |
| Conservative setting | 0.0500 | 2.2581 -> 2.4105 (+0.1524) | 1.5561 | 0.0500 | 5/5 | all planned steps accepted |
| Main corrected setting | 0.1000 | 2.2581 -> 2.6104 (+0.3523) | 1.5935 | 0.1000 | 5/5 | all planned steps accepted |
| Aggressive setting | 0.1500 | 2.2581 -> 2.6880 (+0.4300) | 1.6094 | 0.1200 | 4/5 | stopped by global-loss guard |

Table 5 shows the correction result once the sign is flipped. The maximum-norm-`0.05` case is the most conservative and gives the smallest global-loss increase. The maximum-norm-`0.10` case provides the clearest forgetting gain while remaining within controlled utility degradation, and is therefore the preferred corrected setting. The maximum-norm-`0.15` case is useful mainly as a guard-stopping example: the stepped guard truncates the update at `4/5` steps before the full requested norm is applied. Additional norm-sweep plots are omitted from the main text because they do not change this conclusion.

Compared with the retrain baseline, the preferred negative-direction setting increases the forget-loss proxy much more strongly (`2.6104` versus retrain `2.2168`) but at a higher utility cost (`1.5935` versus `1.5355`). The method should therefore be read as a controllable post-training approximation rather than as a numerical surrogate for retraining.

### 5.4 Generalization across Model Scale and Data Scale

The scale study is kept compact because its role is confirmatory rather than exploratory. Table 5 summarizes the `2 x 2` matrix over model size and data sampling ratio using one representative retained-set negative-direction update (maximum norm `0.10`) and the forget-batch baseline.

Table 6. Model/data scale generalization matrix.

| Setting | Model | Data sampling ratio | FedHDS baseline global | Forget-batch Hessian | Retained-set negative-direction update (maximum norm 0.10) |
| --- | --- | --- | --- | --- | --- |
| Small model + small data | Qwen2-0.5B | 0.4 | 1.5250 | 2.2581 -> 2.2866 (+0.0285), G=1.6257 | 2.2581 -> 2.6104 (+0.3523), G=1.5935 |
| Small model + large data | Qwen2-0.5B | 0.6 | 1.5423 | 2.4377 -> 2.8059 (+0.3681), G=2.0040 | 2.4377 -> 2.6147 (+0.1770), G=1.5701 |
| Large model + small data | Qwen2.5-1.5B | 0.4 | 1.6438 | 1.8812 -> 1.8702 (-0.0110), G=1.6408 | 1.8812 -> 1.9206 (+0.0395), G=1.6405 |
| Large model + large data | Qwen2.5-1.5B | 0.6 | 1.6201 | 2.0880 -> 2.0828 (-0.0052), G=1.6291 | 2.0880 -> 2.1078 (+0.0198), G=1.6132 |

Table 6 preserves the main pattern without repeating every setting-specific detail. In the small-model/large-data regime, forget-batch Hessian still gives the stronger forgetting gain, but it does so with a much larger utility cost (`G=2.0040` versus `G=1.5701`). In the larger-model regimes, forget-batch Hessian no longer reliably increases forget-client loss, whereas the retained-set negative-direction update remains non-negative on forgetting and keeps global loss close to or below the FedHDS baseline. Detailed per-setting tables are omitted because they do not change this scale-level conclusion.

### 5.5 Seed Robustness under the Large-model/Large-data Setting

Seed robustness is treated as boundary evidence rather than as a full new result section. Under the same large-model/large-data configuration, seed 42 reproduces a small positive forgetting gain, seed 43 remains near-neutral on forgetting while preserving utility, and seed 44 is rejected by the forget-loss guard at `0/1` steps. The stable conclusion across seeds is therefore global-loss control rather than guaranteed forgetting strength. Detailed per-seed tables are omitted from the main text for brevity.

## 6. Limitations

This study remains limited to Dolly, the Qwen/Qwen2.5 family, one federated configuration, and a single designated forget client. Forget-client loss is still an operational proxy rather than a gold-standard forgetting metric, and the seed check shows that exact forgetting gain is trajectory-sensitive. Future work should test broader datasets, multi-client removal, and cheaper retained-set HVP implementations.

## 7. Conclusion

This paper studies post-training federated unlearning on top of a fixed FedHDS checkpoint. First, it establishes a protocol that isolates unlearning behavior from federated training quality and verifies that the full pipeline can be measured end to end. Second, it shows that retained-set curvature becomes useful only after direction validation and guard control, because the main difficulty is not the absence of a forgetting signal but sign-sensitive mismatch under utility constraints. Third, the scale and seed checks support a bounded claim: the method is better understood as a utility-preserving and auditable approximate unlearning framework than as a universally strongest forgetting method.

## References
[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," Proceedings of AISTATS, 2017. https://arxiv.org/abs/1602.05629

[2] Y. Cao and J. Yang, "Towards Making Systems Forget with Machine Unlearning," IEEE Symposium on Security and Privacy, pp. 463-480, 2015. https://doi.org/10.1109/SP.2015.35

[3] P. W. Koh and P. Liang, "Understanding Black-box Predictions via Influence Functions," Proceedings of ICML, PMLR 70:1885-1894, 2017. https://proceedings.mlr.press/v70/koh17a.html

[4] N. Agarwal, B. Bullins, and E. Hazan, "Second-order Stochastic Optimization for Machine Learning in Linear Time," Journal of Machine Learning Research, vol. 18, pp. 1-40, 2017. https://www.jmlr.org/papers/v18/16-491.html

[5] Z. Qin, Z. Wu, B. He, and S. Deng, "Federated Data-Efficient Instruction Tuning for Large Language Models," arXiv:2410.10926, 2024. https://arxiv.org/abs/2410.10926

[6] Z. Li, X. Meng, L. Wang, and X. Hao, "Survey on Machine Unlearning," Journal of Software, vol. 36, no. 4, pp. 1637-1664, 2025. https://doi.org/10.13328/j.cnki.jos.007237

[7] G. Liu, Y. Yang, X. Ma, C. Wang, and J. Liu, "Federated Unlearning," CoRR abs/2012.13891, 2020. https://arxiv.org/abs/2012.13891

[8] Y. Liu, L. Xu, X. Yuan, C. Wang, and B. Li, "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining," IEEE INFOCOM, pp. 1749-1758, 2022. https://doi.org/10.1109/INFOCOM48880.2022.9796721

[9] L. Wu, S. Guo, J. Wang, Z. Hong, J. Zhang, and Y. Ding, "Federated Unlearning: Guarantee the Right of Clients to Forget," IEEE Network, vol. 36, no. 5, pp. 129-135, 2022. https://doi.org/10.1109/MNET.001.2200198

[10] J. Wang, S. Guo, X. Xie, and H. Qi, "Federated Unlearning via Class-Discriminative Pruning," Proceedings of the ACM Web Conference 2022, pp. 622-632, 2022. https://doi.org/10.1145/3485447.3512222

[11] Y. Li, J. Zhang, Y. Liu, and C. Chen, "Class-wise Federated Unlearning: Harnessing Active Forgetting with Teacher-Student Memory Generation," Knowledge-Based Systems, vol. 316, article 113353, 2025. https://doi.org/10.1016/j.knosys.2025.113353

[12] L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, and N. Papernot, "Machine Unlearning," IEEE Symposium on Security and Privacy, pp. 141-159, 2021. https://doi.org/10.1109/SP40001.2021.00019

[13] N. Romandini, A. Mora, C. Mazzocca, R. Montanari, and P. Bellavista, "Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics," IEEE Transactions on Neural Networks and Learning Systems, 2025. https://doi.org/10.1109/TNNLS.2024.3478334

[14] A. R. Conn, N. I. M. Gould, and P. L. Toint, Trust Region Methods. SIAM, 2000. https://doi.org/10.1137/1.9780898719857

[15] M. Conover, M. Hayes, A. Mathur, J. Xie, J. Wan, S. Shah, A. Ghodsi, P. Wendell, M. Zaharia, and R. Xin, "Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM," Databricks Blog, 2023. https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm

[16] A. Yang et al., "Qwen2 Technical Report," arXiv:2407.10671, 2024. https://arxiv.org/abs/2407.10671

[17] Qwen Team et al., "Qwen2.5 Technical Report," arXiv:2412.15115, 2024. https://arxiv.org/abs/2412.15115

## Appendix A. Supplementary Figures


