# Guarded Federated Unlearning via Retained-Set Curvature

## Abstract

Federated unlearning seeks to reduce the influence of a target client without fully retraining the federated model. In language-model federated instruction tuning, this problem is challenging because approximate unlearning must weaken the target-client signal while preserving utility on retained clients. This paper studies a guarded post-training federated unlearning method built on the FedHDS training backbone. We adopt FedHDS as the federated fine-tuning backbone and contribute the unlearning stage rather than a new training algorithm. The method uses the target-client gradient to define the signal to be weakened, estimates curvature from a small retained-client reference set, and constrains the resulting second-order update through global norm clipping, stepped trust-region checking, direction probing, and a forget-loss guard. Experiments on Dolly with Qwen2-0.5B and Qwen2.5-1.5B show that forget-batch Hessian updates can produce stronger target-client loss increases but may incur substantial global-loss degradation. In contrast, retained-set curvature, when combined with direction correction and guard constraints, can improve target-client loss under controlled global loss. Scale and seed checks further indicate that the method is best interpreted as an auditable, utility-preserving approximate unlearning framework rather than a universally strongest forgetting algorithm; the exact forgetting gain remains sensitive to model scale, data scale, and random seed.

**Keywords:** federated unlearning; retained-set curvature; federated instruction tuning; large language models; utility-preserving unlearning

## 1. Introduction

Federated learning allows multiple clients to collaboratively train a global model without directly sharing their local data<sup>[1]</sup>. This training paradigm is useful when data are distributed across users, institutions, or devices, and when centralizing the raw data is undesirable. However, a trained federated model may later face an unlearning request: a client may withdraw consent, request data deletion, or become invalid due to data quality or compliance concerns. In such a case, the system should remove or reduce the influence of that client from the global model.

The most direct solution is to retrain the federated model after excluding the target client<sup>[2]</sup>. Although retraining provides a clean reference, it is often impractical. Federated training may involve many communication rounds, repeated local fine-tuning, and large model checkpoints. The cost becomes even more pronounced for language-model fine-tuning, where even one full run can already require substantial computing resources and time. Federated unlearning therefore studies how to approximate the effect of removing a client without performing full retraining.

Approximate federated unlearning is difficult because it has two competing requirements. On the one hand, the update should weaken the forgotten client's influence. In this paper, this is reflected by the forget-client loss: after unlearning, the loss on the forgotten client should not decrease and preferably should increase. On the other hand, the update should preserve the utility of the model on the retained data distribution. If the update increases forget loss but severely damages global performance, the method is not practically useful.

A central challenge in federated unlearning is the trade-off between forgetting and retained utility. An aggressive update may increase the loss on the forget client, but it can also damage performance on the retained clients. Conversely, an overly conservative update may preserve global utility but fail to meaningfully reduce the forgotten client's influence. This trade-off is especially important when second-order approximations are used: Hessian or inverse-Hessian-vector-product based updates can estimate parameter corrections for removing data influence<sup>[3][4]</sup>, but the choice of curvature reference directly affects both the direction and magnitude of the unlearning update.

Existing second-order unlearning approximations often use a gradient from the target data together with a Hessian or inverse-Hessian-vector product. A natural baseline is to compute both the gradient and curvature reference from the forget-client batch. This forget-batch Hessian approach is directly aligned with the forget objective, but it may move the model in a way that is harmful to retained utility. In a federated setting, where the post-unlearning model should still serve the remaining clients, this is a serious limitation.

This work adopts FedHDS, the federated data-efficient instruction-tuning backbone introduced by Qin et al.<sup>[17]</sup>, as the training stage and studies how to perform controlled post-training unlearning on top of it. We therefore do not claim the training backbone as our contribution; our contribution begins after a FedHDS-trained global model is obtained. Instead of estimating curvature only from the forget-client batch, we use retained clients to build a small reference set and approximate the curvature around the retained task distribution. The intuition is that the forget-client gradient defines what should be removed, while the retained-set curvature defines how the model should remain close to the utility-preserving region of the parameter space.

However, retained-set curvature alone is not sufficient. Our experiments show that the retained-set update can preserve global utility but may point in a direction that decreases the forget-client loss. This is an important empirical finding: a method can appear safe under a global utility guard while still failing the forgetting objective. To address this, we introduce a guarded update procedure with four components. First, global update norm clipping limits the magnitude of the second-order correction. Second, a stepped trust-region guard divides the update into smaller steps and rolls back a step if global loss exceeds a threshold. Third, direction probing evaluates positive and negative update signs. Fourth, forget-loss guard rejects a step if it decreases the forget-client loss below a specified tolerance.

We evaluate the method through five purpose-oriented experiment groups: feasibility validation, the main utility-forgetting trade-off comparison, direction and guard ablation, model/data scale generalization, and seed robustness. The feasibility validation establishes that the complete training-unlearning procedure can be evaluated coherently under a small-scale setting. The trade-off comparison shows that forget-batch Hessian gives a clearer forgetting signal but increases global loss substantially, whereas retained-set guard keeps global loss close to the FedHDS baseline but exposes an update-direction problem. The ablation then diagnoses this direction problem and shows that negative-sign correction with forget-loss guard can increase forget-client loss while keeping global loss controlled. The scale study evaluates model/data generalization through a 2 x 2 matrix. The seed robustness check reruns the large-model/large-data core settings to show that utility safety and guard behavior are more stable than the exact forgetting gain.

The main contributions are:

- We build a complete post-training federated unlearning evaluation framework on top of a FedHDS-trained global model, enabling controlled comparison across forgetting, utility, and guard behavior.
- We introduce retained-set curvature approximation for federated unlearning, separating the forget-client gradient from the retained-client curvature reference.
- We stabilize second-order unlearning with global update norm clipping and a stepped trust-region guard, making updates measurable, reversible, and utility-aware.
- We diagnose retained-set update-direction misalignment and show that forget-loss guard with negative-sign correction can increase forget-client loss while keeping global loss controlled.
- We provide a cautious empirical interpretation: retained-set curvature is useful for utility preservation, but it requires direction validation and should not be presented as universally stronger than forget-batch Hessian.
- We add a 2 x 2 model/data scale validation showing that the retained-set negative update remains a controllable trade-off across small/large model and small/large data settings.
- We add a seed robustness check on the large-model/large-data setting, showing stable utility control and meaningful guard behavior while acknowledging that exact forgetting gain is seed-sensitive.

## 2. Research Status in China and Abroad

### 2.1 Research Status in China

Chinese research on federated unlearning is closely connected to the practical need for privacy protection, data deletion, and controllable model maintenance in federated systems. On the broader machine-unlearning side, Chinese survey work has summarized the motivation, technical routes, and evaluation challenges of machine unlearning, emphasizing that unlearning should not only remove target-data influence but also preserve the utility of the remaining model<sup>[5]</sup>.

In federated learning, one representative line of work is efficient client-level removal. FedEraser accelerates federated unlearning by calibrating historical updates instead of fully retraining the federated model<sup>[6]</sup>. Rapid-retraining methods further study how to realize the right to be forgotten in federated learning with lower retraining cost<sup>[7]</sup>. Other work formulates federated unlearning as a client right and discusses the system-level requirements needed to guarantee that a client can be removed from the trained model<sup>[8]</sup>.

Another line focuses on more fine-grained class-level or knowledge-level removal. Class-discriminative pruning removes class-specific influence by pruning parameters related to the target class<sup>[9]</sup>. Active-forgetting methods use teacher-student memory generation to guide class-wise federated unlearning and reduce the dependence on direct access to forgotten data<sup>[10]</sup>. These studies show that domestic and Chinese-author research has moved from "whether a client can be removed" toward more detailed questions of what granularity should be removed, how much utility can be preserved, and how the forgetting effect should be evaluated.

### 2.2 Research Status Abroad

International research provides the general theoretical and methodological foundation for this problem. FedAvg established the standard server-client aggregation framework for federated learning<sup>[1]</sup>. Machine unlearning was originally framed as the problem of making a trained system forget the influence of selected data without rebuilding the whole system from scratch<sup>[2]</sup>. Later SISA-style training made unlearning more efficient by structuring training into shards and slices, so that only affected components need to be retrained<sup>[11]</sup>.

For federated unlearning, recent survey work classifies existing methods into retraining-based, update-correction, historical-update, knowledge-distillation, and second-order approximation families, and points out that evaluation should consider both forgetting effect and retained utility<sup>[12]</sup>. This is consistent with the central tension studied in this paper: a strong forgetting update may damage the global model, while an overly conservative update may preserve utility but fail to remove the target-client influence.

Second-order approximation is another important international line. Influence functions show how training-point influence can be approximated through inverse-Hessian-vector products. LiSSA-style stochastic second-order optimization provides a practical route for approximating inverse-HVPs without explicitly constructing the Hessian. Trust-region methods offer the optimization idea of limiting parameter movement and accepting an update only within a controlled region<sup>[13]</sup>. These foundations motivate the retained-set curvature and guarded update design used in this paper.

### 2.3 Summary and Motivation of This Work

Existing work shows that federated unlearning must handle three coupled requirements: removing target-client influence, preserving retained-client utility, and keeping the update process computationally feasible. Domestic and Chinese-author studies provide efficient client removal, rapid retraining, and class-level forgetting mechanisms, while international studies provide broader unlearning theory, influence approximation, and trust-region style control. However, most existing formulations do not explicitly separate the source of the forgetting signal from the source of the curvature reference in language-model federated fine-tuning.

This paper is motivated by that gap. We adopt FedHDS as the training backbone and focus on the post-training unlearning problem that remains after the federated model has been obtained. We use the forget-client gradient to define what should be weakened, but estimate curvature from retained-client reference data to describe where the model should remain useful. The resulting update is not applied blindly. It is constrained by norm clipping, direction probing, stepped global-utility checking, and forget-loss guard. In this way, the proposed method connects existing federated unlearning and second-order approximation ideas to a more auditable utility-preserving unlearning workflow for language-model fine-tuning.

The table below positions the proposed approach against representative federated unlearning categories. The comparison is qualitative and meant to clarify the design space rather than to rank methods by performance.

**Table. Qualitative comparison of federated unlearning approaches.**

| Category | Client-level removal | Post-training | Second-order | Utility guard | Direction validation | Retrain-free |
|---|---|---|---|---|---|---|
| Retraining-based [2][11] | Yes | No | No | Implicit | No | No |
| History-calibration / FedEraser-like [6][7] | Yes | Yes | No | Partial | No | Yes |
| Second-order approximation [3][4] | Data-level | Yes | Yes | No | No | Yes |
| Ours (guarded retained-set) | Yes | Yes | Yes | Yes | Yes | Yes |

The main structural difference is that our method explicitly separates the forgetting signal source (forget-client gradient) from the curvature reference (retained-client Hessian), and adds guard mechanisms that make the update auditable: each step records whether it was accepted or rejected, and if rejected, whether the cause was global utility violation or forgetting objective failure. No other category in the table simultaneously provides second-order approximation, explicit utility guard, and direction validation in a retrain-free federated unlearning setting.



## 3. Retained-Set Curvature Based Guarded Federated Unlearning Method

FedHDS first produces the pre-unlearning model θ*. The proposed method then applies a post-training unlearning update to that same starting point. The procedure has three phases: (1) compute the forget-client gradient, (2) estimate a second-order direction from retained-client curvature, and (3) validate the update through clipping, direction probing, and guards.

### 3.1 Problem Definition and Optimization Objectives

Assume a federated system with K clients, where client k owns local data Dₖ. After federated training, the server obtains θ*. A target client j then requests unlearning, and the algorithm returns θᵤ. The goal is to weaken client j while preserving utility on retained clients.

**Formula (1). Forget-client loss change**

ΔLⱼ = Lⱼ(θᵤ) - Lⱼ(θ*)

Variable explanation:

- ΔLⱼ: change in forget-client loss after unlearning.
- Lⱼ(θ*): average loss on forget client j before unlearning.
- Lⱼ(θᵤ): average loss on forget client j after unlearning.

Interpretation: ΔLⱼ > 0 is the desired direction, while ΔLⱼ < 0 means the update moved in the wrong direction. This is an auditable loss-based forgetting proxy rather than a proof of complete information removal, so it is always interpreted together with global loss.

**Formula (2). Global utility guard**

Lᴳ(θᵤ) ≤ γᴳ

Variable explanation:

- Lᴳ(θᵤ): global evaluation loss after unlearning.
- γᴳ: maximum allowed global loss after unlearning.

Interpretation: γᴳ defines the utility boundary. An update is accepted only when post-unlearning global loss stays within that boundary.

### 3.2 FedHDS Federated Fine-tuning Backbone

The training stage uses the FedHDS federated fine-tuning backbone introduced by Qin et al.<sup>[17]</sup>. FedHDS is adopted rather than contributed here: it produces θ*, and the proposed module modifies θ* only after training finishes. Accordingly, FL and FedHDS are training baselines, while forget-batch Hessian and retained-set Hessian are post-training unlearning baselines. The comparison therefore asks which update best weakens client j while preserving retained utility from the same trained starting point.

### 3.3 Forget-client Gradient as the Unlearning Signal

The forget client defines the signal to remove. The server computes an average gradient on that client's data.

**Formula (3). Forget-client gradient**

gⱼ = ∇θ Lⱼ(θ*)

Variable explanation:

- gⱼ: gradient signal from the client that should be forgotten.
- j: index of the forget client.
- Lⱼ(θ*): loss on the forget client's data before unlearning.

Interpretation: gⱼ identifies the local signal associated with the forgotten client. It is not used as a plain first-order update; it is combined with a curvature estimate.

### 3.4 Retained-client Reference Set and Curvature Estimation

The main design choice is to estimate curvature from retained clients rather than from the forget-client batch. This separates "what to remove" from "where the remaining model should stay."

**Formula (4). Retained reference set**

Dref = ⋃(k≠j) Sₖ,  Sₖ ⊂ Dₖ

Variable explanation:

- Dref: retained reference set used for curvature estimation.
- Sₖ: reference subset sampled from retained client k.
- Dₖ: local dataset of client k.
- j: the forget client.
- retained clients: all clients whose data should remain useful after unlearning.

Formula (4) means that the reference set is built only from retained clients and is intentionally small. In the main experiments, each retained client contributes one example, so with 20 total clients and one forget client, |Dref| = 19. This keeps HVP cost feasible while preserving the distinction between forgetting signal and retained-utility geometry.

**Formula (5). Retained curvature direction**

v = Href⁻¹ gⱼ

Variable explanation:

- Href: approximate Hessian estimated from Dref.
- gⱼ: gradient from the forget client.
- v: second-order direction used for unlearning.

Interpretation: the forget-client gradient defines the signal to remove, while retained-set curvature constrains movement toward a utility-preserving region. Relative to forget-batch Hessian, this is more conservative with respect to retained utility, although later experiments show that sign validation is still necessary.

### 3.5 LiSSA/HVP Approximation for Inverse Curvature Direction

For a language model, the Hessian cannot be explicitly built or inverted. The implementation therefore uses Hessian-vector products and a lightweight LiSSA-style approximation.

**Formula (6). Approximate inverse-HVP recursion**

vₜ₊₁ = gⱼ + (1 - λ)vₜ - Href vₜ

Variable explanation:

- vₜ: current approximation vector.
- vₜ₊₁: next approximation vector.
- Href vₜ: Hessian-vector product computed using the retained reference loss.
- λ: damping coefficient in the LiSSA-style recursion.

The reported experiments use recursion depth 1 and damping 0.01. This conservative setting keeps computation feasible and treats the second-order routine as a candidate-direction generator; the guards decide whether that direction is acceptable.

### 3.6 Update Scaling and Global Norm Clipping

The raw second-order direction is scaled before being applied.

**Formula (7). Raw unlearning update**

Δθraw = ηv / (K - 1)

Variable explanation:

- η: unlearning step size, set to 0.01 in the reported runs.
- v: inverse-HVP direction from Formula (5).
- K - 1: number of retained clients.
- Δθraw: update before clipping and guard checking.

The raw update may be too large. Therefore, the implementation clips its global L2 norm.

**Formula (8). Global norm clipping**

Δθclip = Δθraw · min(1, τ / ||Δθraw||)

Variable explanation:

- ||Δθraw||: global L2 norm of the raw update across trainable tensors.
- τ: maximum allowed update norm.
- Δθclip: update after norm control.

Interpretation: if the raw update is already small, it is unchanged; otherwise it is scaled down to the allowed norm. Reporting both τ and the realized update norm separates direction quality from step size.

### 3.7 Stepped Trust-region Guard for Global Utility Preservation

After clipping, the update is split into several smaller steps.

**Formula (9). Step update**

δₛ = Δθclip / S

Variable explanation:

- S: number of guarded substeps.
- δₛ: one small update applied during guard checking.

At each step, the algorithm temporarily applies δₛ, measures global loss, and accepts the step only if the global guard is satisfied.

**Formula (10). Step acceptance rule**

Lᴳ(θₜ + δₛ) ≤ γᴳ

Variable explanation:

- θₜ: model parameter before the current guarded step.
- δₛ: candidate substep.
- Lᴳ(θₜ + δₛ): global loss after temporarily applying the candidate substep.
- γᴳ: global-loss acceptance boundary.

Interpretation: if the inequality is satisfied, the candidate substep is accepted; otherwise it is rolled back and the procedure stops. The accepted/attempted step count therefore becomes a direct diagnostic signal for how close the update is to the utility boundary.

### 3.8 Direction Probing and Forget-loss Guard for Valid Forgetting

The main trade-off comparison showed that retained-set curvature can preserve global utility but may reduce forget-client loss. The subsequent ablation therefore probes both update signs.

**Formula (11). Direction candidates**

θ⁺ = θ* + Δθclip

θ⁻ = θ* - Δθclip

Variable explanation:

- θ⁺: positive update candidate.
- θ⁻: negative update candidate.
- θ*: parameter before unlearning.
- Δθclip: clipped update direction.

The algorithm chooses the direction that gives the larger forget-client loss increase while staying compatible with the global guard when possible. The final protection is forget-loss guard.

**Formula (12). Forget-loss guard**

Lⱼ(θₜ + δₛ) ≥ Lⱼᵍᵘᵃʳᵈ - ε

Variable explanation:

- θₜ: model parameter before the current guarded step.
- Lⱼᵍᵘᵃʳᵈ: baseline forget-client loss used by the guard.
- ε: allowed small numerical tolerance, such as 0 or 0.005.

Interpretation: a step cannot be accepted merely because global loss is safe; it must also avoid moving the forget-client loss in the wrong direction. Combining direction probing, global guard, and forget-loss guard yields the final decision rule. A rejected substep is therefore interpretable as either "too damaging" or "not forgetting."



## 4. Experimental Design and Evaluation Protocol

### 4.1 Instruction-tuning Dataset and Base Language Models

Experiments use Dolly as the federated instruction-tuning dataset<sup>[14]</sup>. The feasibility validation, main trade-off comparison, and direction ablation use Qwen2-0.5B<sup>[15]</sup> because it is large enough to represent a language-model fine-tuning setting while still compatible with repeated Hessian-vector product experiments under the available single-device compute budget. The model-scale generalization study additionally uses Qwen2.5-1.5B<sup>[16]</sup>. The federated setting uses 20 clients, a client fraction of 0.2, 2 local steps, batch size 1, maximum sequence length 64, and client 0 as the designated forget client.

For all unlearning experiments, the forget-client gradient is computed from the forget client's local training data with `unlearn_grad_sample_size=8`. Retained-set Hessian experiments use `hessian_ref_per_client=1`, so each retained client contributes one reference example. This gives 19 retained reference examples when one of the 20 clients is forgotten. We use `lissa_depth=1`, `lissa_damping=0.01`, and `unlearn_eta=0.01` for the reported second-order updates.

### 4.2 Experiment Groups and Robustness Check

The raw experiment folders retain stage labels for reproducibility, but the paper reports the results by experimental purpose rather than as a chronological log. The experiments are organized into five groups.

The first group is a feasibility validation with a data sampling ratio of 0.2 and 10 rounds. It verifies that FL, the FedHDS baseline, forget-batch Hessian unlearning, and retained-set Hessian with guard can all be assessed consistently under a small-scale setting. Its goal is feasibility rather than exhaustive comparison.

The second group is the main utility-forgetting trade-off comparison with a data sampling ratio of 0.4 and 20 rounds. It tests whether the method remains stable under a larger setting and provides the main trade-off evidence. In this group, the retained-set method uses automatically calibrated global-guard thresholds, where the guard boundary is set relative to the FedHDS baseline pre-unlearning loss.

The third group is a full direction and guard ablation. It keeps the main trade-off configuration and tests update sign, forget-loss guard, and negative-sign norm sweep. Its purpose is to diagnose why the retained-set update preserves global utility but decreases forget-client loss, and to verify whether a corrected direction can satisfy both forgetting and utility constraints.

The fourth group is a model/data scale generalization study. It forms a 2 x 2 matrix over Qwen2-0.5B versus Qwen2.5-1.5B and data sampling ratios of 0.4 versus 0.6. This group tests whether the retained-set negative direction and guard behavior remain meaningful when model size and data sample increase.

The fifth group is a seed robustness check. It repeats the large-model/large-data core-five settings with seed 43 and seed 44. Its purpose is to test whether the seed 42 large-model/large-data result is an isolated accident and to identify which claims are stable across seeds.



### 4.3 Compared Unlearning Settings

We compare:

- FL.
- FedHDS baseline.
- FedHDS baseline + forget-batch Hessian unlearning.
- FedHDS baseline + retained-set Hessian + global guard.
- FedHDS baseline + retained-set Hessian + negative sign + forget-loss guard.

The forget-batch Hessian setting serves as the direct unlearning baseline. It uses the forget-client batch as both the gradient source and the Hessian reference. The retained-set Hessian setting keeps the forget-client gradient but changes the Hessian reference to retained clients. This isolates the effect of the curvature reference. The negative-sign setting is used only after direction probing shows that the original sign is misaligned with the forget objective.

### 4.4 Evaluation Metrics and Decision Criteria

We report final global loss, forget-client loss before and after unlearning, update L2 norm, accepted and attempted guard steps, guard outcome, and unlearning time. A method is considered utility-preserving when the post-unlearning global loss stays close to the FedHDS baseline pre-unlearning loss or below the specified global guard. A method is considered directionally effective for forgetting when the forget-client loss does not decrease, and preferably increases.

Because the direction/guard ablation is complete, its main evidence is not a new FL/training-baseline comparison, but the relationship among update sign, forget-loss guard, forget-client loss change, and global guard behavior. The scale study then tests the same core retained-negative configuration under larger model and data settings. The seed robustness check further separates two conclusions: global utility control and guard behavior are relatively stable, while the magnitude and sign of the forgetting gain can depend on the random seed.

## 5. Experimental Results and Mechanism Analysis

This section is organized by question rather than by chronology. We first ask whether the complete procedure is feasible and whether FedHDS serves as a reasonable training backbone. We then examine how the curvature reference changes the forgetting-utility trade-off, why the retained-set update can fail, how direction probing and guards correct that failure, whether the corrected path remains meaningful across model/data scales, and where the seed-level boundary lies.

### 5.1 Pipeline Feasibility and Training-Backbone Context

This experiment serves as a small-scale feasibility validation. It is not intended as a definitive performance comparison; instead, it establishes that the training and post-training unlearning procedure can be evaluated stably before moving to the main comparison setting.

Table 1 reports the feasibility validation results.

**Table 1. Minimal end-to-end feasibility results.**

| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) |
|---|---|---|---:|---:|---:|---:|---:|
| FedHDS baseline | step size 1.0000 | none | - | 1.6657 | - | - | - |
| FL baseline | step size 1.0000 | none | - | 1.6443 | - | - | - |
| Forget-batch Hessian | step size 0.0100 | forget-batch | 2.3381 -> 2.3489 | 1.6867 | 0.0872 | 1/1 | 1.44 |
| Retained-set guard | step size 0.0100, maximum norm 0.300, 5 substeps, global guard 1.700 | retained-set | 2.3381 -> 2.3460 | 1.6615 | 0.0427 | 5/5 | 71.13 |

FL obtains final global loss 1.6443, while FedHDS obtains 1.6657. The small difference between FL and FedHDS in this minimal setting is not the main focus, because it uses only a data sampling ratio of 0.2 and 10 rounds. The more important result is that both unlearning paths complete successfully.

The forget-batch Hessian baseline increases forget-client loss from 2.3381 to 2.3489, giving a forget-loss gain of +0.0108. Its final global loss is 1.6867, which is higher than the FedHDS pre-unlearning baseline. The retained-set guard also increases forget-client loss, from 2.3381 to 2.3460, giving a smaller gain of +0.0079. However, its final global loss is 1.6615, slightly lower than the FedHDS final global loss in the same feasibility setting.

This feasibility validation supports two preliminary observations. First, second-order unlearning can be integrated into the adopted FedHDS-based training-and-unlearning framework. Second, the retained-set guard is more conservative in update magnitude and more stable in global utility, although its forgetting gain is weaker than the forget-batch baseline.

### 5.2 Curvature Reference Trade-off

The main trade-off experiment increases the data sampling ratio from 0.2 to 0.4 and the number of training rounds from 10 to 20. It is used as the main evidence for the utility-forgetting trade-off.

Table 2 reports the main trade-off results.

**Table 2. Main utility-forgetting trade-off results with retrain reference.**

| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) |
|---|---|---|---:|---:|---:|---:|---:|
| FedHDS baseline | step size 1.0000 | none | - | 1.5250 | - | - | - |
| FL baseline | step size 1.0000 | none | - | 1.5441 | - | - | - |
| Retrain baseline | exclude client 0, learning rate 5e-5 | retrain-from-scratch | held-out client-0 loss = 2.2168 | 1.5355 | - | - | - |
| Forget-batch Hessian | step size 0.0100 | forget-batch | 2.2581 -> 2.2866 | 1.6257 | 0.5077 | 1/1 | 1.64 |
| Retained-set auto guard | step size 0.0100, maximum norm 0.300, 5 substeps, global guard 1.575 | retained-set | 2.2581 -> 2.2329 | 1.5482 | 0.2486 | 5/5 | 131.13 |

Figure 1 shows the per-round convergence of FL and FedHDS.

![Figure 1. FL vs FedHDS convergence curves.](formal_cloud_results/stage2_dsample04_round20_20260503/figures/convergence_fl_vs_fedhds.png)

*Figure 1. FL vs FedHDS training convergence (data sampling ratio 0.4, 20 rounds). FedHDS converges to a lower final global loss (1.5250) than FL (1.5441), making it the stronger utility baseline for subsequent unlearning experiments.*

FedHDS achieves final global loss 1.5250, lower than FL's 1.5441. As shown in Figure 2, this difference is visually clear in the final global loss comparison.

Table 2 also includes a retrain-from-scratch reference. Excluding client 0 and retraining from scratch yields final global loss 1.5355 and held-out client-0 loss 2.2168. Relative to the FedHDS baseline, the global-loss change is only +0.0105, which is far smaller than the forget-batch change (+0.1007) and also smaller than the corrected retained-negative result with a maximum norm of 0.10 discussed later in Section 5.3 (+0.0685).

![Figure 2. Final global loss comparison in the main trade-off experiment.](formal_cloud_results/stage2_dsample04_round20_20260503/figures/global_loss_final.png)

*Figure 2. Final global loss comparison in the main trade-off experiment.*

This is an important difference from the feasibility validation: with more training rounds and a larger data sample, FedHDS becomes the stronger utility baseline. This supports using FedHDS as the backbone for the unlearning experiments.

The forget-batch Hessian baseline gives the clearest forgetting signal in the main trade-off experiment. It increases forget-client loss from 2.2581 to 2.2866, a gain of +0.0285. However, this comes with a substantial global utility cost: global loss increases from 1.5250 to 1.6257, a degradation of +0.1007. The actual update L2 norm is 0.5077, much larger than the retained-set guarded update. This result establishes the central trade-off: forget-batch Hessian aligns more directly with forgetting, but it is less constrained with respect to retained utility.

Figure 3 shows the same trade-off from the forget-client side: the retained-set auto guard moves in the opposite direction.

![Figure 3. Forget-client loss before and after unlearning in the main trade-off experiment.](formal_cloud_results/stage2_dsample04_round20_20260503/figures/forget_loss_before_after.png)

*Figure 3. Forget-client loss before and after unlearning in the main trade-off experiment.*



It keeps global loss much closer to the FedHDS baseline, changing it from 1.5250 to 1.5482, a degradation of only +0.0232 and still below the global guard threshold 1.575. All 5/5 guard steps are accepted. However, forget-client loss decreases from 2.2581 to 2.2329, a change of -0.0252. This means that the retained-set guard successfully protects utility, but the update direction is not aligned with the forgetting objective.

An important nuance comes from the retrain reference: the held-out client-0 loss in retraining is 2.2168, slightly lower than the FedHDS pre-unlearning value 2.2581. On this IID Dolly split, exact client exclusion therefore does not translate into a monotonic increase in held-out-client loss. Retrain should be interpreted mainly as an exact-removal and utility-cost reference, whereas forget-loss increase remains a behavioral proxy for post-training unlearning strength.

The main conclusion is therefore a reference-selection trade-off rather than a simple win. Forget-batch Hessian produces stronger forgetting but damages global utility. Retained-set Hessian with guard preserves global utility but needs direction correction to become an effective forgetting update.

### 5.3 Direction Misalignment and Guard Validation

The direction and guard ablation is designed to explain the retained-set failure mode observed in the main trade-off experiment. It keeps the same base configuration and runs a complete nine-setting ablation. The goal is not to claim large-scale generalization, but to determine whether the retained-set update direction is misaligned, whether forget-loss guard can reject invalid forgetting steps, and whether a corrected negative direction can improve forgetting while preserving global utility.

Table 3 reports the full ablation.

**Table 3. Full direction and guard ablation.**

| Group | Hessian/sign | Forget loss | Final global loss | Update L2 | Steps | Guard outcome | Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FL baseline | training baseline | - | 1.5441 | - | - | - | - |
| FedHDS baseline | training baseline | - | 1.5250 | - | - | - | - |
| Forget-batch Hessian | forget-batch, sign=positive | 2.2581 -> 2.2866 (+0.0285) | 1.6257 | 0.5077 | 1/1 | all planned steps accepted | 1.69 |
| Retained-set positive auto direction | retained-set, sign=positive | 2.2581 -> 2.2329 (-0.0252) | 1.5482 | 0.2486 | 5/5 | all planned steps accepted | 189.67 |
| Positive + forget-loss guard (tolerance 0) | retained-set, sign=positive | 2.2581 -> 2.2581 (+0.0000) | 1.5250 | 0.0000 | 0/1 | rejected by forget-loss guard | 88.01 |
| Positive + forget-loss guard (tolerance 0.05) | retained-set, sign=positive | 2.2581 -> 2.2581 (+0.0000) | 1.5250 | 0.0000 | 0/1 | rejected by forget-loss guard | 86.58 |
| Negative sign, maximum norm 0.05 | retained-set, sign=negative | 2.2581 -> 2.4105 (+0.1524) | 1.5561 | 0.0500 | 5/5 | all planned steps accepted | 133.76 |
| Negative sign, maximum norm 0.10 | retained-set, sign=negative | 2.2581 -> 2.6104 (+0.3523) | 1.5935 | 0.1000 | 5/5 | all planned steps accepted | 140.23 |
| Negative sign, maximum norm 0.15 | retained-set, sign=negative | 2.2581 -> 2.6880 (+0.4300) | 1.6094 | 0.1200 | 4/5 | stopped by global-loss guard | 137.92 |

The first three rows reproduce the main trade-off baseline setting in the same result directory. FedHDS obtains final global loss 1.5250, while FL obtains 1.5441. The forget-batch Hessian baseline increases forget-client loss from 2.2581 to 2.2866, but it also raises global loss to 1.6257. This confirms the trade-off: direct forget-batch curvature is aligned with forgetting, but it is costly for global utility.

The retained-set positive auto-direction row reproduces the retained-set failure mode. The selected positive direction keeps global loss controlled at 1.5482, but decreases forget-client loss from 2.2581 to 2.2329. This failure is not a random fluctuation but a structural sign-sensitivity issue. Table 4 isolates the direction-probing diagnosis.

**Table 4. Direction-probing diagnosis: positive versus negative candidate under the same retained-set curvature.**

| Candidate sign | Forget delta | Global loss after candidate | Passes global guard? |
|---|---|---|---|
| Positive (+Δθ) | -0.0073 | 1.5482 | Yes |
| Negative (-Δθ) | +0.5014 | 1.7310 | No |

The direction-probing records make the mismatch explicit: the positive candidate changes forget loss by -0.0073 and passes the global guard, whereas the negative candidate changes forget loss by +0.5014 but raises global loss to 1.7310 and fails the guard. This is the key diagnostic result of the paper and should be read as a mechanism finding rather than an ordinary ablation. The core problem with retained-set curvature is not that it lacks a forgetting signal, because the negative direction carries a strong one, but that the effective unconstrained direction lies outside the utility-safe region and the default positive sign points in the wrong direction. In other words, retained-set curvature contains useful forgetting signal, but the effective direction is sign-sensitive and must be validated.

When forget-loss guard is enabled, both tolerance settings reject the first candidate step. The model remains at the FedHDS baseline with update L2 0 and 0/1 accepted steps. This is important guard behavior: the system avoids accepting an update that would look safe under global utility but fail the forgetting objective.

Table 5 focuses on the negative-sign norm sweep.

**Table 5. Negative-sign update-norm sweep.**

| Group | Maximum norm | Forget loss | Global loss | Update L2 | Steps | Guard outcome |
| --- | --- | --- | --- | --- | --- | --- |
| Negative sign, maximum norm 0.05 | 0.0500 | 2.2581 -> 2.4105 (+0.1524) | 1.5250 -> 1.5561 | 0.0500 | 5/5 | all planned steps accepted |
| Negative sign, maximum norm 0.10 | 0.1000 | 2.2581 -> 2.6104 (+0.3523) | 1.5250 -> 1.5935 | 0.1000 | 5/5 | all planned steps accepted |
| Negative sign, maximum norm 0.15 | 0.1500 | 2.2581 -> 2.6880 (+0.4300) | 1.5250 -> 1.6094 | 0.1200 | 4/5 | stopped by global-loss guard |

The negative-sign sweep provides the key correction evidence. As shown in Figure 4, increasing the accepted update norm generally increases forget-loss gain while also requiring global-loss guard control.

![Figure 4. Update norm versus forget-loss gain.](formal_cloud_results/stage3_full_direction_guard_20260504/figures/update_norm_vs_forget_gain.png)

*Figure 4. Update norm versus forget-loss gain.*



With a maximum norm of 0.05, forget-client loss increases from 2.2581 to 2.4105, while global loss remains 1.5561, below the guard 1.575. With a maximum norm of 0.10, forget-client loss increases further to 2.6104, while global loss remains 1.5935, below the guard 1.625. With a maximum norm of 0.15, forget-client loss increases to 2.6880, but the update is stopped after 4/5 accepted steps by the global loss guard. Figure 5 shows the accepted and attempted guard steps behind this early stop.

![Figure 5. Accepted and attempted guard steps.](formal_cloud_results/stage3_full_direction_guard_20260504/figures/guard_steps.png)

*Figure 5. Accepted and attempted guard steps.*



The final accepted update L2 is 0.1200 rather than the requested 0.1500, showing that the stepped guard actively truncates the update when utility approaches the guard boundary. These guard outcomes are not failure cases in the usual sense. The forget-loss guard prevents a utility-safe but forgetting-invalid step from being accepted, while the global-loss guard truncates a forgetting-strong update when it approaches the utility boundary.

These results show that retained-set curvature is sign-sensitive. The original positive direction is not a reliable forgetting direction under the main trade-off configuration. After switching to the negative direction and enabling forget-loss guard, the retained-set update can substantially increase forget-client loss while keeping global loss controlled. Among the three negative-sign settings, a maximum norm of 0.05 is the most conservative and has the smallest global loss increase, while a maximum norm of 0.10 gives a stronger forgetting gain with still-controlled global loss. The maximum-norm-0.15 case is useful as a guard-stopping example rather than the preferred setting.

Compared with the retrain reference, the negative-sign setting with a maximum norm of 0.10 raises the forget-loss proxy much more strongly (2.6104 versus retrain held-out 2.2168) but at a higher utility cost (1.5935 versus retrain 1.5355). The forget-batch baseline is even costlier in utility at 1.6257. This comparison reinforces the paper's interpretation: the proposed retained-negative update is not trying to numerically reproduce retraining. It is an auditable post-training approximation that offers a controllable forgetting/utility trade-off.

### 5.4 Generalization across Model Scale and Data Scale

The scale study evaluates whether the direction correction and guard design remain meaningful when the data scale, model scale, or both are increased. It does not introduce a new algorithmic component. Instead, it forms a 2 x 2 scale matrix: Qwen2-0.5B versus Qwen2.5-1.5B, and a data sampling ratio of 0.4 versus 0.6.

Figure 6 visualizes the large-model/large-data final global loss comparison, while Table 6 summarizes the scale matrix using three comparable entries from each cell: the FedHDS utility baseline, the forget-batch Hessian baseline, and the retained-set negative update with a maximum norm of 0.10.

![Figure 6. Large-model/large-data final global loss comparison.](formal_cloud_results/stage4_large_model_data_20260504/figures/global_loss_final.png)

*Figure 6. Large-model/large-data final global loss comparison.*

**Table 6. Model/data scale generalization matrix.**

| Setting | Model | Data sampling ratio | FedHDS baseline global | Forget-batch Hessian | Retained negative update (maximum norm 0.10) |
| --- | --- | --- | --- | --- | --- |
| Small model + small data | Qwen2-0.5B | 0.4 | 1.5250 | 2.2581 -> 2.2866 (+0.0285), G=1.6257 | 2.2581 -> 2.6104 (+0.3523), G=1.5935 |
| Small model + large data | Qwen2-0.5B | 0.6 | 1.5423 | 2.4377 -> 2.8059 (+0.3681), G=2.0040 | 2.4377 -> 2.6147 (+0.1770), G=1.5701 |
| Large model + small data | Qwen2.5-1.5B | 0.4 | 1.6438 | 1.8812 -> 1.8702 (-0.0110), G=1.6408 | 1.8812 -> 1.9206 (+0.0395), G=1.6405 |
| Large model + large data | Qwen2.5-1.5B | 0.6 | 1.6201 | 2.0880 -> 2.0828 (-0.0052), G=1.6291 | 2.0880 -> 2.1078 (+0.0198), G=1.6132 |

The scale matrix supports three observations. First, increasing data while keeping the small model preserves the main retained-set negative behavior: forget loss increases from 2.4377 to 2.6147 under a maximum norm of 0.10, while final global loss remains 1.5701, close to the FedHDS baseline 1.5423. In the same setting, forget-batch Hessian gives a stronger forgetting gain, 2.4377 to 2.8059, but its final global loss rises sharply to 2.0040. This is a clear utility-cost example.

Second, increasing model size while keeping the data sampling ratio at 0.4 changes the forget-batch behavior. Forget-batch Hessian no longer increases forget loss; it changes from 1.8812 to 1.8702. By contrast, the retained negative update with a maximum norm of 0.10 increases forget loss from 1.8812 to 1.9206, with final global loss 1.6405, slightly lower than the FedHDS baseline 1.6438. This suggests that the retained-set negative direction remains useful even when the direct forget-batch curvature becomes weak or misaligned.

Third, increasing both model and data scale keeps the retained-set negative update stable. Forget-batch Hessian changes forget loss from 2.0880 to 2.0828 and final global loss is 1.6291. The retained negative update with a maximum norm of 0.10 increases forget loss from 2.0880 to 2.1078 and obtains final global loss 1.6132, below the FedHDS baseline 1.6201. The forgetting gain is modest, but the utility behavior is favorable.

These results strengthen the direction-ablation interpretation. The key claim is not that retained-set Hessian always maximizes forget-client loss. Instead, the evidence shows that retained-set Hessian with negative direction and guard constraints gives a more controllable forgetting/utility trade-off across model and data scales than the naive forget-batch Hessian baseline.

The complete scale-study result tables are shown in Tables 7-9.

#### Full results under the small-model/large-data setting

**Table 7. Full results for the small-model/large-data setting.**

| Group | Hessian/sign | Forget loss | Final global loss | Update L2 | Steps |
| --- | --- | --- | --- | --- | --- |
| FL baseline | training baseline | - | 1.5289 | - | - |
| FedHDS baseline | training baseline | - | 1.5423 | - | - |
| Forget-batch Hessian | forget-batch, sign=positive | 2.4377 -> 2.8059 (+0.3681) | 2.0040 | 1.1020 | 1/1 |
| Retained-set auto direction | retained-set, sign=negative | 2.4377 -> 2.7601 (+0.3224) | 1.5862 | 0.1483 | 5/5 |
| Auto-selected direction + forget-loss guard (tolerance 0) | retained-set, sign=negative | 2.4377 -> 2.7601 (+0.3224) | 1.5862 | 0.1483 | 5/5 |
| Auto-selected direction + forget-loss guard (tolerance 0.05) | retained-set, sign=negative | 2.4377 -> 2.7601 (+0.3224) | 1.5862 | 0.1483 | 5/5 |
| Negative sign, maximum norm 0.05 | retained-set, sign=negative | 2.4377 -> 2.5030 (+0.0652) | 1.5554 | 0.0500 | 5/5 |
| Negative sign, maximum norm 0.10 | retained-set, sign=negative | 2.4377 -> 2.6147 (+0.1770) | 1.5701 | 0.1000 | 5/5 |
| Negative sign, maximum norm 0.15 | retained-set, sign=negative | 2.4377 -> 2.7601 (+0.3224) | 1.5862 | 0.1483 | 5/5 |

#### Full results under the large-model/small-data setting

**Table 8. Full results for the large-model/small-data setting.**

| Group | Hessian/sign | Forget loss | Final global loss | Update L2 | Steps |
| --- | --- | --- | --- | --- | --- |
| FL baseline | training baseline | - | 1.6611 | - | - |
| FedHDS baseline | training baseline | - | 1.6438 | - | - |
| Forget-batch Hessian | forget-batch, sign=positive | 1.8812 -> 1.8702 (-0.0110) | 1.6408 | 0.0519 | 1/1 |
| Retained-set auto direction | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Auto-selected direction + forget-loss guard (tolerance 0) | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Auto-selected direction + forget-loss guard (tolerance 0.05) | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Negative sign, maximum norm 0.05 | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Negative sign, maximum norm 0.10 | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Negative sign, maximum norm 0.15 | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |

#### Full results under the large-model/large-data setting

**Table 9. Full results for the large-model/large-data setting.**

| Group | Hessian/sign | Forget loss | Final global loss | Update L2 | Steps |
| --- | --- | --- | --- | --- | --- |
| FL baseline | training baseline | - | 1.6587 | - | - |
| FedHDS baseline | training baseline | - | 1.6201 | - | - |
| Forget-batch Hessian | forget-batch, sign=positive | 2.0880 -> 2.0828 (-0.0052) | 1.6291 | 0.2335 | 1/1 |
| Retained-set auto direction | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Auto-selected direction + forget-loss guard (tolerance 0) | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Auto-selected direction + forget-loss guard (tolerance 0.05) | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Negative sign, maximum norm 0.05 | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Negative sign, maximum norm 0.10 | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Negative sign, maximum norm 0.15 | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |


### 5.5 Seed Robustness under the Large-model/Large-data Setting

After the 2 x 2 scale matrix, we run a small robustness check on the most demanding large-model/large-data setting: Qwen2.5-1.5B with a data sampling ratio of 0.6 and 20 rounds. The goal is not to create a new experimental category, but to test whether the seed-42 observation is an isolated accident. We keep the federated and unlearning configuration fixed and rerun only the five core settings for seed 43 and seed 44.

Figure 7 shows the seed-level forget-client loss changes, while Table 10 compares seed 42, seed 43, and seed 44. Seed 42 is the original large-model/large-data result, while seed 43 and seed 44 come from the robustness run.

![Figure 7. Seed robustness forget-client loss comparison.](formal_cloud_results/stage4_seed_robustness_qwen15b_dsample06_core5_20260504/figures/forget_loss_before_after.png)

*Figure 7. Seed robustness forget-client loss comparison.*



**Table 10. Seed robustness check on the large-model/large-data setting.**

| Seed | FedHDS baseline global | Forget-batch Hessian | Retained auto guard | Retained negative update (maximum norm 0.10) |
| --- | --- | --- | --- | --- |
| 42 | 1.6201 | 2.0880 -> 2.0828 (-0.0052), G=1.6291 | 2.0880 -> 2.1078 (+0.0198), G=1.6132, steps=5/5 | 2.0880 -> 2.1078 (+0.0198), G=1.6132, steps=5/5 |
| 43 | 1.6377 | 1.8483 -> 1.8490 (+0.0007), G=1.6368 | 1.8483 -> 1.8472 (-0.0011), G=1.6380, steps=5/5 | 1.8483 -> 1.8472 (-0.0011), G=1.6380, steps=5/5 |
| 44 | 2.0818 | 1.8732 -> 1.8677 (-0.0056), G=2.0871 | 1.8732 -> 1.8725 (-0.0008), G=2.0826, steps=5/5 | 1.8732 -> 1.8732 (+0.0000), G=2.0818, steps=0/1 |

The robustness check gives a more nuanced picture than a simple success/failure statement. In seed 42, the retained negative update with a maximum norm of 0.10 increases forget loss from 2.0880 to 2.1078 and obtains final global loss 1.6132, below the FedHDS baseline 1.6201. In seed 43, the same retained negative setting preserves utility, with final global loss 1.6380 close to the FedHDS baseline 1.6377, but the forget loss changes from 1.8483 to 1.8472, a very small decrease. In seed 44, the forget-loss guard rejects the retained negative update at 0/1 accepted steps, leaving the model at the FedHDS baseline state with forget loss 1.8732 and global loss 2.0818.

These results support two claims and one boundary. The first supported claim is that retained-set negative updates remain utility-safe in the large-model/large-data setting: even when forgetting gain is absent, global loss stays close to the FedHDS baseline. The second supported claim is that the forget-loss guard is functionally important: in seed 44 it prevents an invalid update from being accepted. The boundary is that the exact forgetting gain is seed-sensitive. Therefore, the paper should not claim that retained negative unlearning always increases forget loss. A more accurate conclusion is that retained-set Hessian with negative direction and guard constraints provides an auditable and utility-safe update path, while the strength of forgetting depends on the local training trajectory and random seed.

### 5.6 Integrated Discussion and Method Positioning

Across the five experiment groups, the results support a coherent interpretation of the method.

First, the FedHDS backbone becomes more meaningful as the training setting becomes larger. In the feasibility validation, FL has slightly lower final global loss than FedHDS, but that setting is only a small-scale feasibility check. In the main trade-off experiment, with a data sampling ratio of 0.4 and 20 rounds, FedHDS achieves lower global loss than FL. This makes FedHDS a reasonable backbone for the unlearning module.

Second, the forget-batch Hessian baseline is useful but costly. It increases forget-client loss in the main small-model setting, where the gain is +0.0285, and it can be even stronger under small-model/large-data scaling. However, it also causes the largest global loss degradation. This confirms that direct forget-batch curvature can serve the forgetting objective but does not sufficiently protect retained utility.

Third, retained-set curvature changes the nature of the update. Its main strength is utility preservation. In the main trade-off experiment, retained-set guard keeps global loss close to the FedHDS baseline and far below the forget-batch global loss. Its weakness is direction alignment: the original sign can decrease forget-client loss. The ablation clarifies that this is not a failure of the guard mechanism itself, but a sign-sensitive update direction issue.

Fourth, the scale study shows that the corrected retained-set negative update is not limited to the original small-model/small-data setting. In the small-model/large-data setting, retained negative unlearning improves forget loss with much lower global utility cost than forget-batch Hessian. In the large-model settings, forget-batch Hessian no longer provides a stable forgetting gain, while retained negative unlearning still increases forget loss or preserves the baseline state under guard control, with global loss staying close to or below the FedHDS baseline.

Fifth, the seed robustness check prevents the large-model/large-data result from being overclaimed. Seed 42 shows a positive retained-negative forgetting gain with global loss below the FedHDS baseline. Seed 43 preserves utility but gives a very small negative forgetting delta, and seed 44 shows the forget-loss guard rejecting the negative update and preserving the FedHDS baseline state. The robust conclusion is therefore utility-safe and auditable behavior, not seed-independent forgetting improvement.

Sixth, the guard mechanisms make the unlearning update auditable. Update norm, clipping coefficient, accepted steps, rejected steps, guard outcome, and post-step losses are all recorded. This is useful for a federated unlearning setting because an approximate unlearning update should not be a black-box parameter jump. The method provides an explicit record of why an update was accepted, rejected, or truncated.

The final experimental message is therefore: retained-set curvature with trust-region guard is a utility-preserving unlearning framework, but it requires direction validation. With negative-sign correction and forget-loss guard, it can increase forget-client loss while keeping global loss within guard constraints. The scale study further shows that this interpretation remains meaningful when the data scale, model scale, or both are increased within the current Dolly/Qwen experimental scope. The seed robustness check adds an important boundary: the framework is more reliable as a guarded utility-preserving update path than as a guarantee of seed-independent forgetting gain.

## 6. Method Limitations and Applicability Boundaries

This work has several limitations.

First, the experiments still use a single dataset family and a fixed forget client. The scale study adds model/data scale validation, but it does not yet provide multi-dataset, multi-model-family, multi-forget-client, or privacy-attack evidence. Therefore, the scale results should be interpreted as controlled generalization within the current FedHDS/Dolly/Qwen setting rather than universal validation.

Second, the manuscript now includes a retrain-from-scratch baseline, but the result also exposes an evaluation nuance. On the current IID Dolly split, retraining without client 0 gives a held-out client loss of 2.2168, slightly lower than the FedHDS pre-unlearning value 2.2581. This means that forget-client loss is a useful operational proxy, but not a perfect monotonic gold-standard forgetting score; cross-client generalization can still reduce held-out loss after exact client exclusion.

Third, the seed robustness check is intentionally small. It repeats only the large-model/large-data core-five settings for two additional seeds. The results are valuable because they show stable utility control and useful guard rejection behavior, but they also show that exact forget-loss improvement is seed-sensitive. A stronger generalization claim would require more seeds and more forget clients.

Fourth, retained-set Hessian computation is more expensive than the forget-batch Hessian baseline. The retained-set method requires constructing a reference loader, computing HVPs on retained data, and evaluating guard losses during stepped updates. In the reported experiments, retained-set unlearning takes much longer than the forget-batch baseline. This cost is acceptable for validating the method, but efficiency improvements are needed for larger deployments.

Fifth, the method is sensitive to guard and reference settings. The retained reference size affects memory usage and curvature quality. The maximum update norm controls the forgetting-utility trade-off. The global guard threshold must be calibrated to the scale of the current model's global loss; a threshold that works in one setting may reject all updates in another. These sensitivities are not unique to our method, but they must be handled carefully in practice.

Sixth, the current experiments use Dolly and the Qwen/Qwen2.5 model family. This is suitable for validating the full unlearning chain and for a controlled scale study, but it does not prove behavior on other datasets, other model families, or more diverse client distributions. Future work should evaluate additional seeds, different forget clients, multiple datasets, and other model families.

Finally, the paper mainly uses loss-based metrics to evaluate forgetting and utility. Forget-client loss is a practical and measurable proxy, but it does not fully capture all possible privacy or memorization risks. More detailed generation-based, membership-style, or privacy-oriented evaluations may be needed for stronger unlearning claims.

## 7. Conclusion and Future Work

This paper studies post-training federated unlearning over the FedHDS training backbone with retained-set curvature approximation and guarded trust-region updates. The main goal is to build a controllable unlearning procedure that can weaken the forget-client contribution while preserving global utility for retained clients.

The experiments lead to five main conclusions. First, the complete procedure is feasible under the current experimental setting. Second, the main trade-off experiment shows that forget-batch Hessian produces a stronger forgetting signal but causes a larger global loss increase, whereas retained-set guarded updates preserve global utility more effectively but can suffer from direction misalignment. Third, the direction and guard ablation explains and corrects this behavior: retained-set updates are sign-sensitive, and negative-sign correction with forget-loss guard can increase forget-client loss while keeping global loss within guard constraints. Fourth, the model/data scale study shows that this retained-negative guarded trade-off remains meaningful across a 2 x 2 scale matrix, although the exact forgetting gain remains scale-dependent. Fifth, the seed robustness check shows that utility control and guard rejection behavior are stable enough to support a cautious claim, while exact forget-loss improvement should be reported as seed-sensitive.

Overall, the results support retained-set curvature as a useful utility-preserving component for federated unlearning, provided that it is combined with explicit guard mechanisms and direction validation. The method should not be interpreted as a universal replacement for forget-batch Hessian. Instead, it provides a more auditable and controlled framework for approximate federated unlearning, where update magnitude, accepted steps, stopping reasons, forgetting behavior, and global utility are all explicitly tracked.

Future work should extend this validation to multiple seeds, multiple forget clients, multiple datasets, other model families, and privacy-oriented attacks such as membership inference or canary extraction. It is also important to add stronger influence-based or privacy-oriented deletion metrics beyond loss proxies, and to improve retained-set HVP efficiency for larger deployments.

## Appendix A. Figure Captions

Figure A1. Stage 1: forget-client loss before and after unlearning.

Figure A2. Stage 1: final global loss comparison.

Figure A3. Stage 1: accepted and attempted guard steps.

Figure A4. Stage 1: unlearning time comparison.

Figure A5. Stage 1: update norm versus forget-loss gain.

Figure A6. Stage 1: update norm versus final global loss.

Figure A7. Stage 2: forget-client loss before and after unlearning.

Figure A8. Stage 2: final global loss comparison.

Figure A9. Stage 2: accepted and attempted guard steps.

Figure A10. Stage 2: unlearning time comparison.

Figure A11. Stage 2: update norm versus forget-loss gain.

Figure A12. Stage 2: update norm versus final global loss.

Figure A13. Stage 3: forget-client loss before and after unlearning.

Figure A14. Stage 3: final global loss comparison.

Figure A15. Stage 3: accepted and attempted guard steps.

Figure A16. Stage 3: unlearning time comparison.

Figure A17. Stage 3: update norm versus forget-loss gain.

Figure A18. Stage 3: update norm versus final global loss.

Figure A19. Stage 4A: forget-client loss before and after unlearning.

Figure A20. Stage 4A: final global loss comparison.

Figure A21. Stage 4A: accepted and attempted guard steps.

Figure A22. Stage 4A: unlearning time comparison.

Figure A23. Stage 4A: update norm versus forget-loss gain.

Figure A24. Stage 4A: update norm versus final global loss.

Figure A25. Stage 4B: forget-client loss before and after unlearning.

Figure A26. Stage 4B: final global loss comparison.

Figure A27. Stage 4B: accepted and attempted guard steps.

Figure A28. Stage 4B: unlearning time comparison.

Figure A29. Stage 4B: update norm versus forget-loss gain.

Figure A30. Stage 4B: update norm versus final global loss.

Figure A31. Stage 4C: forget-client loss before and after unlearning.

Figure A32. Stage 4C: final global loss comparison.

Figure A33. Stage 4C: accepted and attempted guard steps.

Figure A34. Stage 4C: unlearning time comparison.

Figure A35. Stage 4C: update norm versus forget-loss gain.

Figure A36. Stage 4C: update norm versus final global loss.

Figure A37. Stage 4 seed summary: forget-client loss before and after unlearning.

Figure A38. Stage 4 seed summary: final global loss comparison.

Figure A39. Stage 4 seed summary: accepted and attempted guard steps.

Figure A40. Stage 4 seed summary: unlearning time comparison.

Figure A41. Stage 4 seed summary: update norm versus forget-loss gain.

Figure A42. Stage 4 seed summary: update norm versus final global loss.

Figure A43. Stage 4 seed43: forget-client loss before and after unlearning.

Figure A44. Stage 4 seed43: final global loss comparison.

Figure A45. Stage 4 seed43: accepted and attempted guard steps.

Figure A46. Stage 4 seed43: unlearning time comparison.

Figure A47. Stage 4 seed43: update norm versus forget-loss gain.

Figure A48. Stage 4 seed43: update norm versus final global loss.

Figure A49. Stage 4 seed44: forget-client loss before and after unlearning.

Figure A50. Stage 4 seed44: final global loss comparison.

Figure A51. Stage 4 seed44: accepted and attempted guard steps.

Figure A52. Stage 4 seed44: unlearning time comparison.

Figure A53. Stage 4 seed44: update norm versus forget-loss gain.

Figure A54. Stage 4 seed44: update norm versus final global loss.

## References

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," Proceedings of AISTATS, 2017. https://arxiv.org/abs/1602.05629

[2] Y. Cao and J. Yang, "Towards Making Systems Forget with Machine Unlearning," IEEE Symposium on Security and Privacy, pp. 463-480, 2015. https://doi.org/10.1109/SP.2015.35

[3] P. W. Koh and P. Liang, "Understanding Black-box Predictions via Influence Functions," Proceedings of ICML, PMLR 70:1885-1894, 2017. https://proceedings.mlr.press/v70/koh17a.html

[4] N. Agarwal, B. Bullins, and E. Hazan, "Second-order Stochastic Optimization for Machine Learning in Linear Time," Journal of Machine Learning Research, vol. 18, pp. 1-40, 2017. https://www.jmlr.org/papers/v18/16-491.html

[5] Z. Li, X. Meng, L. Wang, and X. Hao, "Survey on Machine Unlearning," Journal of Software, vol. 36, no. 4, pp. 1637-1664, 2025. https://doi.org/10.13328/j.cnki.jos.007237

[6] G. Liu, Y. Yang, X. Ma, C. Wang, and J. Liu, "Federated Unlearning," CoRR abs/2012.13891, 2020. https://arxiv.org/abs/2012.13891

[7] Y. Liu, L. Xu, X. Yuan, C. Wang, and B. Li, "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining," IEEE INFOCOM, pp. 1749-1758, 2022. https://doi.org/10.1109/INFOCOM48880.2022.9796721

[8] L. Wu, S. Guo, J. Wang, Z. Hong, J. Zhang, and Y. Ding, "Federated Unlearning: Guarantee the Right of Clients to Forget," IEEE Network, vol. 36, no. 5, pp. 129-135, 2022. https://doi.org/10.1109/MNET.001.2200198

[9] J. Wang, S. Guo, X. Xie, and H. Qi, "Federated Unlearning via Class-Discriminative Pruning," Proceedings of the ACM Web Conference 2022, pp. 622-632, 2022. https://doi.org/10.1145/3485447.3512222

[10] Y. Li, J. Zhang, Y. Liu, and C. Chen, "Class-wise Federated Unlearning: Harnessing Active Forgetting with Teacher-Student Memory Generation," Knowledge-Based Systems, vol. 316, article 113353, 2025. https://doi.org/10.1016/j.knosys.2025.113353

[11] L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, and N. Papernot, "Machine Unlearning," IEEE Symposium on Security and Privacy, pp. 141-159, 2021. https://doi.org/10.1109/SP40001.2021.00019

[12] N. Romandini, A. Mora, C. Mazzocca, R. Montanari, and P. Bellavista, "Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics," IEEE Transactions on Neural Networks and Learning Systems, 2025. https://doi.org/10.1109/TNNLS.2024.3478334

[13] A. R. Conn, N. I. M. Gould, and P. L. Toint, Trust Region Methods. SIAM, 2000. https://doi.org/10.1137/1.9780898719857

[14] M. Conover, M. Hayes, A. Mathur, J. Xie, J. Wan, S. Shah, A. Ghodsi, P. Wendell, M. Zaharia, and R. Xin, "Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM," Databricks Blog, 2023. https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm

[15] A. Yang et al., "Qwen2 Technical Report," arXiv:2407.10671, 2024. https://arxiv.org/abs/2407.10671

[16] Qwen Team et al., "Qwen2.5 Technical Report," arXiv:2412.15115, 2024. https://arxiv.org/abs/2412.15115

[17] Z. Qin, Z. Wu, B. He, and S. Deng, "Federated Data-Efficient Instruction Tuning for Large Language Models," arXiv:2410.10926, 2024. https://arxiv.org/abs/2410.10926


