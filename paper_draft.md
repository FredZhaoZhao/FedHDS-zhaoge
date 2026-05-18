# Guarded Federated Unlearning via Retained-Set Curvature

## Abstract

Federated unlearning seeks to reduce the influence of a target client without fully retraining the federated model. In language-model federated instruction tuning, this problem is challenging because approximate unlearning must weaken the target-client signal while preserving utility on retained clients. This paper studies a guarded post-training federated unlearning method built on the FedHDS training backbone. We adopt FedHDS as the federated fine-tuning backbone and contribute the unlearning stage rather than a new training algorithm. The method uses the target-client gradient to define the signal to be weakened, estimates curvature from a small retained-client reference set, and constrains the resulting second-order update through global norm clipping, stepped trust-region checking, direction probing, and a forget-loss guard. Experiments on Dolly with Qwen2-0.5B and Qwen2.5-1.5B show that forget-batch Hessian updates can produce stronger target-client loss increases but may incur substantial global-loss degradation. In contrast, retained-set curvature, when combined with direction correction and guard constraints, can improve target-client loss under controlled global loss. Scale and seed checks further indicate that the method is best interpreted as an auditable, utility-preserving approximate unlearning framework rather than a universally strongest forgetting algorithm; the exact forgetting gain remains sensitive to model scale, data scale, and random seed.

**Keywords:** federated unlearning; retained-set curvature; federated instruction tuning; large language models; utility-preserving unlearning

## 1. Introduction

Federated learning allows multiple clients to collaboratively train a global model without directly sharing their local data <sup>[1]</sup>. This training paradigm is useful when data are distributed across users, institutions, or devices, and when centralizing the raw data is undesirable. However, a trained federated model may later face an unlearning request: a client may withdraw consent, request data deletion, or become invalid due to data quality or compliance concerns. In such a case, the system should remove or reduce the influence of that client from the global model.

The most direct solution is to retrain the federated model after excluding the target client <sup>[2]</sup>. Although retraining provides a clean reference, it is often impractical. Federated training may involve many communication rounds, repeated local fine-tuning, and large model checkpoints. The cost becomes even more pronounced for language-model fine-tuning, where even one full run can already require substantial computing resources and time. Federated unlearning therefore studies how to approximate the effect of removing a client without performing full retraining.

Approximate federated unlearning is difficult because it has two competing requirements. On the one hand, the update should weaken the forgotten client's influence. In this paper, this is reflected by the forget-client loss: after unlearning, the loss on the forgotten client should not decrease and preferably should increase. On the other hand, the update should preserve the utility of the model on the retained data distribution. If the update increases forget loss but severely damages global performance, the method is not practically useful.

A central challenge in federated unlearning is the trade-off between forgetting and retained utility. An aggressive update may increase the loss on the forget client, but it can also damage performance on the retained clients. Conversely, an overly conservative update may preserve global utility but fail to meaningfully reduce the forgotten client's influence. This trade-off is especially important when second-order approximations are used: the choice of Hessian or curvature reference directly affects both the direction and magnitude of the unlearning update.

Existing second-order unlearning approximations often use a gradient from the target data together with a Hessian or inverse-Hessian-vector product <sup>[3][4]</sup>. A natural baseline is to compute both the gradient and curvature reference from the forget-client batch. This forget-batch Hessian approach is directly aligned with the forget objective, but it may move the model in a way that is harmful to retained utility. In a federated setting, where the post-unlearning model should still serve the remaining clients, this is a serious limitation.

This work adopts FedHDS, the federated data-efficient instruction-tuning backbone introduced by Qin et al.<sup>[17]</sup>, as the training stage and studies how to perform controlled post-training unlearning on top of it. We therefore do not claim the training backbone as our contribution; our contribution begins after a FedHDS-trained global model is obtained. Instead of estimating curvature only from the forget-client batch, we use retained clients to build a small reference set and approximate the curvature around the retained task distribution. The intuition is that the forget-client gradient defines what should be removed, while the retained-set curvature defines how the model should remain close to the utility-preserving region of the parameter space.

However, retained-set curvature alone is not sufficient. Our experiments show that the retained-set update can preserve global utility but may point in a direction that decreases the forget-client loss. This is an important empirical finding: a method can appear safe under a global utility guard while still failing the forgetting objective. To address this, we introduce a guarded update procedure with four components. First, global update norm clipping limits the magnitude of the second-order correction. Second, a stepped trust-region guard divides the update into smaller steps and rolls back a step if global loss exceeds a threshold. Third, direction probing evaluates positive and negative update signs. Fourth, forget-loss guard rejects a step if it decreases the forget-client loss below a specified tolerance.

We evaluate the method through five purpose-oriented experiment groups: feasibility validation, the main utility-forgetting trade-off comparison, direction and guard ablation, model/data scale generalization, and seed robustness. The feasibility validation establishes that the complete training-unlearning procedure can be evaluated coherently under a small-scale setting. The trade-off comparison shows that forget-batch Hessian gives a clearer forgetting signal but increases global loss substantially, whereas retained-set guard keeps global loss close to the training baseline but exposes an update-direction problem. The ablation then diagnoses this direction problem and shows that negative-sign correction with forget-loss guard can increase forget-client loss while keeping global loss controlled. The scale study evaluates model/data generalization through a 2 x 2 matrix. The seed robustness check reruns the large-model/large-data core settings to show that utility safety and guard behavior are more stable than the exact forgetting gain.

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

Chinese research on federated unlearning is closely connected to the practical need for privacy protection, data deletion, and controllable model maintenance in federated systems. On the broader machine-unlearning side, Chinese survey work has summarized the motivation, technical routes, and evaluation challenges of machine unlearning, emphasizing that unlearning should not only remove target-data influence but also preserve the utility of the remaining model <sup>[5]</sup>.

In federated learning, one representative line of work is efficient client-level removal. FedEraser accelerates federated unlearning by calibrating historical updates instead of fully retraining the federated model <sup>[6]</sup>. Rapid-retraining methods further study how to realize the right to be forgotten in federated learning with lower retraining cost <sup>[7]</sup>. Other work formulates federated unlearning as a client right and discusses the system-level requirements needed to guarantee that a client can be removed from the trained model <sup>[8]</sup>.

Another line focuses on more fine-grained class-level or knowledge-level removal. Class-discriminative pruning removes class-specific influence by pruning parameters related to the target class <sup>[9]</sup>. Active-forgetting methods use teacher-student memory generation to guide class-wise federated unlearning and reduce the dependence on direct access to forgotten data <sup>[10]</sup>. These studies show that domestic and Chinese-author research has moved from "whether a client can be removed" toward more detailed questions of what granularity should be removed, how much utility can be preserved, and how the forgetting effect should be evaluated.

### 2.2 Research Status Abroad

International research provides the general theoretical and methodological foundation for this problem. FedAvg established the standard server-client aggregation framework for federated learning <sup>[1]</sup>. Machine unlearning was originally framed as the problem of making a trained system forget the influence of selected data without rebuilding the whole system from scratch <sup>[2]</sup>. Later SISA-style training made unlearning more efficient by structuring training into shards and slices, so that only affected components need to be retrained <sup>[11]</sup>.

For federated unlearning, recent survey work classifies existing methods into retraining-based, update-correction, historical-update, knowledge-distillation, and second-order approximation families, and points out that evaluation should consider both forgetting effect and retained utility <sup>[12]</sup>. This is consistent with the central tension studied in this paper: a strong forgetting update may damage the global model, while an overly conservative update may preserve utility but fail to remove the target-client influence.

Second-order approximation is another important international line. Influence functions show how training-point influence can be approximated through inverse-Hessian-vector products <sup>[3]</sup>. LiSSA-style stochastic second-order optimization provides a practical route for approximating inverse-HVPs without explicitly constructing the Hessian <sup>[4]</sup>. Trust-region methods offer the optimization idea of limiting parameter movement and accepting an update only within a controlled region <sup>[13]</sup>. These foundations motivate the retained-set curvature and guarded update design used in this paper.

### 2.3 Summary and Motivation of This Work

Existing work shows that federated unlearning must handle three coupled requirements: removing target-client influence, preserving retained-client utility, and keeping the update process computationally feasible. Domestic and Chinese-author studies provide efficient client removal, rapid retraining, and class-level forgetting mechanisms, while international studies provide broader unlearning theory, influence approximation, and trust-region style control. However, most existing formulations do not explicitly separate the source of the forgetting signal from the source of the curvature reference in language-model federated fine-tuning.

This paper is motivated by that gap. We use the forget-client gradient to define what should be weakened, but estimate curvature from retained-client reference data to describe where the model should remain useful. The resulting update is not applied blindly. It is constrained by norm clipping, direction probing, stepped global-utility checking, and forget-loss guard. In this way, the proposed method connects existing federated unlearning and second-order approximation ideas to a more auditable utility-preserving unlearning workflow for language-model fine-tuning.



## 3. Retained-Set Curvature Based Guarded Federated Unlearning Method

FedHDS first produces the pre-unlearning model θ*. The proposed method then applies a post-training unlearning update to that same starting point. The procedure has three phases: (1) compute the forget-client gradient, (2) estimate a second-order direction from retained-client curvature, and (3) validate the update through clipping, direction probing, and guards.

The method is organized as a post-training unlearning procedure rather than a replacement for federated training. The training backbone is responsible for producing a usable global model, while the unlearning module modifies that trained model after a client requests removal. This separation is important because it makes the unlearning behavior easier to analyze: the paper can compare the original training-baseline model, a direct forget-batch Hessian update, and the proposed retained-set guarded update under the same trained starting point.

At a high level, the method has three phases. The first phase defines the forget signal by computing the gradient on the target client. The second phase estimates a second-order direction using retained-client curvature, so that the update is shaped by the data distribution that should remain useful. The third phase validates the update through direction probing and guards. A candidate update is not accepted merely because it is mathematically produced by the inverse-HVP approximation; it must also satisfy measurable global-utility and forget-loss conditions.

### 3.1 Problem Definition and Optimization Objectives

Assume that the federated system has K clients. Client k owns local data Dₖ. After federated training, the server obtains a trained global parameter θ*. A target client j then requests unlearning, and the unlearning procedure returns a post-unlearning parameter θᵤ.

The method has two goals. First, the model should weaken the influence of client j. Second, the model should still work well on retained clients, namely all clients except j.

**Formula (1). Forget-client loss change**

ΔLⱼ = Lⱼ(θᵤ) - Lⱼ(θ*)

Variable explanation:

- ΔLⱼ: change in forget-client loss after unlearning.
- Lⱼ(θ*): average loss on forget client j before unlearning.
- Lⱼ(θᵤ): average loss on forget client j after unlearning.

Interpretation: a positive ΔLⱼ means the model performs worse on the forgotten client after unlearning, which is the desired direction in this paper. A negative value means the update moved in the wrong direction.

This definition follows the experimental goal of this paper: the target client should become less well fitted by the post-unlearning model. It does not by itself prove complete removal of all information about the target client, but it gives a direct and auditable loss-based signal for comparing update directions. For this reason, the paper always reads forget-client loss together with global loss. A valid update should not simply maximize ΔLⱼ; it must do so without causing uncontrolled utility degradation on retained clients.

**Formula (2). Global utility guard**

Lᴳ(θᵤ) ≤ γᴳ

Variable explanation:

- Lᴳ(θᵤ): global evaluation loss after unlearning.
- γᴳ: maximum allowed global loss after unlearning.

Interpretation: the unlearning update is accepted only when it does not push global loss beyond the guard threshold. This is the main utility-preservation constraint.

The global guard turns approximate unlearning into a constrained update problem. Without this constraint, a large second-order update may increase forget-client loss but also destroy the model's general behavior. With the guard, the model is allowed to move only inside a monitored utility region. This is especially important for language-model fine-tuning, where small parameter changes may produce visible changes in loss because the model has many coupled parameters.

### 3.2 FedHDS Federated Fine-tuning Backbone

The training stage uses the FedHDS federated fine-tuning backbone introduced by Qin et al.<sup>[17]</sup>. FedHDS is adopted rather than contributed here: it produces the pre-unlearning global model, and the proposed module modifies that model only after training finishes.

The unlearning method is applied after this training stage. The FedHDS backbone provides the common starting point, while the retained-set curvature method serves as the post-training unlearning mechanism built on top of that trained model.

Accordingly, FL and FedHDS are training baselines, while forget-batch Hessian and retained-set Hessian are post-training unlearning baselines. The comparison therefore asks which update best weakens client j while preserving retained utility from the same trained starting point.

### 3.3 Forget-client Gradient as the Unlearning Signal

The forget client defines the signal that should be removed. The server computes an average gradient on the forget client's data.

**Formula (3). Forget-client gradient**

gⱼ = ∇θ Lⱼ(θ*)

Variable explanation:

- gⱼ: gradient signal from the client that should be forgotten.
- j: index of the forget client.
- Lⱼ(θ*): loss on the forget client's data before unlearning.

Interpretation: gⱼ tells the algorithm which local direction is associated with the forgotten client. It is not applied directly as a simple first-order update. Instead, it is combined with a curvature estimate.

### 3.4 Retained-client Reference Set and Curvature Estimation

The main design choice is to estimate curvature from retained clients rather than from the forget-client batch. This separates "what to remove" from "where the remaining model should stay".

**Formula (4). Retained reference set**

Dref = ⋃(k≠j) Sₖ,  Sₖ ⊂ Dₖ

Variable explanation:

- Dref: retained reference set used for curvature estimation.
- Sₖ: reference subset sampled from retained client k.
- Dₖ: local dataset of client k.
- j: the forget client.
- retained clients: all clients whose data should remain useful after unlearning.

Formula (4) means that the reference set is formed only from retained clients. It is not the full retained training data; it is a small sampled set used to estimate curvature. In the main experiments, each retained client contributes one reference example. With 20 total clients and one forget client, this gives 19 retained reference examples.

Using a small retained reference set is a practical compromise. A larger retained reference set may estimate curvature more accurately, but it increases HVP cost and memory pressure. A very small reference set is cheaper and easier to audit, but it may produce noisier curvature. The reported experiments intentionally use a conservative retained reference size so that the method can be tested under feasible GPU constraints and so that the observed behavior mainly reflects the curvature-source choice and guard design.

**Formula (5). Retained curvature direction**

v = Href⁻¹ gⱼ

Variable explanation:

- Href: approximate Hessian estimated from Dref.
- gⱼ: gradient from the forget client.
- v: second-order direction used for unlearning.

Interpretation: the forget-client gradient defines the signal to remove, while retained-set curvature controls how the model moves so that retained utility is not destroyed.

This separation is the core difference between the proposed path and a direct forget-batch Hessian baseline. In the forget-batch baseline, both the gradient and curvature are estimated from the same target client. That design is naturally aligned with increasing forget-client loss, but it can overfit the unlearning update to the forgotten distribution. In the retained-set design, the update still starts from the forget-client gradient, but the local geometry is measured on retained clients. The expectation is that the update will be more cautious with respect to retained utility, although the experiments show that direction validation is still necessary.

### 3.5 LiSSA/HVP Approximation for Inverse Curvature Direction

For a language model, the Hessian matrix cannot be explicitly built or inverted. The implementation therefore uses Hessian-vector products and a lightweight LiSSA-style approximation.

**Formula (6). Approximate inverse-HVP recursion**

vₜ₊₁ = gⱼ + (1 - λ)vₜ - Href vₜ

Variable explanation:

- vₜ: current approximation vector.
- vₜ₊₁: next approximation vector.
- Href vₜ: Hessian-vector product computed using the retained reference loss.
- λ: damping coefficient in the LiSSA-style recursion.

In the reported experiments, the recursion depth is 1 and the damping coefficient is 0.01. This conservative setting keeps the computation feasible and makes the comparison focus on the curvature reference and guard behavior rather than on a large second-order optimization sweep.

The approximation is therefore used as a controlled engineering component rather than as an exact Hessian inverse. This is a realistic choice for language models, because exact inverse Hessian computation is infeasible. The goal is to obtain a usable local direction and then rely on guard checks to decide whether that direction should be applied. In other words, the second-order approximation proposes an update, while the guard mechanism decides whether the update is acceptable.

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

Interpretation: if the raw update is already small, it is unchanged. If it is too large, it is scaled down to the allowed norm.

Norm clipping also makes results comparable across settings. Different Hessian references can produce update vectors with very different magnitudes. If these raw vectors were applied directly, it would be difficult to tell whether a method works because of its direction or simply because it takes a much larger step. By reporting update L2 norm and clipping threshold together, the experiments make the update scale explicit.

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

Interpretation: if the inequality is satisfied, the candidate substep is accepted. If it fails, the substep is rolled back and the unlearning procedure stops. This is why the update is described as trust-region style: the model is allowed to move only while measured global utility remains inside the allowed region.

The stepped form is more informative than a single all-or-nothing update. If a full update violates the global guard, the stepped guard can still keep the accepted prefix and stop before utility damage becomes excessive. The number of accepted and attempted steps therefore becomes an additional diagnostic signal. A method that repeatedly stops early is not necessarily useless, but it indicates that the proposed update direction is close to the utility boundary.

### 3.8 Direction Probing and Forget-loss Guard for Valid Forgetting

Stage 2 showed that retained-set curvature can preserve global utility but may reduce forget-client loss. That is not valid forgetting. Stage 3 therefore probes both update signs.

**Formula (11). Direction candidates**

θ⁺ = θ* + Δθclip

θ⁻ = θ* - Δθclip

Variable explanation:

- θ⁺: positive update candidate.
- θ⁻: negative update candidate.
- θ*: parameter before unlearning.
- Δθclip: clipped update direction.

The algorithm chooses the direction that gives the larger forget-client loss increase while staying compatible with the global guard when possible.

The final protection is forget-loss guard.

**Formula (12). Forget-loss guard**

Lⱼ(θₜ + δₛ) ≥ Lⱼᵍᵘᵃʳᵈ - ε

Variable explanation:

- θₜ: model parameter before the current guarded step.
- Lⱼᵍᵘᵃʳᵈ: baseline forget-client loss used by the guard.
- ε: allowed small numerical tolerance, such as 0 or 0.005.

Interpretation: a step cannot be accepted merely because global loss is safe. It must also avoid moving the forget-client loss in the wrong direction. This guard is important because it prevents utility-preserving but forgetting-invalid updates from being accepted.

Combining direction probing, global guard, and forget-loss guard gives the final decision rule used in the experiments. The algorithm first checks whether the positive or negative sign is more consistent with increasing forget-client loss. It then applies the selected update in guarded substeps. A substep can be rejected for two different reasons: it may violate global utility, or it may fail the forget-loss condition. Recording these two guard outcomes separately is useful because they correspond to different failure modes. A global-loss rejection means the update is too damaging to retained utility, while a forget-loss rejection means the update does not satisfy the unlearning objective.



## 4. Experimental Design and Evaluation Protocol

### 4.1 Instruction-tuning Dataset and Base Language Models

Experiments use Dolly as the federated instruction-tuning dataset <sup>[14]</sup>. Stages 1-3 use Qwen2-0.5B <sup>[15]</sup> because it is large enough to represent a language-model fine-tuning setting while still compatible with repeated Hessian-vector product experiments under the available single-device compute budget. Stage 4 additionally uses Qwen2.5-1.5B <sup>[16]</sup> to evaluate model-scale generalization. The federated setting uses 20 clients, client fraction 0.2, local step 2, batch size 1, maximum sequence length 64, and forget client index 0.

For all unlearning experiments, the forget-client gradient is computed from the forget client's local training data with `unlearn_grad_sample_size=8`. Retained-set Hessian experiments use `hessian_ref_per_client=1`, so each retained client contributes one reference example. This gives 19 retained reference examples when one of the 20 clients is forgotten. We use `lissa_depth=1`, `lissa_damping=0.01`, and `unlearn_eta=0.01` for the reported second-order updates.

### 4.2 Four-stage Experimental Design and Robustness Check

The raw experiment folders retain stage labels for reproducibility, but the paper reports the results by experimental purpose rather than as a chronological log. The experiments are organized into five groups.

The first group is a feasibility validation with data_sample 0.2 and 10 rounds. It verifies that FL, the training baseline, forget-batch Hessian unlearning, and retained-set Hessian with guard can all be assessed consistently under a small-scale setting. Its goal is feasibility rather than exhaustive comparison.

The second group is the main utility-forgetting trade-off comparison with data_sample 0.4 and 20 rounds. It tests whether the method remains stable under a larger setting and provides the main trade-off evidence. In this group, the retained-set method uses auto global guard calibration, where the guard threshold is set relative to the training baseline pre-unlearning loss.

The third group is a full direction and guard ablation. It keeps the main trade-off configuration and tests update sign, forget-loss guard, and negative-sign norm sweep. Its purpose is to diagnose why the retained-set update preserves global utility but decreases forget-client loss, and to verify whether a corrected direction can satisfy both forgetting and utility constraints.

The fourth group is a model/data scale generalization study. It forms a 2 x 2 matrix over Qwen2-0.5B versus Qwen2.5-1.5B and data_sample 0.4 versus 0.6. This group tests whether the retained-set negative direction and guard behavior remain meaningful when model size and data sample increase.

The fifth group is a seed robustness check. It repeats the large-model/large-data core-five settings with seed 43 and seed 44. Its purpose is to test whether the seed 42 large-model/large-data result is an isolated accident and to identify which claims are stable across seeds.



### 4.3 Compared Unlearning Settings

We compare:

- FL.
- training baseline.
- training baseline + forget-batch Hessian unlearning.
- training baseline + retained-set Hessian + global guard.
- training baseline + retained-set Hessian + negative sign + forget-loss guard.

The forget-batch Hessian setting serves as the direct unlearning baseline. It uses the forget-client batch as both the gradient source and the Hessian reference. The retained-set Hessian setting keeps the forget-client gradient but changes the Hessian reference to retained clients. This isolates the effect of the curvature reference. The negative-sign setting in Stage 3 is used only after direction probing shows that the original sign is misaligned with the forget objective.

### 4.4 Evaluation Metrics and Decision Criteria

We report final global loss, forget-client loss before and after unlearning, update L2 norm, accepted and attempted guard steps, guard outcome, and unlearning time. A method is considered utility-preserving when the post-unlearning global loss stays close to the training baseline pre-unlearning loss or below the specified global guard. A method is considered directionally effective for forgetting when the forget-client loss does not decrease, and preferably increases.

Because Stage 3 is a complete direction/guard ablation, its main evidence is not a new FL/training-baseline comparison, but the relationship among update sign, forget-loss guard, forget-client loss change, and global guard behavior. Stage 4 then tests the same core retained-negative configuration under larger model and data settings. The seed robustness check further separates two conclusions: global utility control and guard behavior are relatively stable, while the magnitude and sign of the forgetting gain can depend on the random seed.

## 5. Experimental Results and Mechanism Analysis

### 5.1 Stage 1: Feasibility and Training-Backbone Context

Stage 1 serves as a small-scale feasibility validation. It is not intended as a definitive performance comparison; instead, it establishes that the training and post-training unlearning procedure can be evaluated stably before moving to the main comparison setting.

Table 1 reports the Stage-1 results.

**Table 1. Stage-1 minimal feasibility results.**

| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) |
|---|---|---|---:|---:|---:|---:|---:|
| Training baseline | eta=1.0000 | none | - | 1.6657 | - | - | - |
| FL baseline | eta=1.0000 | none | - | 1.6443 | - | - | - |
| Forget-batch Hessian | eta=0.0100 | forget-batch | 2.3381 -> 2.3489 | 1.6867 | 0.0872 | 1/1 | 1.44 |
| Retained-set guard | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.700 | retained-set | 2.3381 -> 2.3460 | 1.6615 | 0.0427 | 5/5 | 71.13 |

FL obtains final global loss 1.6443, while the training baseline obtains 1.6657. The small difference between FL and the training baseline in this minimal setting is not the main focus, because Stage 1 uses only data_sample 0.2 and 10 rounds. The more important result is that both unlearning paths complete successfully.

The forget-batch Hessian baseline increases forget-client loss from 2.3381 to 2.3489, giving a forget-loss gain of +0.0108. Its final global loss is 1.6867, which is higher than the training baseline pre-unlearning value. The retained-set guard also increases forget-client loss, from 2.3381 to 2.3460, giving a smaller gain of +0.0079. However, its final global loss is 1.6615, slightly lower than the training baseline final global loss in the same stage.

This stage supports two preliminary observations. First, second-order unlearning can be integrated into the adopted training-and-unlearning framework. Second, the retained-set guard is more conservative in update magnitude and more stable in global utility, although its forgetting gain is weaker than the forget-batch baseline.

### 5.2 Stage 2: Utility-Forgetting Trade-off between Forget-batch and Retained-set Curvature

Stage 2 is the main formal experiment. It increases data_sample from 0.2 to 0.4 and training rounds from 10 to 20. This stage is used as the main evidence for the utility-forgetting trade-off.

Table 2 reports the Stage-2 results.

**Table 2. Stage-2 main trade-off results with retrain reference.**

| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) |
|---|---|---|---:|---:|---:|---:|---:|
| Training baseline | eta=1.0000 | none | - | 1.5250 | - | - | - |
| FL baseline | eta=1.0000 | none | - | 1.5441 | - | - | - |
| Retrain baseline | exclude client 0, learning rate 5e-5 | retrain-from-scratch | heldout client-0 loss = 2.2168 | 1.5355 | - | - | - |
| Forget-batch Hessian | eta=0.0100 | forget-batch | 2.2581 -> 2.2866 | 1.6257 | 0.5077 | 1/1 | 1.64 |
| Retained-set auto guard | eta=0.0100, max_norm=0.300, steps=5, global_guard=1.575 | retained-set | 2.2581 -> 2.2329 | 1.5482 | 0.2486 | 5/5 | 131.13 |

As shown in Figure 1, the training baseline achieves final global loss 1.5250, lower than FL's 1.5441.

Table 2 also includes a retrain-from-scratch reference. Excluding client 0 and retraining from scratch yields final global loss 1.5355 and heldout client-0 loss 2.2168. Relative to the training baseline, the global-loss change is only +0.0105, which is far smaller than the forget-batch change (+0.1007) and also smaller than the corrected retained-negative result with a maximum norm of 0.10 discussed later in Stage 3 (+0.0685).

This is an important difference from Stage 1: with more training rounds and a larger data sample, the training baseline becomes the stronger utility baseline. This supports using it as the backbone for the unlearning experiments.

The forget-batch Hessian baseline gives the clearest forgetting signal in Stage 2. It increases forget-client loss from 2.2581 to 2.2866, a gain of +0.0285. However, this comes with a substantial global utility cost: global loss increases from 1.5250 to 1.6257, a degradation of +0.1007. The actual update L2 norm is 0.5077, much larger than the retained-set guarded update. This result is useful as a trade-off baseline: forget-batch Hessian can move the forget objective in the desired direction, but the update is less constrained with respect to global utility.

Figure 2 shows the same trade-off from the forget-client side: the retained-set auto guard moves in the opposite direction.

![Figure 2. Stage-2 forget-client loss before and after unlearning.](formal_cloud_results/stage2_dsample04_round20_20260503/figures/forget_loss_before_after.png)

*Figure 2. Stage-2 forget-client loss before and after unlearning.*



It keeps global loss much closer to the training baseline, changing it from 1.5250 to 1.5482, a degradation of only +0.0232 and still below the global guard threshold 1.575. All 5/5 guard steps are accepted. However, forget-client loss decreases from 2.2581 to 2.2329, a change of -0.0252. This means that the retained-set guard successfully protects utility, but the update direction is not aligned with the forgetting objective.

An important nuance comes from the retrain reference: the heldout client-0 loss in retraining is 2.2168, slightly lower than the training-baseline pre-unlearning value 2.2581. On this IID Dolly split, exact client exclusion therefore does not translate into a monotonic increase in heldout-client loss. Retrain should be interpreted mainly as an exact-removal and utility-cost reference, whereas forget-loss increase remains a behavioral proxy for post-training unlearning strength.

The main Stage-2 conclusion is therefore a trade-off rather than a simple win. Forget-batch Hessian produces stronger forgetting but damages global utility. Retained-set Hessian with guard preserves global utility but needs direction correction to become an effective forgetting update.

### 5.3 Stage 3: Direction Misalignment Diagnosis and Guard Ablation

Stage 3 is designed to explain the Stage-2 retained-set failure mode. It keeps the Stage-2 base configuration and runs a complete nine-setting direction and guard ablation. The goal is not to claim large-scale generalization, but to determine whether the retained-set update direction is misaligned, whether forget-loss guard can reject invalid forgetting steps, and whether a corrected negative direction can improve forgetting while preserving global utility.

Table 3 reports the full Stage-3 ablation.

**Table 3. Full Stage-3 direction and guard ablation.**

| Group | Hessian/sign | Forget loss | Final global loss | Update L2 | Steps | Guard outcome | Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FL baseline | training baseline | - | 1.5441 | - | - | - | - |
| Training baseline | training baseline | - | 1.5250 | - | - | - | - |
| Forget-batch Hessian | forget-batch, sign=positive | 2.2581 -> 2.2866 (+0.0285) | 1.6257 | 0.5077 | 1/1 | all planned steps accepted | 1.69 |
| Retained-set positive auto direction | retained-set, sign=positive | 2.2581 -> 2.2329 (-0.0252) | 1.5482 | 0.2486 | 5/5 | all planned steps accepted | 189.67 |
| Positive + forget-loss guard (tolerance 0) | retained-set, sign=positive | 2.2581 -> 2.2581 (+0.0000) | 1.5250 | 0.0000 | 0/1 | rejected by forget-loss guard | 88.01 |
| Positive + forget-loss guard (tolerance 0.05) | retained-set, sign=positive | 2.2581 -> 2.2581 (+0.0000) | 1.5250 | 0.0000 | 0/1 | rejected by forget-loss guard | 86.58 |
| Negative sign, max_norm=0.05 | retained-set, sign=negative | 2.2581 -> 2.4105 (+0.1524) | 1.5561 | 0.0500 | 5/5 | all planned steps accepted | 133.76 |
| Negative sign, max_norm=0.10 | retained-set, sign=negative | 2.2581 -> 2.6104 (+0.3523) | 1.5935 | 0.1000 | 5/5 | all planned steps accepted | 140.23 |
| Negative sign, max_norm=0.15 | retained-set, sign=negative | 2.2581 -> 2.6880 (+0.4300) | 1.6094 | 0.1200 | 4/5 | stopped by global-loss guard | 137.92 |

The first three rows reproduce the Stage-2 baseline setting in the same result directory. The training baseline obtains final global loss 1.5250, while FL obtains 1.5441. The forget-batch Hessian baseline increases forget-client loss from 2.2581 to 2.2866, but it also raises global loss to 1.6257. This confirms the Stage-2 trade-off: direct forget-batch curvature is aligned with forgetting, but it is costly for global utility.

The retained-set positive auto-direction row reproduces the retained-set failure mode. The selected positive direction keeps global loss controlled at 1.5482, but decreases forget-client loss from 2.2581 to 2.2329. The best probe forget delta is -0.0073, which indicates that the candidate direction does not improve forgetting. When forget-loss guard is enabled, both tolerance settings reject the first candidate step. The model remains at the training baseline with update L2 0 and 0/1 accepted steps. This is important guard behavior: the system avoids accepting an update that would look safe under global utility but fail the forgetting objective.

Table 4 focuses on the negative-sign norm sweep.

**Table 4. Negative-sign update-norm sweep in Stage 3.**

| Group | max_norm | Forget loss | Global loss | Update L2 | Steps | Guard outcome |
| --- | --- | --- | --- | --- | --- | --- |
| Negative sign, max_norm=0.05 | 0.0500 | 2.2581 -> 2.4105 (+0.1524) | 1.5250 -> 1.5561 | 0.0500 | 5/5 | all planned steps accepted |
| Negative sign, max_norm=0.10 | 0.1000 | 2.2581 -> 2.6104 (+0.3523) | 1.5250 -> 1.5935 | 0.1000 | 5/5 | all planned steps accepted |
| Negative sign, max_norm=0.15 | 0.1500 | 2.2581 -> 2.6880 (+0.4300) | 1.5250 -> 1.6094 | 0.1200 | 4/5 | stopped by global-loss guard |

The negative-sign sweep provides the key correction evidence. As shown in Figure 3, increasing the accepted update norm generally increases forget-loss gain while also requiring global-loss guard control.

![Figure 3. Stage-3 update norm versus forget-loss gain.](formal_cloud_results/stage3_full_direction_guard_20260504/figures/update_norm_vs_forget_gain.png)

*Figure 3. Stage-3 update norm versus forget-loss gain.*



With max_norm 0.05, forget-client loss increases from 2.2581 to 2.4105, while global loss remains 1.5561, below the guard 1.575. With max_norm 0.10, forget-client loss increases further to 2.6104, while global loss remains 1.5935, below the guard 1.625. With max_norm 0.15, forget-client loss increases to 2.6880, but the update is stopped after 4/5 accepted steps by the global loss guard. Figure 4 shows the accepted and attempted guard steps behind this early stop.

![Figure 4. Stage-3 accepted and attempted guard steps.](formal_cloud_results/stage3_full_direction_guard_20260504/figures/guard_steps.png)

*Figure 4. Stage-3 accepted and attempted guard steps.*



The final accepted update L2 is 0.1200 rather than the requested 0.1500, showing that the stepped guard actively truncates the update when utility approaches the guard boundary.

These results show that retained-set curvature is sign-sensitive. The original positive direction is not a reliable forgetting direction under the Stage-2 configuration. After switching to the negative direction and enabling forget-loss guard, the retained-set update can substantially increase forget-client loss while keeping global loss controlled. Among the three negative-sign settings, max_norm 0.05 is the most conservative and has the smallest global loss increase, while max_norm 0.10 gives a stronger forgetting gain with still-controlled global loss. The max_norm 0.15 case is useful as a guard-stopping example rather than the preferred setting.

Compared with the retrain reference, the negative-sign setting with a maximum norm of 0.10 raises the forget-loss proxy much more strongly (2.6104 versus retrain heldout 2.2168) but at a higher utility cost (1.5935 versus retrain 1.5355). The forget-batch baseline is even costlier in utility at 1.6257. This comparison reinforces the interpretation that the retained-negative update is not trying to numerically reproduce retraining. It is an auditable post-training approximation that offers a controllable forgetting/utility trade-off.

### 5.4 Stage 4: Generalization across Model Scale and Data Scale

Stage 4 evaluates whether the Stage-3 direction correction and guard design remain meaningful when the data scale, model scale, or both are increased. It does not introduce a new algorithmic component. Instead, it forms a 2 x 2 scale matrix: Qwen2-0.5B versus Qwen2.5-1.5B, and data_sample 0.4 versus 0.6. Stage 3 provides the small-model/small-data cell, while Stage 4A, Stage 4B, and Stage 4C fill the remaining cells.

Figure 5 visualizes the Stage-4C final global loss comparison, while Table 5 summarizes the scale matrix using three comparable entries from each cell: the training baseline, the forget-batch Hessian baseline, and the retained-set negative update with a maximum norm of 0.10.

**Table 5. Model/data scale generalization matrix.**

| Setting | Model | Data sample | Training baseline global | Forget-batch Hessian | Retained negative update (maximum norm 0.10) |
| --- | --- | --- | --- | --- | --- |
| Stage 3: Small model + small data | Qwen2-0.5B | 0.4 | 1.5250 | 2.2581 -> 2.2866 (+0.0285), G=1.6257 | 2.2581 -> 2.6104 (+0.3523), G=1.5935 |
| Stage 4A: Small model + large data | Qwen2-0.5B | 0.6 | 1.5423 | 2.4377 -> 2.8059 (+0.3681), G=2.0040 | 2.4377 -> 2.6147 (+0.1770), G=1.5701 |
| Stage 4B: Large model + small data | Qwen2.5-1.5B | 0.4 | 1.6438 | 1.8812 -> 1.8702 (-0.0110), G=1.6408 | 1.8812 -> 1.9206 (+0.0395), G=1.6405 |
| Stage 4C: Large model + large data | Qwen2.5-1.5B | 0.6 | 1.6201 | 2.0880 -> 2.0828 (-0.0052), G=1.6291 | 2.0880 -> 2.1078 (+0.0198), G=1.6132 |

The scale matrix supports three observations. First, increasing data while keeping the small model in Stage 4A preserves the main retained-set negative behavior: forget loss increases from 2.4377 to 2.6147 under a maximum norm of 0.10, while final global loss remains 1.5701, close to the training baseline 1.5423. In the same setting, forget-batch Hessian gives a stronger forgetting gain, 2.4377 to 2.8059, but its final global loss rises sharply to 2.0040. This is a clear utility-cost example.

Second, increasing model size while keeping data_sample 0.4 in Stage 4B changes the forget-batch behavior. Forget-batch Hessian no longer increases forget loss; it changes from 1.8812 to 1.8702. By contrast, the retained negative update with a maximum norm of 0.10 increases forget loss from 1.8812 to 1.9206, with final global loss 1.6405, slightly lower than the training baseline 1.6438. This suggests that the retained-set negative direction remains useful even when the direct forget-batch curvature becomes weak or misaligned.

Third, increasing both model and data scale in Stage 4C keeps the retained-set negative update stable. Forget-batch Hessian changes forget loss from 2.0880 to 2.0828 and final global loss is 1.6291. The retained negative update with a maximum norm of 0.10 increases forget loss from 2.0880 to 2.1078 and obtains final global loss 1.6132, below the training baseline 1.6201. The forgetting gain is modest, but the utility behavior is favorable.

These results strengthen the Stage-3 interpretation. The key claim is not that retained-set Hessian always maximizes forget-client loss. Instead, the evidence shows that retained-set Hessian with negative direction and guard constraints gives a more controllable forgetting/utility trade-off across model and data scales than the naive forget-batch Hessian baseline.

The complete Stage-4 result tables are shown in Tables 6-8.

#### Stage 4A: Full results under the small-model/large-data setting

**Table 6. Full Stage-4A results for the small-model/large-data setting.**

| Group | Hessian/sign | Forget loss | Final global loss | Update L2 | Steps |
| --- | --- | --- | --- | --- | --- |
| FL baseline | training baseline | - | 1.5289 | - | - |
| Training baseline | training baseline | - | 1.5423 | - | - |
| Forget-batch Hessian | forget-batch, sign=positive | 2.4377 -> 2.8059 (+0.3681) | 2.0040 | 1.1020 | 1/1 |
| Retained-set auto direction | retained-set, sign=negative | 2.4377 -> 2.7601 (+0.3224) | 1.5862 | 0.1483 | 5/5 |
| Auto direction + forget-loss guard (tolerance 0) | retained-set, sign=negative | 2.4377 -> 2.7601 (+0.3224) | 1.5862 | 0.1483 | 5/5 |
| Auto direction + forget-loss guard (tolerance 0.05) | retained-set, sign=negative | 2.4377 -> 2.7601 (+0.3224) | 1.5862 | 0.1483 | 5/5 |
| Negative sign, max_norm=0.05 | retained-set, sign=negative | 2.4377 -> 2.5030 (+0.0652) | 1.5554 | 0.0500 | 5/5 |
| Negative sign, max_norm=0.10 | retained-set, sign=negative | 2.4377 -> 2.6147 (+0.1770) | 1.5701 | 0.1000 | 5/5 |
| Negative sign, max_norm=0.15 | retained-set, sign=negative | 2.4377 -> 2.7601 (+0.3224) | 1.5862 | 0.1483 | 5/5 |

#### Stage 4B: Full results under the large-model/small-data setting

**Table 7. Full Stage-4B results for the large-model/small-data setting.**

| Group | Hessian/sign | Forget loss | Final global loss | Update L2 | Steps |
| --- | --- | --- | --- | --- | --- |
| FL baseline | training baseline | - | 1.6611 | - | - |
| Training baseline | training baseline | - | 1.6438 | - | - |
| Forget-batch Hessian | forget-batch, sign=positive | 1.8812 -> 1.8702 (-0.0110) | 1.6408 | 0.0519 | 1/1 |
| Retained-set auto direction | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Auto direction + forget-loss guard (tolerance 0) | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Auto direction + forget-loss guard (tolerance 0.05) | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Negative sign, max_norm=0.05 | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Negative sign, max_norm=0.10 | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |
| Negative sign, max_norm=0.15 | retained-set, sign=negative | 1.8812 -> 1.9206 (+0.0395) | 1.6405 | 0.0282 | 5/5 |

#### Stage 4C: Full results under the large-model/large-data setting

**Table 8. Full Stage-4C results for the large-model/large-data setting.**

| Group | Hessian/sign | Forget loss | Final global loss | Update L2 | Steps |
| --- | --- | --- | --- | --- | --- |
| FL baseline | training baseline | - | 1.6587 | - | - |
| Training baseline | training baseline | - | 1.6201 | - | - |
| Forget-batch Hessian | forget-batch, sign=positive | 2.0880 -> 2.0828 (-0.0052) | 1.6291 | 0.2335 | 1/1 |
| Retained-set auto direction | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Auto direction + forget-loss guard (tolerance 0) | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Auto direction + forget-loss guard (tolerance 0.05) | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Negative sign, max_norm=0.05 | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Negative sign, max_norm=0.10 | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |
| Negative sign, max_norm=0.15 | retained-set, sign=negative | 2.0880 -> 2.1078 (+0.0198) | 1.6132 | 0.0305 | 5/5 |


### 5.5 Seed Robustness under the Large-model/Large-data Setting

After the 2 x 2 scale matrix, we run a small robustness check on the most demanding Stage-4C setting: Qwen2.5-1.5B with data_sample 0.6 and 20 rounds. The goal is not to create a new experimental stage, but to test whether the seed=42 observation is an isolated accident. We keep the federated and unlearning configuration fixed and rerun only the core five settings for seed 43 and seed 44.

Figure 6 shows the seed-level forget-client loss changes, while Table 9 compares seed 42, seed 43, and seed 44. Seed 42 is the original Stage-4C result, while seed 43 and seed 44 come from the robustness run.

![Figure 6. Seed robustness forget-client loss comparison.](formal_cloud_results/stage4_seed_robustness_qwen15b_dsample06_core5_20260504/figures/forget_loss_before_after.png)

*Figure 6. Seed robustness forget-client loss comparison.*



**Table 9. Seed robustness check on the large-model/large-data setting.**

| Seed | Training baseline global | Forget-batch Hessian | Retained auto guard | Retained negative update (maximum norm 0.10) |
| --- | --- | --- | --- | --- |
| 42 | 1.6201 | 2.0880 -> 2.0828 (-0.0052), G=1.6291 | 2.0880 -> 2.1078 (+0.0198), G=1.6132, steps=5/5 | 2.0880 -> 2.1078 (+0.0198), G=1.6132, steps=5/5 |
| 43 | 1.6377 | 1.8483 -> 1.8490 (+0.0007), G=1.6368 | 1.8483 -> 1.8472 (-0.0011), G=1.6380, steps=5/5 | 1.8483 -> 1.8472 (-0.0011), G=1.6380, steps=5/5 |
| 44 | 2.0818 | 1.8732 -> 1.8677 (-0.0056), G=2.0871 | 1.8732 -> 1.8725 (-0.0008), G=2.0826, steps=5/5 | 1.8732 -> 1.8732 (+0.0000), G=2.0818, steps=0/1 |

The robustness check gives a more nuanced picture than a simple success/failure statement. In seed 42, the retained negative update with a maximum norm of 0.10 increases forget loss from 2.0880 to 2.1078 and obtains final global loss 1.6132, below the training baseline 1.6201. In seed 43, the same retained negative setting preserves utility, with final global loss 1.6380 close to the training baseline 1.6377, but the forget loss changes from 1.8483 to 1.8472, a very small decrease. In seed 44, the forget-loss guard rejects the retained negative update at 0/1 accepted steps, leaving the model at the training baseline state with forget loss 1.8732 and global loss 2.0818.

These results support two claims and one boundary. The first supported claim is that retained-set negative updates remain utility-safe in the large-model/large-data setting: even when forgetting gain is absent, global loss stays close to the training baseline. The second supported claim is that the forget-loss guard is functionally important: in seed 44 it prevents an invalid update from being accepted. The boundary is that the exact forgetting gain is seed-sensitive. Therefore, the paper should not claim that retained negative unlearning always increases forget loss. A more accurate conclusion is that retained-set Hessian with negative direction and guard constraints provides an auditable and utility-safe update path, while the strength of forgetting depends on the local training trajectory and random seed.

### 5.6 Cross-stage Discussion and Method Positioning

Across the four stages, the results support a coherent interpretation of the method.

First, the training backbone becomes more meaningful as the training setting becomes larger. In Stage 1, FL has slightly lower final global loss than the training baseline, but Stage 1 is only a feasibility test. In Stage 2, with data_sample 0.4 and 20 rounds, the training baseline achieves lower global loss than FL. This makes it a reasonable backbone for the unlearning stage.

Second, the forget-batch Hessian baseline is useful but costly. It consistently increases forget-client loss, especially in Stage 2, where the gain is +0.0285. However, it also causes the largest global loss degradation. This confirms that direct forget-batch curvature can serve the forgetting objective but does not sufficiently protect retained utility.

Third, retained-set curvature changes the nature of the update. Its main strength is utility preservation. In Stage 2, retained-set guard keeps global loss close to the training baseline and far below the forget-batch global loss. Its weakness is direction alignment: the original sign can decrease forget-client loss. Stage 3 clarifies that this is not a failure of the guard mechanism itself, but a sign-sensitive update direction issue.

Fourth, Stage 4 shows that the corrected retained-set negative update is not limited to the original small-model/small-data setting. In the small-model/large-data setting, retained negative unlearning improves forget loss with much lower global utility cost than forget-batch Hessian. In the large-model settings, forget-batch Hessian no longer provides a stable forgetting gain, while retained negative unlearning still increases forget loss or preserves the baseline state under guard control, with global loss staying close to or below the training baseline.

Fifth, the seed robustness check prevents the Stage-4C result from being overclaimed. Seed 42 shows a positive retained-negative forgetting gain with global loss below the training baseline. Seed 43 preserves utility but gives a very small negative forgetting delta, and seed 44 shows the forget-loss guard rejecting the negative update and preserving the training-baseline state. The robust conclusion is therefore utility-safe and auditable behavior, not seed-independent forgetting improvement.

Sixth, the guard mechanisms make the unlearning update auditable. Update norm, clipping coefficient, accepted steps, rejected steps, guard outcome, and post-step losses are all recorded. This is useful for a federated unlearning setting because an approximate unlearning update should not be a black-box parameter jump. The method provides an explicit record of why an update was accepted, rejected, or truncated.

The final experimental message is therefore: retained-set curvature with trust-region guard is a utility-preserving unlearning framework, but it requires direction validation. With negative-sign correction and forget-loss guard, it can increase forget-client loss while keeping global loss within guard constraints. Stage 4 further shows that this interpretation remains meaningful when the data scale, model scale, or both are increased within the current Dolly/Qwen experimental scope. The seed robustness check adds an important boundary: the framework is more reliable as a guarded utility-preserving update path than as a guarantee of seed-independent forgetting gain.

## 6. Method Limitations and Applicability Boundaries

This work has several limitations.

First, the experiments still use a single dataset family and a fixed forget client. Stage 4 adds model/data scale validation, but it does not yet provide multi-dataset, multi-model-family, multi-forget-client, or privacy-attack evidence. Therefore, the scale results should be interpreted as controlled generalization within the current Dolly/Qwen setting with the adopted training backbone rather than universal validation.

Second, the seed robustness check is intentionally small. It repeats only the large-model/large-data core-five settings for two additional seeds. The results are valuable because they show stable utility control and useful guard rejection behavior, but they also show that exact forget-loss improvement is seed-sensitive. A stronger generalization claim would require more seeds and more forget clients.

Third, retained-set Hessian computation is more expensive than the forget-batch Hessian baseline. The retained-set method requires constructing a reference loader, computing HVPs on retained data, and evaluating guard losses during stepped updates. In the reported experiments, retained-set unlearning takes much longer than the forget-batch baseline. This cost is acceptable for validating the method, but efficiency improvements are needed for larger deployments.

Fourth, the method is sensitive to guard and reference settings. The retained reference size affects memory usage and curvature quality. The maximum update norm controls the forgetting-utility trade-off. The global guard threshold must be calibrated to the scale of the current model's global loss; a threshold that works in one setting may reject all updates in another. These sensitivities are not unique to our method, but they must be handled carefully in practice.

Fifth, the current experiments use Dolly and the Qwen/Qwen2.5 model family. This is suitable for validating the full unlearning chain and for a controlled scale study, but it does not prove behavior on other datasets, other model families, or more diverse client distributions. Future work should evaluate additional seeds, different forget clients, multiple datasets, and other model families.

Finally, the paper uses loss-based metrics to evaluate forgetting and utility. Forget-client loss is a practical and measurable proxy, but it does not fully capture all possible privacy or memorization risks. More detailed generation-based, membership-style, or privacy-oriented evaluations may be needed for stronger unlearning claims.

## 7. Conclusion and Future Work

This paper studies federated unlearning with retained-set curvature approximation and guarded trust-region updates on top of an adopted training backbone. The main goal is to build a controllable unlearning procedure that can weaken the forget-client contribution while preserving global utility for retained clients.

The experiments lead to five main conclusions. First, the complete procedure is feasible under the current experimental setting. Second, the main Stage-2 experiment shows a clear trade-off. The forget-batch Hessian baseline produces a stronger forgetting signal, but it also causes a larger global loss increase. The retained-set guarded update preserves global utility more effectively, but its original direction can decrease forget-client loss. Third, Stage 3 explains and corrects this behavior. Retained-set updates are sign-sensitive; with forget-loss guard and negative-sign correction, forget-client loss can be substantially increased while global loss remains within guard constraints. Fourth, Stage 4 shows that this retained-negative guarded trade-off remains meaningful across a 2 x 2 model/data scale matrix, although the exact forgetting gain remains scale-dependent. Fifth, the seed robustness check shows that utility control and guard rejection behavior are stable enough to support a cautious claim, while exact forget-loss improvement should be reported as seed-sensitive.

Overall, the results support retained-set curvature as a useful utility-preserving component for federated unlearning, provided that it is combined with explicit guard mechanisms and direction validation. The method should not be interpreted as a universal replacement for forget-batch Hessian. Instead, it provides a more auditable and controlled framework for approximate federated unlearning, where update magnitude, accepted steps, stopping reasons, forgetting behavior, and global utility are all explicitly tracked.

Future work should extend this validation to multiple seeds, multiple forget clients, multiple datasets, other model families, and privacy-oriented attacks such as membership inference or canary extraction. Efficiency improvements for retained-set HVP computation are also important for practical deployment.

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
