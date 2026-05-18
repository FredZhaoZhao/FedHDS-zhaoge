from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from update_stage3_full_reports import (  # noqa: E402
    STAGE3,
    STAGE4A,
    STAGE4B,
    STAGE4C,
    STAGE4_SEED,
    ROOT as REPORT_ROOT,
    appendix_caption,
    load_seed_rows,
    load_stage4_sets,
    markdown_to_docx,
    read_csv_rows,
    rows_for_seed,
    table_md,
)


EN_ABSTRACT = """Federated unlearning seeks to reduce the influence of a target client without fully retraining the federated model. In language-model federated instruction tuning, this problem is challenging because approximate unlearning must weaken the target-client signal while preserving utility on retained clients. This paper presents a guarded federated unlearning method built on an implemented data-filtering federated training backbone. The method uses the target-client gradient to define the signal to be weakened, estimates curvature from a small retained-client reference set, and constrains the resulting second-order update through global norm clipping, stepped trust-region checking, direction probing, and a forget-loss guard. Experiments on Dolly with Qwen2-0.5B and Qwen2.5-1.5B show that forget-batch Hessian updates can produce stronger target-client loss increases but may incur substantial global-loss degradation. In contrast, retained-set curvature, when combined with direction correction and guard constraints, can improve target-client loss under controlled global loss. Scale and seed checks further indicate that the method is best interpreted as an auditable, utility-preserving approximate unlearning framework rather than a universally strongest forgetting algorithm; the exact forgetting gain remains sensitive to model scale, data scale, and random seed."""


EN_KEYWORDS = "**Keywords:** federated unlearning; retained-set curvature; Hessian-vector product; trust-region guard; language-model fine-tuning"


REFERENCES = """## References

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
"""


ZH_REFERENCES = """## 参考文献

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," Proceedings of AISTATS, 2017. https://arxiv.org/abs/1602.05629

[2] Y. Cao and J. Yang, "Towards Making Systems Forget with Machine Unlearning," IEEE Symposium on Security and Privacy, pp. 463-480, 2015. https://doi.org/10.1109/SP.2015.35

[3] P. W. Koh and P. Liang, "Understanding Black-box Predictions via Influence Functions," Proceedings of ICML, PMLR 70:1885-1894, 2017. https://proceedings.mlr.press/v70/koh17a.html

[4] N. Agarwal, B. Bullins, and E. Hazan, "Second-order Stochastic Optimization for Machine Learning in Linear Time," Journal of Machine Learning Research, vol. 18, pp. 1-40, 2017. https://www.jmlr.org/papers/v18/16-491.html

[5] 李梓童, 孟小峰, 王雷霞, 郝新丽. 机器遗忘综述[J]. 软件学报, 2025, 36(4): 1637-1664. https://doi.org/10.13328/j.cnki.jos.007237

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
"""


FIGURE_APPENDIX_EN = """## Appendix A. Figure Captions

Figure A1. Forget-client loss before and after unlearning across the reported settings.

Figure A2. Final global loss after FL, FedHDS, forget-batch Hessian, and retained-set Hessian variants.

Figure A3. Relationship between update L2 norm and final global loss.

Figure A4. Relationship between update L2 norm and forget-loss gain.

Figure A5. Accepted and attempted guard steps under the stepped trust-region update.

Figure A6. Unlearning time comparison between forget-batch Hessian and retained-set guarded updates.
"""


APPENDIX_FIGURE_FOLDERS = [
    (REPORT_ROOT / "formal_cloud_results_formal_minimal_20260503_combined" / "figures", "Stage 1"),
    (REPORT_ROOT / "formal_cloud_results" / "stage2_dsample04_round20_20260503" / "figures", "Stage 2"),
    (STAGE3 / "figures", "Stage 3"),
    (STAGE4A / "figures", "Stage 4A"),
    (STAGE4B / "figures", "Stage 4B"),
    (STAGE4C / "figures", "Stage 4C"),
    (STAGE4_SEED / "figures", "Stage 4 seed summary"),
    (STAGE4_SEED / "seed43" / "figures", "Stage 4 seed43"),
    (STAGE4_SEED / "seed44" / "figures", "Stage 4 seed44"),
]


def appendix_figure_caption_list(zh: bool = False) -> str:
    heading = "## 附录 A：图编号说明" if zh else "## Appendix A. Figure Captions"
    lines = [heading]
    index = 1
    for folder, title in APPENDIX_FIGURE_FOLDERS:
        if not folder.exists():
            continue
        for img in sorted(folder.glob("*.png")):
            lines.append("")
            lines.append(appendix_caption(title, img, index, zh=zh))
            index += 1
    return "\n".join(lines) + "\n"


EN_METHOD_TEXT = """## 3. Retained-Set Curvature Based Guarded Federated Unlearning Method

This section presents the proposed FedHDS-style federated unlearning method. The method first obtains a trained global model from the FedHDS-style filtering backbone, then estimates an unlearning direction by combining the forget-client gradient with retained-set curvature. The final parameter update is applied only after norm clipping, direction probing, stepped global-utility checking, and forget-loss guarding.

The method is organized as a post-training unlearning procedure rather than a replacement for federated training. The training backbone is responsible for producing a usable global model, while the unlearning module modifies that trained model after a client requests removal. This separation is important because it makes the unlearning behavior easier to analyze: the paper can compare the original FedHDS-style model, a direct forget-batch Hessian update, and the proposed retained-set guarded update under the same trained starting point.

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

### 3.2 FedHDS-style Federated Fine-tuning Backbone

The training stage uses the FedHDS-style filtering backbone implemented in this project. The term FedHDS-style describes the local training backbone used in this work, not an external citation. In each communication round, selected clients perform local fine-tuning, and the server aggregates their model updates.

The unlearning method is applied after this training stage. Therefore, FedHDS-style training produces θ*, while the retained-set curvature method is the post-training unlearning mechanism that constructs θᵤ.

This design also makes the comparison with FL and FedHDS clearer. FL and FedHDS are treated as training baselines, while forget-batch Hessian and retained-set Hessian are treated as post-training unlearning baselines. Therefore, a lower global loss for FedHDS does not by itself solve unlearning; it only provides a stronger starting point. The actual unlearning question is whether the post-training update can weaken the forget client while preserving the trained model's retained utility.

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

Combining direction probing, global guard, and forget-loss guard gives the final decision rule used in the experiments. The algorithm first checks whether the positive or negative sign is more consistent with increasing forget-client loss. It then applies the selected update in guarded substeps. A substep can be rejected for two different reasons: it may violate global utility, or it may fail the forget-loss condition. Recording these two stop reasons separately is useful because they correspond to different failure modes. A global-loss rejection means the update is too damaging to retained utility, while a forget-loss rejection means the update does not satisfy the unlearning objective.
"""


EN_RESEARCH_STATUS = """## 2. Research Status in China and Abroad

### 2.1 Research Status in China

Chinese research on federated unlearning is closely connected to the practical need for privacy protection, data deletion, and controllable model maintenance in federated systems. On the broader machine-unlearning side, Chinese survey work has summarized the motivation, technical routes, and evaluation challenges of machine unlearning, emphasizing that unlearning should not only remove target-data influence but also preserve the utility of the remaining model [Machine-Unlearning-Survey-ZH].

In federated learning, one representative line of work is efficient client-level removal. FedEraser accelerates federated unlearning by calibrating historical updates instead of fully retraining the federated model [FedEraser]. Rapid-retraining methods further study how to realize the right to be forgotten in federated learning with lower retraining cost [Rapid-Retraining]. Other work formulates federated unlearning as a client right and discusses the system-level requirements needed to guarantee that a client can be removed from the trained model [Client-Forget].

Another line focuses on more fine-grained class-level or knowledge-level removal. Class-discriminative pruning removes class-specific influence by pruning parameters related to the target class [Class-Pruning]. Active-forgetting methods use teacher-student memory generation to guide class-wise federated unlearning and reduce the dependence on direct access to forgotten data [Active-Forgetting]. These studies show that domestic and Chinese-author research has moved from "whether a client can be removed" toward more detailed questions of what granularity should be removed, how much utility can be preserved, and how the forgetting effect should be evaluated.

### 2.2 Research Status Abroad

International research provides the general theoretical and methodological foundation for this problem. FedAvg established the standard server-client aggregation framework for federated learning [FedAvg]. Machine unlearning was originally framed as the problem of making a trained system forget the influence of selected data without rebuilding the whole system from scratch [Machine-Unlearning]. Later SISA-style training made unlearning more efficient by structuring training into shards and slices, so that only affected components need to be retrained [SISA-Unlearning].

For federated unlearning, recent survey work classifies existing methods into retraining-based, update-correction, historical-update, knowledge-distillation, and second-order approximation families, and points out that evaluation should consider both forgetting effect and retained utility [Federated-Unlearning-Survey]. This is consistent with the central tension studied in this paper: a strong forgetting update may damage the global model, while an overly conservative update may preserve utility but fail to remove the target-client influence.

Second-order approximation is another important international line. Influence functions show how training-point influence can be approximated through inverse-Hessian-vector products [Influence-Functions]. LiSSA-style stochastic second-order optimization provides a practical route for approximating inverse-HVPs without explicitly constructing the Hessian [LiSSA]. Trust-region methods offer the optimization idea of limiting parameter movement and accepting an update only within a controlled region [Trust-Region]. These foundations motivate the retained-set curvature and guarded update design used in this paper.

### 2.3 Summary and Motivation of This Work

Existing work shows that federated unlearning must handle three coupled requirements: removing target-client influence, preserving retained-client utility, and keeping the update process computationally feasible. Domestic and Chinese-author studies provide efficient client removal, rapid retraining, and class-level forgetting mechanisms, while international studies provide broader unlearning theory, influence approximation, and trust-region style control. However, most existing formulations do not explicitly separate the source of the forgetting signal from the source of the curvature reference in language-model federated fine-tuning.

This paper is motivated by that gap. We use the forget-client gradient to define what should be weakened, but estimate curvature from retained-client reference data to describe where the model should remain useful. The resulting update is not applied blindly. It is constrained by norm clipping, direction probing, stepped global-utility checking, and forget-loss guard. In this way, the proposed method connects existing federated unlearning and second-order approximation ideas to a more auditable utility-preserving unlearning workflow for language-model fine-tuning.
"""


ZH_RESEARCH_STATUS = """## 2 国内外研究现状

### 2.1 国内研究现状

国内关于联邦遗忘的研究主要受到隐私保护、数据删除合规和联邦系统可维护性的共同驱动。在更广义的机器遗忘方向上，国内已有综述工作系统总结了机器遗忘的基本目标、主要技术路线和评估难点，强调遗忘算法不仅要削弱目标数据对模型的影响，还要尽可能保持剩余模型效用 [Machine-Unlearning-Survey-ZH]。这为联邦场景下讨论“遗忘效果”和“保留效用”的平衡提供了基础。

在联邦学习场景中，国内及华人学者的研究较早关注客户端级别的高效移除。FedEraser 通过校准历史模型更新来加速联邦遗忘，避免每次收到遗忘请求都完整重训联邦模型 [FedEraser]。面向“被遗忘权”的快速重训方法进一步研究如何在联邦学习中以较低代价实现客户端删除请求 [Rapid-Retraining]。也有研究从系统权利和协议设计角度讨论如何保证客户端在联邦模型中被遗忘的权利 [Client-Forget]。

除客户端级别遗忘外，国内及华人学者还开展了类别级、知识级等更细粒度的联邦遗忘研究。类别判别剪枝方法通过剪除与目标类别相关的参数来削弱类别影响 [Class-Pruning]。主动遗忘方法则利用 teacher-student memory generation 引导类别级联邦遗忘，降低对被遗忘数据直接访问的依赖 [Active-Forgetting]。这些研究说明，国内相关工作已经从“能否删除某个客户端”逐步扩展到“删除粒度如何定义、效用如何保持、遗忘效果如何评估”等更细的问题。

### 2.2 国外研究现状

国外研究为本文问题提供了更基础的方法框架。FedAvg 提出了经典的联邦平均训练范式，为后续联邦学习和联邦遗忘研究奠定了基本训练流程 [FedAvg]。机器遗忘早期工作将问题定义为：在不从头重建整个系统的情况下，使训练系统移除特定数据的影响 [Machine-Unlearning]。随后 SISA training 通过 shard 和 slice 结构降低遗忘时需要重训的范围，提高了机器遗忘的可操作性 [SISA-Unlearning]。

在联邦遗忘方向，国外已有综述将现有方法归纳为重训类、梯度修正类、历史更新补偿类、知识蒸馏类和二阶近似类，并指出评价联邦遗忘不能只看遗忘强度，还必须同时考察保留数据上的效用 [Federated-Unlearning-Survey]。这一判断与本文关注的问题一致：强遗忘更新可能破坏全局模型，而过度保守的更新虽然保护了效用，却可能无法削弱目标客户端影响。

二阶近似方向也是国外研究中的重要基础。Influence functions 说明可以通过 inverse-Hessian-vector product 近似训练样本对模型参数和预测结果的影响 [Influence-Functions]。LiSSA-style 随机二阶优化提供了在不显式构造 Hessian 的情况下近似 inverse-HVP 的可行路径 [LiSSA]。信赖域方法则提供了“限制参数移动范围、在可接受区域内接收更新”的优化思想 [Trust-Region]。这些工作共同支撑了本文使用 retained-set curvature 和 guarded update 的方法设计。

### 2.3 研究述评与本文工作引出

综上，现有联邦遗忘研究需要同时处理三个问题：削弱目标客户端影响、保持保留客户端效用，以及控制遗忘更新的计算成本。国内及华人学者研究强调客户端快速移除、被遗忘权实现和类别级遗忘；国外研究则提供机器遗忘理论、影响函数近似、二阶优化和信赖域控制等基础。然而，在语言模型联邦微调场景下，已有工作较少显式区分“遗忘信号来自哪里”和“曲率参考来自哪里”。

本文正是围绕这一缺口展开。本文用 forget-client gradient 表示需要削弱的目标客户端信号，同时用 retained-client reference set 估计曲率，使遗忘方向和效用保持参考来源分离。随后通过全局范数裁剪、方向探测、分步全局效用检查和 forget-loss guard 控制更新过程。这样，本文把国内外联邦遗忘和二阶近似研究中的思想结合到语言模型微调场景中，形成一个更可审计、更强调效用保持的受控联邦遗忘流程。
"""


def replace_abstract(text: str) -> str:
    return re.sub(
        r"## Abstract\n\n.*?(?=\n## 1\. Introduction)",
        "## Abstract\n\n" + EN_ABSTRACT + "\n\n" + EN_KEYWORDS + "\n",
        text,
        flags=re.S,
    )


def replace_english_research_status(text: str) -> str:
    return re.sub(
        r"## 2\. (?:Related Work|Research Status in China and Abroad)\n.*?(?=\n## 3\.)",
        EN_RESEARCH_STATUS + "\n\n",
        text,
        flags=re.S,
    )


def replace_references(text: str) -> str:
    if "## References" in text:
        text = re.sub(r"## References\n.*\Z", REFERENCES.rstrip() + "\n", text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + REFERENCES
    return text


def replace_zh_references(text: str) -> str:
    if "## 参考文献" in text:
        text = re.sub(r"## 参考文献\n.*\Z", ZH_REFERENCES.rstrip() + "\n", text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + ZH_REFERENCES
    return text


def numeric_citations(text: str) -> str:
    citation_map = [
        ("[Influence-Functions; LiSSA]", "<sup>[3][4]</sup>"),
        ("[FedAvg]", "<sup>[1]</sup>"),
        ("[Machine-Unlearning]", "<sup>[2]</sup>"),
        ("[Influence-Functions]", "<sup>[3]</sup>"),
        ("[LiSSA]", "<sup>[4]</sup>"),
        ("[Machine-Unlearning-Survey-ZH]", "<sup>[5]</sup>"),
        ("[FedEraser]", "<sup>[6]</sup>"),
        ("[Rapid-Retraining]", "<sup>[7]</sup>"),
        ("[Client-Forget]", "<sup>[8]</sup>"),
        ("[Class-Pruning]", "<sup>[9]</sup>"),
        ("[Active-Forgetting]", "<sup>[10]</sup>"),
        ("[SISA-Unlearning]", "<sup>[11]</sup>"),
        ("[Federated-Unlearning-Survey]", "<sup>[12]</sup>"),
        ("[Trust-Region]", "<sup>[13]</sup>"),
        ("[Dolly]", "<sup>[14]</sup>"),
        ("[Qwen2.5]", "<sup>[16]</sup>"),
        ("[Qwen2]", "<sup>[15]</sup>"),
    ]
    for old, new in citation_map:
        text = text.replace(old, new)
    return text


def ensure_caption(text: str, table_phrase: str, caption: str) -> str:
    if caption in text:
        return text
    return text.replace(table_phrase + "\n\n|", table_phrase + "\n\n**" + caption + "**\n\n|")


def ensure_heading_table_caption(text: str, heading: str, caption: str) -> str:
    if caption in text:
        return text
    return text.replace(heading + "\n\n|", heading + "\n\n**" + caption + "**\n\n|")


def insert_after_once(text: str, anchor: str, insertion: str) -> str:
    if insertion.strip() in text or anchor not in text:
        return text
    return text.replace(anchor, anchor + "\n\n" + insertion.rstrip() + "\n\n", 1)


def figure_block(caption: str, path: str) -> str:
    return f"![{caption}]({path})\n\n*{caption}*"


def remove_figure_block(text: str, caption: str, path: str) -> str:
    image = re.escape(f"![{caption}]({path})")
    caption_line = re.escape(f"*{caption}*")
    return re.sub(rf"\n*{image}(?:\n\n{caption_line})?\n*", "\n\n", text)


def normalize_inline_images(text: str) -> str:
    return re.sub(r"(!\[[^\]]+\]\([^)]+\))\s+(?=\S)", r"\1\n\n", text)


def normalize_zh_figures(text: str) -> str:
    figures = [
        (
            "阶段二将 data_sample 提高到 0.4、rounds 提高到 20，用于观察更正式设置下的效用和遗忘 trade-off。",
            "阶段二将 data_sample 提高到 0.4、rounds 提高到 20，用于观察更正式设置下的效用和遗忘 trade-off。如图 1 所示，阶段二中 FedHDS 的 final global loss 低于 FL，因此后续遗忘实验以 FedHDS-style 训练底座作为主要 utility baseline。",
            "图 1. 阶段二 final global loss 对比。",
            "formal_cloud_results/stage2_dsample04_round20_20260503/figures/global_loss_final.png",
        ),
        (
            "阶段二的主要结论是：FedHDS 在更大设置下成为更好的 utility baseline；forget-batch Hessian 能提高 forget loss，但 global loss 代价较大；retained-set guard 能显著保护 global utility，但原方向可能降低 forget-client loss。这一问题直接引出阶段三的方向与 guard 消融。",
            "阶段二的主要结论是：FedHDS 在更大设置下成为更好的 utility baseline；forget-batch Hessian 能提高 forget loss，但 global loss 代价较大；retained-set guard 能显著保护 global utility，但原方向可能降低 forget-client loss。如图 2 所示，retained-set guard 在 forget-client loss 上出现反向变化，这一问题直接引出阶段三的方向与 guard 消融。",
            "图 2. 阶段二遗忘前后 forget-client loss 对比。",
            "formal_cloud_results/stage2_dsample04_round20_20260503/figures/forget_loss_before_after.png",
        ),
        (
            "negative-sign sweep 是阶段三的关键证据。",
            "negative-sign sweep 是阶段三的关键证据。如图 3 所示，accepted update norm 增大时，forget-loss gain 通常也随之增大，但这需要 global guard 控制效用代价。",
            "图 3. 阶段三 update norm 与 forget-loss gain 的关系。",
            "formal_cloud_results/stage3_full_direction_guard_20260504/figures/update_norm_vs_forget_gain.png",
        ),
        (
            "max_norm=0.15 时，guard 在 4/5 步停止，最终 update L2 为 0.1200。",
            "max_norm=0.15 时，guard 在 4/5 步停止，最终 update L2 为 0.1200。图 4 展示了该设置下 attempted steps 与 accepted steps 的差异，说明分步 guard 实际截断了过大的更新。",
            "图 4. 阶段三分步 guard 接受情况。",
            "formal_cloud_results/stage3_full_direction_guard_20260504/figures/guard_steps.png",
        ),
        (
            "阶段四不引入新方法，而是用 2 x 2 矩阵验证阶段三得到的 corrected retained-negative update 是否在更大模型或更大数据下仍有解释性。阶段三作为小模型小数据格子，Stage 4A、4B、4C 补齐其余三个格子。",
            "阶段四不引入新方法，而是用 2 x 2 矩阵验证阶段三得到的 corrected retained-negative update 是否在更大模型或更大数据下仍有解释性。阶段三作为小模型小数据格子，Stage 4A、4B、4C 补齐其余三个格子。如图 5 所示，在 Stage 4C 大模型大数据设置下，retained negative 的 final global loss 仍低于 FedHDS baseline。",
            "图 5. 阶段四C final global loss 对比。",
            "formal_cloud_results/stage4_large_model_data_20260504/figures/global_loss_final.png",
        ),
        (
            "seed robustness check 不是新阶段，而是对 Stage 4C 的核心设置补充 seed 43 和 seed 44，用于判断 seed 42 的结果是否偶然。",
            "seed robustness check 不是新阶段，而是对 Stage 4C 的核心设置补充 seed 43 和 seed 44，用于判断 seed 42 的结果是否偶然。如图 6 所示，不同 seed 下 forget-client loss 变化幅度存在差异，因此本文对遗忘增益保持谨慎结论。",
            "图 6. Seed robustness 中 forget-client loss 对比。",
            "formal_cloud_results/stage4_seed_robustness_qwen15b_dsample06_core5_20260504/figures/forget_loss_before_after.png",
        ),
    ]
    for old_anchor, new_anchor, caption, path in figures:
        text = remove_figure_block(text, caption, path)
        if new_anchor not in text and old_anchor in text:
            text = text.replace(old_anchor, new_anchor, 1)
        text = insert_after_once(text, new_anchor, figure_block(caption, path))
    return text


def translate_english_table_headers(text: str) -> str:
    return text.replace(
        "| 组别 | Hessian/方向 | Forget loss | Final global loss | Update L2 | 步数 | 停止原因 | 时间(s) |",
        "| Group | Hessian/sign | Forget loss | Final global loss | Update L2 | Steps | Stop reason | Time (s) |",
    ).replace(
        "| 组别 | max_norm | Forget loss | Global loss | Update L2 | 步数 | 停止原因 |",
        "| Group | max_norm | Forget loss | Global loss | Update L2 | Steps | Stop reason |",
    )


def insert_english_figures(text: str) -> str:
    figures = [
        (
            "In this setting, FedHDS achieves final global loss 1.5250, lower than FL's 1.5441.",
            "As shown in Figure 1, FedHDS achieves final global loss 1.5250, lower than FL's 1.5441.",
            "Figure 1. Stage-2 final global loss comparison.",
            "formal_cloud_results/stage2_dsample04_round20_20260503/figures/global_loss_final.png",
        ),
        (
            "The retained-set auto guard shows the opposite behavior.",
            "Figure 2 shows the same trade-off from the forget-client side: the retained-set auto guard moves in the opposite direction.",
            "Figure 2. Stage-2 forget-client loss before and after unlearning.",
            "formal_cloud_results/stage2_dsample04_round20_20260503/figures/forget_loss_before_after.png",
        ),
        (
            "The negative-sign sweep provides the key correction evidence.",
            "The negative-sign sweep provides the key correction evidence. As shown in Figure 3, increasing the accepted update norm generally increases forget-loss gain while also requiring global-loss guard control.",
            "Figure 3. Stage-3 update norm versus forget-loss gain.",
            "formal_cloud_results/stage3_full_direction_guard_20260504/figures/update_norm_vs_forget_gain.png",
        ),
        (
            "With max_norm 0.15, forget-client loss increases to 2.6880, but the update is stopped after 4/5 accepted steps by the global loss guard.",
            "With max_norm 0.15, forget-client loss increases to 2.6880, but the update is stopped after 4/5 accepted steps by the global loss guard. Figure 4 shows the accepted and attempted guard steps behind this early stop.",
            "Figure 4. Stage-3 accepted and attempted guard steps.",
            "formal_cloud_results/stage3_full_direction_guard_20260504/figures/guard_steps.png",
        ),
        (
            "Table 5 summarizes the scale matrix using three comparable entries from each cell: the FedHDS utility baseline, the forget-batch Hessian baseline, and the retained-set negative norm010 configuration.",
            "Figure 5 visualizes the Stage-4C final global loss comparison, while Table 5 summarizes the scale matrix using three comparable entries from each cell: the FedHDS utility baseline, the forget-batch Hessian baseline, and the retained-set negative norm010 configuration.",
            "Figure 5. Stage-4C final global loss comparison.",
            "formal_cloud_results/stage4_large_model_data_20260504/figures/global_loss_final.png",
        ),
        (
            "Table 9 compares seed 42, seed 43, and seed 44. Seed 42 is the original Stage-4C result, while seed 43 and seed 44 come from the robustness run.",
            "Figure 6 shows the seed-level forget-client loss changes, while Table 9 compares seed 42, seed 43, and seed 44. Seed 42 is the original Stage-4C result, while seed 43 and seed 44 come from the robustness run.",
            "Figure 6. Seed robustness forget-client loss comparison.",
            "formal_cloud_results/stage4_seed_robustness_qwen15b_dsample06_core5_20260504/figures/forget_loss_before_after.png",
        ),
    ]
    for old_anchor, new_anchor, caption, path in figures:
        text = remove_figure_block(text, caption, path)
        if new_anchor not in text and old_anchor in text:
            text = text.replace(old_anchor, new_anchor, 1)
        text = insert_after_once(text, new_anchor, figure_block(caption, path))
    return text


def normalize_english_headings(text: str) -> str:
    replacements = {
        "## 4. Experimental Setup": "## 4. Experimental Design and Evaluation Protocol",
        "### 4.1 Model and Dataset": "### 4.1 Instruction-tuning Dataset and Base Language Models",
        "### 4.2 Experimental Stages": "### 4.2 Four-stage Experimental Design and Robustness Check",
        "### 4.3 Compared Settings": "### 4.3 Compared Unlearning Settings",
        "### 4.4 Metrics": "### 4.4 Evaluation Metrics and Decision Criteria",
        "## 5. Results and Analysis": "## 5. Experimental Results and Mechanism Analysis",
        "### 5.1 Stage 1: Minimal Cloud Validation": "### 5.1 Stage 1: Minimal Cloud Loop and Pipeline Feasibility",
        "### 5.2 Stage 2: Main Trade-off Experiment": "### 5.2 Stage 2: Utility-Forgetting Trade-off between Forget-batch and Retained-set Curvature",
        "### 5.3 Stage 3: Full Direction and Guard Ablation": "### 5.3 Stage 3: Direction Misalignment Diagnosis and Guard Ablation",
        "### 5.4 Stage 4: Model and Data Scale Generalization": "### 5.4 Stage 4: Generalization across Model Scale and Data Scale",
        "#### Stage 4A: Small model + large data": "#### Stage 4A: Full results under the small-model/large-data setting",
        "#### Stage 4B: Large model + small data": "#### Stage 4B: Full results under the large-model/small-data setting",
        "#### Stage 4C: Large model + large data": "#### Stage 4C: Full results under the large-model/large-data setting",
        "### 5.5 Seed Robustness Check on the Large-Model/Large-Data Setting": "### 5.5 Seed Robustness under the Large-model/Large-data Setting",
        "### 5.6 Cross-stage Discussion": "### 5.6 Cross-stage Discussion and Method Positioning",
        "## 6. Limitations": "## 6. Method Limitations and Applicability Boundaries",
        "## 7. Conclusion": "## 7. Conclusion and Future Work",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_english_fedhds_terms(text: str) -> str:
    replacements = {
        "# FedHDS-style Federated Unlearning with Retained-Set Curvature and Guarded Trust-Region Updates": "# Guarded Federated Unlearning with Retained-Set Curvature over a Data-Filtering Training Backbone",
        "FedHDS-style federated filtering backbone": "implemented data-filtering federated training backbone",
        "FedHDS-style filtering backbone": "implemented data-filtering federated training backbone",
        "FedHDS-style federated unlearning": "data-filtering-backbone federated unlearning",
        "FedHDS-style federated": "data-filtering federated",
        "FedHDS-style Federated": "Data-filtering Federated",
        "FedHDS-style Federated Fine-tuning Backbone": "Data-filtering Federated Fine-tuning Backbone",
        "FedHDS-style training backbone": "implemented data-filtering federated training backbone",
        "FedHDS-style backbone": "data-filtering training backbone",
        "FedHDS-style model": "data-filtering baseline model",
        "FedHDS-style pipeline": "implemented data-filtering pipeline",
        "FedHDS-style": "data-filtering",
        "FedHDS model": "data-filtering baseline model",
        "FedHDS pre-unlearning": "data-filtering baseline pre-unlearning",
        "FedHDS final": "data-filtering baseline final",
        "FedHDS global": "data-filtering baseline global",
        "FedHDS baseline": "data-filtering baseline",
        "FedHDS utility baseline": "data-filtering utility baseline",
        "| fedhds |": "| data-filtering baseline |",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\bFedHDS\b", "data-filtering baseline", text)
    text = text.replace("an implemented implemented", "an implemented")
    text = text.replace("a implemented", "an implemented")
    text = text.replace("implemented implemented", "implemented")
    return text


def normalize_chinese_terms(text: str) -> str:
    replacements = [
        ("# 基于 FedHDS-style 训练底座、保留集曲率和受控信赖域更新的联邦遗忘方法", "# 基于数据过滤式联邦训练底座、保留集曲率和受控信赖域更新的联邦遗忘方法"),
        ("FedHDS-style 联邦微调训练底座", "数据过滤式联邦微调训练底座"),
        ("FedHDS-style 联邦过滤训练底座", "数据过滤式联邦训练底座"),
        ("FedHDS-style 过滤训练底座", "数据过滤式联邦训练底座"),
        ("FedHDS-style 训练底座", "数据过滤式联邦训练底座"),
        ("FedHDS-style 联邦遗忘", "基于数据过滤式训练底座的联邦遗忘"),
        ("FedHDS-style 起点", "数据过滤式训练基线起点"),
        ("FedHDS-style", "数据过滤式"),
        ("这里的 数据过滤式 指本文代码中的本地训练底座，不作为外部论文引用。", "这里的“数据过滤式联邦训练底座”指本文实现的训练流程，用于提供遗忘前模型，并不作为单独外部算法贡献进行引用。"),
        ("这里的 数据过滤式 指本文代码中的本地训练流程，不作为外部论文引用。", "这里的“数据过滤式联邦训练底座”指本文实现的训练流程，用于提供遗忘前模型，并不作为单独外部算法贡献进行引用。"),
        ("FedHDS utility baseline", "数据过滤式效用基线"),
        ("FedHDS baseline", "数据过滤式训练基线"),
        ("FedHDS global loss", "数据过滤式训练基线全局损失"),
        ("FedHDS global", "数据过滤式训练基线全局损失"),
        ("FedHDS final", "数据过滤式训练基线最终结果"),
        ("FedHDS 的", "数据过滤式训练基线的"),
        ("FedHDS 为", "数据过滤式训练基线为"),
        ("FedHDS 中", "数据过滤式训练基线中"),
        ("FedHDS 状态", "数据过滤式训练基线状态"),
        ("FedHDS", "数据过滤式训练基线"),
        ("fedhds", "数据过滤式训练基线"),
        ("forget-client loss before/after", "目标客户端遗忘前后损失"),
        ("forget-client loss change", "目标客户端损失变化"),
        ("forget-client loss", "目标客户端损失"),
        ("forget-client gradient", "目标客户端梯度"),
        ("forget-client 设置", "目标客户端设置"),
        ("forget loss", "目标客户端损失"),
        ("Forget loss", "目标客户端损失"),
        ("forget-loss gain", "目标客户端损失增量"),
        ("forget-loss 约束", "目标客户端损失约束"),
        ("forget-loss guard", "目标客户端损失保护机制"),
        ("forget_loss_guard", "目标客户端损失保护机制"),
        ("forget_loss", "目标客户端损失"),
        ("forget client", "目标客户端"),
        ("forget client", "目标客户端"),
        ("forget-batch Hessian", "遗忘批次 Hessian"),
        ("Forget-batch Hessian", "遗忘批次 Hessian"),
        ("遗忘客户端 batch", "遗忘客户端批次"),
        ("目标客户端 batch", "目标客户端批次"),
        ("forget-batch", "遗忘批次"),
        ("final global loss", "最终全局损失"),
        ("Final global loss", "最终全局损失"),
        ("global loss change", "全局损失变化"),
        ("global loss", "全局损失"),
        ("Global loss", "全局损失"),
        ("global utility preservation", "全局效用保持"),
        ("global utility", "全局效用"),
        ("global guard", "全局效用保护机制"),
        ("global_loss_guard", "全局损失保护机制"),
        ("global_loss", "全局损失"),
        ("guard threshold", "保护阈值"),
        ("guard behavior", "保护机制行为"),
        ("guard calibration", "保护阈值校准"),
        ("guard steps", "保护机制步数"),
        ("guard 接受步数", "保护机制接受步数"),
        ("guard 阈值", "保护阈值"),
        ("guard 检查", "保护机制检查"),
        ("guard 机制", "保护机制"),
        ("guard 判断", "保护机制判断"),
        ("guard 拆分", "保护机制拆分"),
        ("guard 小步", "保护机制小步"),
        ("guard 下", "保护机制下"),
        ("guard 停止", "保护机制停止"),
        ("guard 截断", "保护机制截断"),
        ("guard 消融", "保护机制消融"),
        ("guard 行为", "保护机制行为"),
        ("guard 配合", "保护机制配合"),
        ("guard 控制", "保护机制控制"),
        ("guard 接受情况", "保护机制接受情况"),
        ("guard 尝试步数", "保护机制尝试步数"),
        ("该 guard", "该保护机制"),
        ("分步 guard", "分步保护机制"),
        ("分步信赖域 guard", "分步信赖域保护机制"),
        ("信赖域 guard", "信赖域保护机制"),
        ("分步信赖域 guard", "分步信赖域保护机制"),
        ("全局效用 guard", "全局效用保护机制"),
        ("不同 guard", "不同保护机制"),
        ("guard 策略", "保护机制策略"),
        ("guard 是否", "保护机制是否"),
        ("guard 使用", "保护机制使用"),
        ("、guard、", "、保护机制、"),
        ("与 guard", "与保护机制"),
        ("对 guard", "对保护机制"),
        ("这说明 guard", "这说明保护机制"),
        ("guard 在", "保护机制在"),
        ("+ guard", "+ 保护机制"),
        ("加入 guard 后", "加入保护机制后"),
        ("auto guard", "自动保护阈值"),
        ("retained-set curvature approximation", "保留集曲率近似"),
        ("retained-set curvature", "保留集曲率"),
        ("retained-set Hessian", "保留集 Hessian"),
        ("Retained-set Hessian", "保留集 Hessian"),
        ("retained-set update", "保留集更新"),
        ("retained-set guard", "保留集保护机制"),
        ("retained-set negative update", "保留集负方向更新"),
        ("retained-set negative", "保留集负方向"),
        ("retained-set", "保留集"),
        ("retained-negative update", "保留集负方向更新"),
        ("retained-negative", "保留集负方向"),
        ("retained-client reference set", "保留客户端参考集"),
        ("retained-client reference", "保留客户端参考"),
        ("retained reference set", "保留参考集"),
        ("retained reference", "保留参考"),
        ("retained negative norm010", "保留集负方向 norm010"),
        ("retained negative", "保留集负方向"),
        ("negative-sign correction", "负方向修正"),
        ("negative-sign update norm sweep", "负方向更新范数扫描"),
        ("negative-sign norm sweep", "负方向范数扫描"),
        ("negative-sign sweep", "负方向扫描"),
        ("negative-sign", "负方向"),
        ("negative sign", "负方向"),
        ("negative direction", "负方向"),
        ("direction probing", "方向探测"),
        ("direction validation", "方向验证"),
        ("direction misalignment", "方向错配"),
        ("positive/auto retained-set", "正向/自动保留集"),
        ("positive/auto", "正向/自动"),
        ("loss-based unlearning proxy", "基于损失的遗忘代理指标"),
        ("utility-preserving unlearning path", "效用保持型遗忘路径"),
        ("utility-preserving unlearning framework", "效用保持型遗忘框架"),
        ("utility-preserving 的受控联邦遗忘框架", "效用保持型受控联邦遗忘框架"),
        ("utility-preserving", "效用保持型"),
        ("utility preservation", "效用保持"),
        ("utility baseline", "效用基线"),
        ("utility control", "效用控制"),
        ("utility baseline", "效用基线"),
        ("utility", "效用"),
        ("unlearning update", "遗忘更新"),
        ("unlearning objective", "遗忘目标"),
        ("unlearning 框架", "遗忘框架"),
        ("unlearning path", "遗忘路径"),
        ("unlearning time", "遗忘耗时"),
        ("post-training unlearning", "训练后遗忘"),
        ("unlearning framework", "遗忘框架"),
        ("baseline model", "基线模型"),
        ("training baseline", "训练基线"),
        ("baseline 状态", "基线状态"),
        ("baseline", "基线"),
        ("trade-off", "权衡"),
        ("exact forgetting gain", "具体遗忘增益"),
        ("forgetting gain", "遗忘增益"),
        ("loss proxy", "损失代理指标"),
        ("loss 上升", "损失上升"),
        ("loss 上", "损失上"),
        ("loss 受控", "损失受控"),
        ("loss 接受边界", "损失接受边界"),
        ("loss 区间", "损失区间"),
        ("loss 阈值", "损失阈值"),
        ("loss 计算", "损失计算"),
        ("loss 变化", "损失变化"),
        ("loss 的", "损失的"),
        ("loss 指标", "损失指标"),
        ("loss 状态", "损失状态"),
        ("loss 代价", "损失代价"),
        ("loss 数值", "损失数值"),
        ("上的 loss", "上的损失"),
        ("平均 loss", "平均损失"),
        ("全局评估 loss", "全局评估损失"),
        ("全局 loss", "全局损失"),
        ("目标客户端 loss", "目标客户端损失"),
        ("update norm clipping", "更新范数裁剪"),
        ("norm clipping", "范数裁剪"),
        ("L2 范数 norm", "L2 范数"),
        ("norm=0.10", "最大范数=0.10"),
        ("norm010", "最大范数 0.10"),
        ("update norm", "更新范数"),
        ("Update L2", "更新 L2 范数"),
        ("update L2", "更新 L2 范数"),
        ("update L2", "更新 L2 范数"),
        ("update direction", "更新方向"),
        ("max_norm", "最大范数"),
        ("accepted update norm", "已接受更新范数"),
        ("accepted steps", "接受步数"),
        ("attempted steps", "尝试步数"),
        ("Stop reason", "停止原因"),
        ("stop reason", "停止原因"),
        ("Time (s)", "时间(s)"),
        ("Steps", "步数"),
        ("Source", "来源"),
        ("Setting", "设置"),
        ("Data sample", "数据采样"),
        ("data_sample", "数据采样比例"),
        ("20 rounds", "20 轮训练"),
        ("rounds", "训练轮次"),
        ("batch size", "批大小"),
        ("Hessian reference", "Hessian 参考来源"),
        ("Hessian/sign", "Hessian/方向"),
        ("reference set", "参考集"),
        ("reference", "参考"),
        ("Main reading", "主要解读"),
        ("Group", "组别"),
        ("Note", "说明"),
        ("Model", "模型"),
        ("Seed robustness check", "随机种子鲁棒性检查"),
        ("seed robustness check", "随机种子鲁棒性检查"),
        ("Seed robustness", "随机种子鲁棒性"),
        ("seed robustness", "随机种子鲁棒性"),
        ("seed-level", "随机种子层面"),
        ("seed 42", "随机种子 42"),
        ("seed 43", "随机种子 43"),
        ("seed 44", "随机种子 44"),
        ("seed-sensitive", "对随机种子敏感"),
        ("随机种子-sensitive", "对随机种子敏感"),
        ("seed", "随机种子"),
        ("Stage 3:", "阶段三："),
        ("Stage 4A", "阶段四A"),
        ("Stage 4B", "阶段四B"),
        ("Stage 4C", "阶段四C"),
        ("Stage-4C", "阶段四C"),
        ("Stage-2", "阶段二"),
        ("Stage-3", "阶段三"),
        ("Stage-4", "阶段四"),
        ("Small model + large data", "小模型 + 大数据"),
        ("Large model + small data", "大模型 + 小数据"),
        ("Large model + large data", "大模型 + 大数据"),
        ("Small model + small data", "小模型 + 小数据"),
        ("small-model/large-data", "小模型/大数据"),
        ("large-model/small-data", "大模型/小数据"),
        ("large-model/large-data", "大模型/大数据"),
        ("2 x 2", "2×2"),
        ("trust-region style update", "信赖域式更新"),
        ("stepped trust-region", "分步信赖域"),
        ("paper table 和 figures", "表格和图像"),
        ("paper table and figures", "表格和图像"),
        ("guarded update", "受控更新"),
        ("guarded", "受控"),
        ("summary", "汇总"),
        ("client 0", "客户端 0"),
        ("sign=positive", "方向=正向"),
        ("sign=negative", "方向=负向"),
        ("completed", "完成"),
        ("目标客户端损失 increased", "目标客户端损失上升"),
        ("效用 preserved, weak forgetting", "效用保持，但遗忘较弱"),
        ("best probe sign negative", "方向探测选择负向"),
        ("negative update", "负方向更新"),
        ("corrected 保留集负方向更新", "修正后的保留集负方向更新"),
        ("retained_auto_direction_guard05", "保留集正向自动设置"),
        ("mechanism diagnosis and corrected 负方向", "机制诊断与负方向修正"),
        ("large model + large data preserves the 保留集负方向 权衡", "大模型 + 大数据设置保持保留集负方向权衡"),
        ("效用 remains close to 数据过滤式训练基线, but retained 遗忘增益 is 对随机种子敏感", "效用接近数据过滤式训练基线，但保留集遗忘增益对随机种子敏感"),
        ("目标客户端损失保护机制 rejects the 负方向更新 and preserves the 数据过滤式训练基线 state", "目标客户端损失保护机制拒绝负方向更新，并保留数据过滤式训练基线状态"),
        ("random seed", "随机种子"),
        ("delta ", "探测差值 "),
        ("Influence functions", "影响函数"),
        ("SISA training", "SISA 训练"),
        ("teacher-student memory generation", "教师-学生记忆生成"),
        ("shard 和 slice", "分片和切片"),
        ("LiSSA-style", "LiSSA 风格"),
        ("Hessian-vector product，也就是 HVP", "Hessian 向量积（HVP）"),
        ("Hessian-vector product", "Hessian 向量积"),
        ("inverse-Hessian-vector product", "逆 Hessian 向量积"),
        ("inverse-Hessian 向量积", "逆 Hessian 向量积"),
        ("inverse-Hessian 方向", "逆 Hessian 方向"),
        ("inverse-HVP", "逆 HVP"),
        ("Inverse-Hessian", "逆 Hessian"),
    ]
    def apply_replacements(segment: str) -> str:
        for _ in range(2):
            for old, new in replacements:
                segment = segment.replace(old, new)
        return segment

    normalized_lines: list[str] = []
    image_pattern = re.compile(r"^(\s*!\[)(.*?)(\]\()(.+?)(\)\s*)$")
    for line in text.splitlines():
        match = image_pattern.match(line)
        if match:
            normalized_lines.append(
                f"{match.group(1)}{apply_replacements(match.group(2))}{match.group(3)}{match.group(4)}{match.group(5)}"
            )
        else:
            normalized_lines.append(apply_replacements(line))
    return "\n".join(normalized_lines) + ("\n" if text.endswith("\n") else "")


def polish_chinese(text: str) -> str:
    text = normalize_chinese_terms(text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[，。；：、（“《])[ \t]+", "", text)
    text = re.sub(r"[ \t]+(?=[，。；：、）”》])", "", text)
    text = re.sub(r"数据过滤式训练基线-style", "数据过滤式", text)
    text = re.sub(r"数据过滤式训练基线\s+训练基线", "数据过滤式训练基线", text)
    text = re.sub(r"效用\s+效用", "效用", text)
    text = re.sub(r"损失\s+损失", "损失", text)
    text = text.replace("Hessian 向量积（HVP，Hessian 向量积）", "Hessian 向量积（HVP）")
    text = text.replace("数据过滤式训练基线最终结果全局损失", "数据过滤式训练基线最终全局损失")
    text = text.replace("数据过滤式训练基线全局损失 loss", "数据过滤式训练基线全局损失")
    text = text.replace("全局 损失", "全局损失")
    text = text.replace("目标客户端 损失", "目标客户端损失")
    text = text.replace("遗忘客户端 损失", "遗忘客户端损失")
    text = text.replace("分步 保护机制", "分步保护机制")
    text = text.replace("不同保护机制 策略", "不同保护机制策略")
    text = text.replace("stepped 全局效用保护机制", "分步全局效用保护机制")
    text = text.replace("accepted 更新范数", "已接受更新范数")
    text = text.replace("paper table and figures", "表格和图像")
    text = text.replace("paper table 和 图像", "表格和图像")
    text = text.replace("figures 可以形成闭环", "图像可以形成闭环")
    text = text.replace("paper table 和 图像", "表格和图像")
    text = text.replace("tolerance 设置", "容忍阈值设置")
    text = text.replace("两个 容忍阈值设置", "两个容忍阈值设置")
    text = text.replace("norm sweep", "范数扫描")
    text = text.replace("最大最大范数", "最大范数")
    text = text.replace("guard 尝试步数", "保护机制尝试步数")
    text = text.replace("guarded update", "受控更新")
    text = text.replace("分步信赖域 guard", "分步信赖域保护机制")
    text = text.replace("信赖域 guard", "信赖域保护机制")
    return text


def polish_english(text: str) -> str:
    text = replace_abstract(text)
    text = replace_english_research_status(text)
    text = re.sub(
        r"## 3\. (?:Method|Retained-Set Curvature Based Guarded Federated Unlearning Method)\n.*?(?=\n## 4\.)",
        EN_METHOD_TEXT + "\n\n",
        text,
        flags=re.S,
    )
    text = text.replace(
        "This work builds on a FedHDS training backbone and studies how to perform controlled unlearning after federated training [FedHDS]. FedHDS is used as the training backbone because it provides a practical federated fine-tuning pipeline with data filtering or selection.",
        "This work builds on an implemented FedHDS-style filtering backbone and studies how to perform controlled unlearning after federated training. We use the term FedHDS-style to describe the local training backbone implemented in this project, not as a citation to an external paper. The backbone provides a practical federated fine-tuning pipeline with data filtering or selection.",
    )
    text = text.replace("FedHDS-assisted", "FedHDS-style")
    text = text.replace("FedHDS training backbone", "FedHDS-style filtering backbone")
    text = text.replace("FedHDS pipeline", "implemented FedHDS-style pipeline")
    text = text.replace("FedHDS is used", "the FedHDS-style backbone is used")
    text = text.replace("FedHDS is not itself", "the FedHDS-style backbone is not itself")
    text = text.replace("The training stage follows the existing FedHDS-style pipeline.", "The training stage follows the implemented FedHDS-style filtering pipeline.")
    text = text.replace(
        "This work builds on a FedHDS-style filtering backbone and studies how to perform controlled unlearning after federated training [FedHDS]. the FedHDS-style backbone is used as the training backbone because it provides a practical federated fine-tuning pipeline with data filtering or selection.",
        "This work builds on an implemented FedHDS-style filtering backbone and studies how to perform controlled unlearning after federated training. We use the term FedHDS-style to describe the local training backbone implemented in this project, not as a citation to an external paper. The backbone provides a practical federated fine-tuning pipeline with data filtering or selection.",
    )
    text = text.replace(
        "In this paper, FedHDS is not itself the unlearning mechanism.",
        "In this paper, the FedHDS-style backbone is not itself the unlearning mechanism.",
    )
    text = text.replace(
        "Experiments use Dolly as the federated instruction-tuning dataset. Stages 1-3 use Qwen2-0.5B because it is large enough to represent a language-model fine-tuning setting while still feasible for repeated Hessian-vector product experiments on a single RTX 5090 32GB GPU. Stage 4 additionally uses Qwen2.5-1.5B to evaluate model-scale generalization.",
        "Experiments use Dolly as the federated instruction-tuning dataset [Dolly]. Stages 1-3 use Qwen2-0.5B [Qwen2] because it is large enough to represent a language-model fine-tuning setting while still feasible for repeated Hessian-vector product experiments on a single RTX 5090 32GB GPU. Stage 4 additionally uses Qwen2.5-1.5B [Qwen2.5] to evaluate model-scale generalization.",
    )
    text = re.sub(
        r"Experiments use Dolly as the federated instruction-tuning dataset <sup>\[\d+\]</sup>\. Stages 1-3 use Qwen2-0\.5B <sup>\[\d+\]</sup> because it is large enough to represent a language-model fine-tuning setting while still feasible for repeated Hessian-vector product experiments on a single RTX 5090 32GB GPU\. Stage 4 additionally uses Qwen2\.5-1\.5B <sup>\[\d+\]</sup> to evaluate model-scale generalization\.",
        "Experiments use Dolly as the federated instruction-tuning dataset [Dolly]. Stages 1-3 use Qwen2-0.5B [Qwen2] because it is large enough to represent a language-model fine-tuning setting while still feasible for repeated Hessian-vector product experiments on a single RTX 5090 32GB GPU. Stage 4 additionally uses Qwen2.5-1.5B [Qwen2.5] to evaluate model-scale generalization.",
        text,
    )
    text = text.replace(
        "This is why the method is described as a trust-region style update: the model is only allowed to move as long as the measured global utility stays within the specified region.",
        "This is why the method is described as a trust-region style update [Trust-Region]: the model is only allowed to move as long as the measured global utility stays within the specified region.",
    )
    text = text.replace(
        "Federated unlearning studies how to remove the influence of a target client or target data subset from a trained federated model. The ideal reference is retraining from scratch without the target client.",
        "Federated unlearning studies how to remove the influence of a target client or target data subset from a trained federated model [Federated-Unlearning-Survey]. The ideal reference is retraining from scratch without the target client.",
    )
    text = text.replace(
        "Trust-region style ideas limit how far the model can move before checking utility.",
        "Trust-region style ideas [Trust-Region] limit how far the model can move before checking utility.",
    )
    captions = [
        ("Table 1 reports the Stage-1 results.", "Table 1. Stage-1 minimal cloud validation results."),
        ("Table 2 reports the Stage-2 results.", "Table 2. Stage-2 main trade-off results."),
        ("Table 3 reports the full Stage-3 ablation.", "Table 3. Full Stage-3 direction and guard ablation."),
        ("Table 4 focuses on the negative-sign norm sweep.", "Table 4. Negative-sign update-norm sweep in Stage 3."),
        ("Table 5 summarizes the scale matrix using three comparable entries from each cell: the FedHDS utility baseline, the forget-batch Hessian baseline, and the retained-set negative norm010 configuration.", "Table 5. Model/data scale generalization matrix."),
        ("Table 9 compares seed 42, seed 43, and seed 44. Seed 42 is the original Stage-4C result, while seed 43 and seed 44 come from the robustness run.", "Table 9. Seed robustness check on the large-model/large-data setting."),
    ]
    for phrase, caption in captions:
        text = ensure_caption(text, phrase, caption)
    text = text.replace(
        "Table 6 compares seed 42, seed 43, and seed 44.",
        "Table 9 compares seed 42, seed 43, and seed 44.",
    )
    text = text.replace(
        "Table 6. Seed robustness check on the large-model/large-data setting.",
        "Table 9. Seed robustness check on the large-model/large-data setting.",
    )
    text = text.replace(
        "The complete Stage-4 result tables are shown below.",
        "The complete Stage-4 result tables are shown in Tables 6-8.",
    )
    for heading, caption in [
        ("#### Stage 4A: Full results under the small-model/large-data setting", "Table 6. Full Stage-4A results for the small-model/large-data setting."),
        ("#### Stage 4B: Full results under the large-model/small-data setting", "Table 7. Full Stage-4B results for the large-model/small-data setting."),
        ("#### Stage 4C: Full results under the large-model/large-data setting", "Table 8. Full Stage-4C results for the large-model/large-data setting."),
    ]:
        text = ensure_heading_table_caption(text, heading, caption)
    if "## Appendix A. Figure Captions" in text:
        text = re.sub(
            r"\n## Appendix A\. Figure Captions\n.*?(?=\n## References)",
            "\n" + appendix_figure_caption_list(zh=False).rstrip() + "\n",
            text,
            flags=re.S,
        )
    else:
        text = text.replace("\n## References", "\n" + appendix_figure_caption_list(zh=False) + "\n## References")
    text = insert_english_figures(text)
    text = re.sub(r"\n\n +(With max_norm)", r"\n\n\1", text)
    text = normalize_inline_images(text)
    text = translate_english_table_headers(text)
    text = normalize_english_headings(text)
    text = normalize_english_fedhds_terms(text)
    text = numeric_citations(text)
    return replace_references(text)


def extract_table_after(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx < 0:
        return ""
    lines = text[idx:].splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("|"):
            start = i
            break
    if start is None:
        return ""
    table_lines: list[str] = []
    for line in lines[start:]:
        if line.startswith("|"):
            table_lines.append(line)
        elif table_lines:
            break
    return "\n".join(table_lines)


def zh_table(table: list[list[str]], header_map: dict[str, str] | None = None) -> str:
    if not header_map:
        return table_md(table)
    mapped = [list(table[0]), *[list(row) for row in table[1:]]]
    mapped[0] = [header_map.get(cell, cell) for cell in mapped[0]]
    return table_md(mapped)


def zh_f4(value: str | None) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return str(value)


def zh_groups(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["group"]: row for row in rows}


def zh_final_global(row: dict[str, str] | None) -> str:
    if not row:
        return "-"
    return zh_f4(row.get("global_loss_after_unlearning") or row.get("final_global_loss"))


def zh_loss_change(row: dict[str, str] | None) -> str:
    if not row:
        return "-"
    before = row.get("forget_client_loss_before_unlearning", "")
    after = row.get("forget_client_loss_after_unlearning", "")
    if not before or not after:
        return "-"
    delta = float(after) - float(before)
    return f"{zh_f4(before)} -> {zh_f4(after)} ({delta:+.4f})"


def zh_global_change(row: dict[str, str] | None) -> str:
    if not row:
        return "-"
    before = row.get("global_loss_before_unlearning", "")
    after = row.get("global_loss_after_unlearning", "")
    if not before or not after:
        return "-"
    return f"{zh_f4(before)} -> {zh_f4(after)}"


def zh_steps(row: dict[str, str] | None) -> str:
    if not row:
        return "-"
    accepted = row.get("unlearn_steps_accepted", "")
    attempted = row.get("unlearn_steps_attempted", "")
    if not accepted or not attempted:
        return "-"
    return f"{int(float(accepted))}/{int(float(attempted))}"


def zh_update_summary(row: dict[str, str] | None) -> str:
    if not row:
        return "-"
    norm = row.get("actual_param_delta_l2_norm", "")
    steps = zh_steps(row)
    if not norm and steps == "-":
        return "-"
    if norm:
        return f"L2={zh_f4(norm)}，步数={steps}"
    return f"步数={steps}"


def zh_stage3_main_table(rows: list[dict[str, str]]) -> list[list[str]]:
    g = zh_groups(rows)
    specs = [
        ("fl", "FL 训练基线", "训练基线", "普通联邦训练对照"),
        ("fedhds", "数据过滤式训练基线", "训练基线", "遗忘前主要起点"),
        ("forget_batch_hessian", "遗忘批次 Hessian", "遗忘批次/正向", "遗忘增强，但全局损失代价较大"),
        ("retained_auto_direction_guard05", "保留集正向自动", "保留集/正向", "效用安全，但遗忘方向错误"),
        ("retained_auto_forget_guard_tol0_guard05", "正向 + 目标损失保护", "保留集/正向+保护", "拒绝无效遗忘小步"),
        ("retained_negative_norm010_guard10", "保留集负方向 norm=0.10", "保留集/负向", "遗忘增益明显，效用代价可控"),
        ("retained_negative_norm015_guard10", "保留集负方向 norm=0.15", "保留集/负向", "遗忘更强，但被全局保护机制截断"),
    ]
    table = [["设置", "方向/机制", "目标客户端损失", "全局损失", "更新情况", "论文解读"]]
    for key, label, mechanism, reading in specs:
        row = g.get(key)
        table.append([
            label,
            mechanism,
            zh_loss_change(row),
            zh_final_global(row),
            zh_update_summary(row),
            reading,
        ])
    return table


def zh_negative_sweep_table(rows: list[dict[str, str]]) -> list[list[str]]:
    g = zh_groups(rows)
    specs = [
        ("retained_negative_norm005_guard05", "0.05", "遗忘较温和，效用损失较小"),
        ("retained_negative_norm010_guard10", "0.10", "遗忘增益更明显，是主文重点配置"),
        ("retained_negative_norm015_guard10", "0.15", "触发全局保护机制截断，不宜作为默认配置"),
    ]
    table = [["最大范数", "目标客户端损失", "全局损失变化", "接受步数", "说明"]]
    for key, max_norm, note in specs:
        row = g.get(key)
        table.append([max_norm, zh_loss_change(row), zh_global_change(row), zh_steps(row), note])
    return table


def zh_scale_key_table(stage3_rows: list[dict[str, str]], stage4_sets: list[dict[str, object]]) -> list[list[str]]:
    all_sets: list[dict[str, object]] = [{
        "tag": "阶段三",
        "title": "小模型 + 小数据",
        "model": "Qwen2-0.5B",
        "data_sample": "0.4",
        "rows": stage3_rows,
    }]
    tag_map = {"Stage 4A": "阶段四A", "Stage 4B": "阶段四B", "Stage 4C": "阶段四C"}
    for item in stage4_sets:
        copied = dict(item)
        copied["tag"] = tag_map.get(str(item.get("tag")), str(item.get("tag")))
        all_sets.append(copied)
    readings = {
        "阶段三": "机制诊断：负方向修正有效",
        "阶段四A": "遗忘批次更强但效用代价大；保留集负方向更稳",
        "阶段四B": "大模型下遗忘批次失效，保留集负方向仍有小幅增益",
        "阶段四C": "大模型大数据下保留集负方向保持效用优势",
    }
    table = [["设置", "模型/数据", "训练基线全局损失", "遗忘批次结果", "保留集负方向结果", "主要结论"]]
    for item in all_sets:
        rows = item["rows"]  # type: ignore[index]
        g = zh_groups(rows)  # type: ignore[arg-type]
        fb = g.get("forget_batch_hessian")
        rn = g.get("retained_negative_norm010_guard10")
        tag = str(item["tag"])
        table.append([
            f"{tag}：{item['title']}",
            f"{item['model']} / 数据采样 {item['data_sample']}",
            zh_final_global(g.get("fedhds")),
            f"{zh_loss_change(fb)}，全局={zh_final_global(fb)}",
            f"{zh_loss_change(rn)}，全局={zh_final_global(rn)}",
            readings.get(tag, ""),
        ])
    return table


def zh_stage4_key_table(rows: list[dict[str, str]]) -> list[list[str]]:
    g = zh_groups(rows)
    specs = [
        ("fedhds", "数据过滤式训练基线", "遗忘前效用起点"),
        ("forget_batch_hessian", "遗忘批次 Hessian", "直接遗忘基线"),
        ("retained_auto_direction_guard05", "保留集自动方向", "方向探测后的保留集更新"),
        ("retained_negative_norm010_guard10", "保留集负方向 norm=0.10", "主文可比配置"),
    ]
    table = [["设置", "目标客户端损失", "最终全局损失", "更新情况", "说明"]]
    for key, label, note in specs:
        row = g.get(key)
        table.append([label, zh_loss_change(row), zh_final_global(row), zh_update_summary(row), note])
    return table


def zh_seed_key_table(stage4c_rows: list[dict[str, str]], seed_rows: list[dict[str, str]]) -> list[list[str]]:
    seed_sources = [
        ("42", stage4c_rows),
        ("43", rows_for_seed(seed_rows, "43")),
        ("44", rows_for_seed(seed_rows, "44")),
    ]
    readings = {
        "42": "保留集负方向提高目标客户端损失，且全局损失低于训练基线",
        "43": "效用接近训练基线，但遗忘增益对随机种子敏感",
        "44": "目标客户端损失保护机制拒绝负方向更新，模型保持在训练基线状态",
    }
    table = [["随机种子", "训练基线全局损失", "遗忘批次结果", "保留集负方向结果", "结论"]]
    for seed, rows in seed_sources:
        if not rows:
            continue
        g = zh_groups(rows)
        fb = g.get("forget_batch_hessian")
        rn = g.get("retained_negative_norm010_guard10")
        table.append([
            seed,
            zh_final_global(g.get("fedhds")),
            f"{zh_loss_change(fb)}，全局={zh_final_global(fb)}",
            f"{zh_loss_change(rn)}，全局={zh_final_global(rn)}，步数={zh_steps(rn)}",
            readings[seed],
        ])
    return table


def polish_chinese_basic_table(table_text: str) -> str:
    replacements = {
        "data-filtering baseline": "数据过滤式训练基线",
        "data-filtering 基线": "数据过滤式训练基线",
        "fl": "FL 训练基线",
        "forget_batch_hessian": "遗忘批次 Hessian",
        "retained_guard_ref1_guard170": "保留集保护机制（阈值 1.70）",
        "retained_guard_ref1_autoguard": "保留集自动保护机制",
        "eta=": "步长=",
        "steps=": "小步数=",
        "global_guard=": "全局阈值=",
        "none": "无",
        "效用基线": "效用基线",
    }
    for old, new in replacements.items():
        table_text = table_text.replace(old, new)
    return table_text


def make_chinese_paper(en_text: str) -> str:
    rows = read_csv_rows(STAGE3 / "summary.csv")
    stage4_sets = load_stage4_sets()
    seed_rows = load_seed_rows()
    stage1_table = polish_chinese_basic_table(extract_table_after(en_text, "Table 1 reports the Stage-1 results."))
    stage2_table = polish_chinese_basic_table(extract_table_after(en_text, "Table 2 reports the Stage-2 results."))
    scale = zh_scale_key_table(rows, stage4_sets)
    stage4c_rows: list[dict[str, str]] = []
    for item in stage4_sets:
        if item.get("tag") == "Stage 4C":
            stage4c_rows = item["rows"]  # type: ignore[assignment]
            break

    parts: list[str] = []
    parts.append(r"""# 基于 FedHDS-style 训练底座、保留集曲率和受控信赖域更新的联邦遗忘方法

## 摘要

联邦遗忘旨在不完整重训联邦模型的前提下削弱目标客户端对全局模型的影响。在语言模型联邦指令微调场景中，近似遗忘既需要降低模型对目标客户端数据的拟合程度，又需要保持保留客户端上的全局效用。针对这一问题，本文提出一种基于数据过滤式联邦训练底座的受控联邦遗忘方法。该方法以目标客户端梯度刻画待削弱信号，以保留客户端小规模参考集估计二阶曲率，并通过全局范数裁剪、分步信赖域检查、方向探测和目标客户端损失保护机制约束遗忘更新。本文在 Dolly 数据集、Qwen2-0.5B 和 Qwen2.5-1.5B 模型上开展四阶段实验和随机种子鲁棒性检查。实验结果表明，遗忘批次 Hessian 更新能够产生更强的目标客户端损失增量，但容易带来明显的全局损失代价；保留集曲率在方向修正和保护机制约束下，可以在全局损失受控条件下提高目标客户端损失，并表现出更好的效用保持特征。进一步的模型规模、数据规模和随机种子实验说明，本文方法更适合作为一种可审计、效用保持型的近似联邦遗忘框架，而不应被解释为所有设置下遗忘强度最强的算法；具体遗忘增益仍受模型规模、数据规模和随机种子影响。

**关键词：** 联邦遗忘；保留集曲率；Hessian 向量积；信赖域保护机制；语言模型微调

## 1 引言

联邦学习允许多个客户端在不直接共享本地原始数据的情况下共同训练全局模型 [FedAvg]。这种训练范式适用于数据分散在不同用户、机构或设备上的场景。然而，联邦模型训练完成后仍可能面对遗忘请求：某个客户端可能撤回授权、要求删除数据，或因数据质量和合规原因需要从模型中移除其影响。

最直接的方案是在排除目标客户端后重新训练模型 [Machine-Unlearning]。但在联邦语言模型微调中，完整重训需要重复大量通信轮次、本地更新和模型保存，代价很高。因此，联邦遗忘研究如何在不完整重训的情况下近似移除目标客户端影响。

本文关注遗忘强度与保留效用之间的 trade-off。遗忘更新应当提高或至少不降低目标客户端上的 loss，同时又不能显著破坏保留客户端上的全局效用。二阶近似方法可以通过 Hessian 或 inverse-Hessian-vector product 估计参数修正，但 Hessian reference 的选择会直接影响更新方向和更新幅度。forget-batch Hessian 与遗忘目标更直接对齐，却可能损害全局效用；retained-set Hessian 更贴近保留任务分布，却可能出现遗忘方向错配。

本文在本项目实现的 FedHDS-style 过滤训练底座上构建联邦遗忘流程。这里的 FedHDS-style 指本文代码中的本地训练底座，不作为外部论文引用。核心思想是：用 forget-client gradient 表示“要移除什么”，用 retained-client reference set 的曲率表示“模型应尽量保留在哪个任务区域附近”。随后通过 update norm clipping、stepped trust-region guard、direction probing 和 forget-loss guard 对二阶更新进行约束。本文的目标不是宣称 retained-set Hessian 在所有情况下都优于 forget-batch Hessian，而是证明 retained-set curvature 在方向验证和 guard 机制配合下，可以形成更可控、可审计的 utility-preserving unlearning 框架。

本文贡献如下：

- 构建了面向语言模型微调的 FedHDS-style 联邦遗忘完整实验链路，包括训练、遗忘、汇总、表格和图像生成。
- 提出 retained-set curvature approximation，将遗忘梯度来源与曲率参考来源分离。
- 使用全局更新范数裁剪和分步信赖域 guard，使二阶遗忘更新可度量、可回滚、可审计。
- 诊断 retained-set update direction misalignment，并证明 forget-loss guard 与 negative-sign correction 能在 global loss 受控时提高 forget-client loss。
- 通过 2 x 2 模型/数据规模矩阵验证该 trade-off 在当前 Dolly/Qwen 设置下具有一定泛化性。
- 通过大模型大数据设置下的 seed robustness check，说明 utility control 与 guard 行为较稳定，但 exact forgetting gain 对 seed 敏感。

""" + ZH_RESEARCH_STATUS + r"""

## 3 基于保留集曲率的受控联邦遗忘方法

本章介绍本文提出的 FedHDS-style 联邦遗忘方法。该方法先通过 FedHDS-style 过滤训练底座得到遗忘前全局模型，再用遗忘客户端梯度刻画需要削弱的信号，用保留客户端参考集估计二阶曲率，并通过范数裁剪、方向探测、分步全局效用检查和 forget-loss guard 控制最终更新。

从整体流程看，本文方法并不是重新设计联邦训练算法，而是在已有训练模型之后增加一个可审计的 post-training unlearning 模块。训练底座负责得到具有基本效用的全局模型，遗忘模块负责在目标客户端提出删除请求后，对该模型进行有限幅度的参数修正。这样的拆分有两个好处：第一，可以把“训练效果”和“遗忘效果”分开分析；第二，可以在相同的 FedHDS-style 起点上公平比较 forget-batch Hessian、retained-set Hessian 和不同 guard 策略。

本文方法可以概括为三个阶段。第一阶段计算遗忘客户端梯度，用它表示模型中需要被削弱的目标信号。第二阶段从保留客户端构造参考集，并用该参考集估计曲率，从而让二阶更新更多受到保留任务分布约束。第三阶段对候选更新进行范数裁剪、方向探测和分步 guard 检查。只有当候选更新同时满足全局效用约束和 forget-loss 约束时，参数更新才会被真正写入模型。

### 3.1 问题定义与优化目标

设联邦系统中共有 K 个客户端。客户端 k 拥有本地数据 Dₖ。联邦训练结束后，服务器得到遗忘前全局参数 θ*。目标客户端 j 提出遗忘请求，遗忘算法输出遗忘后的全局参数 θᵤ。

本文有两个目标：第一，削弱客户端 j 对模型的影响；第二，保留其他客户端上的全局效用。

**公式 (1)：遗忘客户端 loss 变化**

ΔLⱼ = Lⱼ(θᵤ) - Lⱼ(θ*)

变量说明：

- ΔLⱼ：遗忘前后 forget-client loss 的变化。
- Lⱼ(θ*)：遗忘前模型在目标客户端 j 上的平均 loss。
- Lⱼ(θᵤ)：遗忘后模型在目标客户端 j 上的平均 loss。

解释：如果 ΔLⱼ 为正，说明遗忘后模型在目标客户端上表现变差，这是本文希望看到的遗忘方向。如果该值为负，说明更新方向错误，模型反而更适应目标客户端。

需要说明的是，本文使用 forget-client loss 作为遗忘强度的可观测代理指标。该指标不能等价于严格的隐私证明，也不能保证目标客户端的所有信息都被完全删除，但它可以直接反映模型对目标客户端数据的拟合程度是否下降。因此，本文不会单独用 ΔLⱼ 判断方法优劣，而是始终把 forget-client loss change 与 global loss change 放在一起分析。一个合格的遗忘更新应当在提高或至少不降低 forget-client loss 的同时，避免全局效用出现不可控损失。

**公式 (2)：全局效用 guard**

Lᴳ(θᵤ) ≤ γᴳ

变量说明：

- Lᴳ(θᵤ)：遗忘后模型的全局评估 loss。
- γᴳ：允许的最大全局 loss 阈值。

解释：遗忘更新只有在不超过全局效用阈值时才会被接受。该约束用于防止遗忘更新破坏保留客户端上的整体效用。

全局效用 guard 把遗忘问题从单纯的参数修正问题转化为受约束更新问题。如果只追求提高目标客户端 loss，较大的二阶更新可能很容易产生遗忘信号，但也可能显著破坏保留客户端上的语言建模能力。引入 γᴳ 后，更新必须在可接受的全局 loss 区间内进行。对于语言模型微调而言，这一点尤其重要，因为模型参数高度耦合，小范围参数变化也可能在全局 loss 上产生明显影响。

### 3.2 FedHDS-style 联邦微调训练底座

训练阶段使用本项目实现的 FedHDS-style 过滤训练底座。这里的 FedHDS-style 指本文代码中的本地训练流程，不作为外部论文引用。在每轮通信中，被选中的客户端进行本地微调，服务器聚合客户端更新。

遗忘方法发生在训练结束之后。因此，FedHDS-style 训练底座负责得到 θ*，而本文提出的 retained-set curvature 方法负责由 θ* 构造 θᵤ。

因此，本文中的 FL 和 FedHDS 主要是训练阶段 baseline，而 forget-batch Hessian 与 retained-set Hessian 是遗忘阶段 baseline。FedHDS-style 训练底座得到更好的 global loss 并不意味着已经完成遗忘，它只是为后续遗忘更新提供更稳定的起点。真正需要比较的是：在相同遗忘前模型 θ* 上，不同 unlearning update 能否在削弱目标客户端影响的同时保持保留客户端效用。

### 3.3 作为遗忘信号的目标客户端梯度

遗忘客户端定义“要移除什么”。服务器在目标客户端 j 的数据上计算平均梯度。

**公式 (3)：遗忘客户端梯度**

gⱼ = ∇θ Lⱼ(θ*)

变量说明：

- gⱼ：来自遗忘客户端的梯度信号。
- j：被遗忘客户端编号。
- Lⱼ(θ*)：遗忘前目标客户端上的 loss。

解释：gⱼ 表示模型中与目标客户端相关的局部方向。但本文并不直接把它当成普通一阶更新，而是把它和曲率估计结合，形成二阶近似遗忘更新。

### 3.4 保留客户端参考集与曲率估计

本文的关键设计是：梯度来自遗忘客户端，但曲率来自保留客户端。这样可以把“删什么”和“模型应留在哪个任务区域附近”分开。

**公式 (4)：保留参考集**

Dref = ⋃(k≠j) Sₖ,  Sₖ ⊂ Dₖ

变量说明：

- Dref：用于估计曲率的保留参考集。
- Sₖ：从保留客户端 k 中采样得到的参考子集。
- Dₖ：客户端 k 的本地数据集。
- j：目标遗忘客户端。
- 保留客户端：除 j 之外的所有客户端。

公式 (4) 表示，保留参考集只由保留客户端提供的数据子集组成。它不是全部保留训练数据，而是用于估计曲率的小规模参考集。在主实验中，每个保留客户端贡献 1 个参考样本。总客户端数为 20、遗忘 1 个客户端时，Dref 的大小为 19。

使用小规模保留参考集是一种计算成本和曲率质量之间的折中。参考集越大，曲率估计可能越稳定，但 HVP 计算和显存压力也会明显增加；参考集越小，计算更轻量，但估计噪声更大。本文主实验采用每个保留客户端 1 个参考样本，是为了在单卡 GPU 条件下保持实验可运行，并把分析重点放在曲率来源、方向选择和 guard 机制本身，而不是追求最昂贵的二阶近似。

**公式 (5)：保留集曲率方向**

v = Href⁻¹ gⱼ

变量说明：

- Href：由保留参考集近似得到的 Hessian。
- gⱼ：遗忘客户端梯度。
- v：用于遗忘的二阶近似方向。

解释：遗忘客户端梯度说明要削弱哪个方向，保留集曲率说明沿哪个参数几何方向移动更不容易破坏保留任务效用。

这也是本文方法与 forget-batch Hessian baseline 的核心区别。forget-batch Hessian 中，梯度和曲率都来自遗忘客户端 batch，因此它天然更贴近遗忘目标，但也更容易沿着只服务目标客户端 loss 的方向移动。retained-set curvature 中，梯度仍然来自遗忘客户端，但曲率参考来自保留客户端，目的是让模型移动时受到保留任务几何结构约束。实验结果也表明，这种设计更有利于效用保持，但必须配合方向验证，否则可能出现“效用安全但遗忘方向错误”的情况。

### 3.5 基于 LiSSA/HVP 的 inverse-Hessian 方向近似

语言模型参数量很大，不能显式构造或求逆 Hessian。因此实现中使用 Hessian-vector product，也就是 HVP，并用轻量 LiSSA-style 递推近似 inverse-HVP。

**公式 (6)：近似 inverse-HVP 递推**

vₜ₊₁ = gⱼ + (1 - λ)vₜ - Href vₜ

变量说明：

- vₜ：当前递推向量。
- vₜ₊₁：下一步递推向量。
- Href vₜ：基于保留参考集 loss 计算的 Hessian-vector product。
- λ：LiSSA-style 递推中的阻尼系数。

本文实验使用递推深度 1、阻尼系数 0.01。这是保守设置，目的是验证完整链路、比较曲率来源和诊断方向问题，而不是做昂贵的二阶优化搜索。

因此，本文并不把 LiSSA/HVP 部分表述为精确 Hessian 逆，而是把它作为可运行的近似方向生成器。二阶近似只负责提出候选更新，是否应用该更新还要由后续 guard 判断。这样的设计更适合工程化联邦遗忘场景：即使近似方向存在误差，系统仍然可以通过 global guard 和 forget-loss guard 拒绝不合格更新。

### 3.6 遗忘更新缩放与全局范数裁剪

得到二阶方向后，先进行缩放。

**公式 (7)：原始遗忘更新**

Δθraw = ηv / (K - 1)

变量说明：

- η：遗忘更新步长，本文主要实验中为 0.01。
- v：公式 (5) 得到的 inverse-HVP 方向。
- K - 1：保留客户端数量。
- Δθraw：裁剪和 guard 检查之前的原始更新。

原始更新可能过大，因此需要进行全局 L2 范数裁剪。

**公式 (8)：全局范数裁剪**

Δθclip = Δθraw · min(1, τ / ||Δθraw||)

变量说明：

- ||Δθraw||：原始更新在所有可训练参数上的全局 L2 范数。
- τ：允许的最大更新范数。
- Δθclip：裁剪后的更新。

解释：如果原始更新本来就小于最大允许范数，则保持不变；如果超过最大允许范数，则按比例缩小。

范数裁剪还让不同实验设置之间更容易比较。不同 Hessian reference 可能产生尺度差异很大的更新向量，如果直接应用原始更新，很难判断结果来自方向本身还是来自步长过大。本文在表格中同时报告 max_norm、实际 update L2 和 guard steps，就是为了把更新幅度透明化，使实验结论不只依赖最终 loss 数值。

### 3.7 面向全局效用保持的分步信赖域 guard

裁剪后，更新不会一次性写入模型，而是拆成多个小步。

**公式 (9)：单步更新**

δₛ = Δθclip / S

变量说明：

- S：guard 拆分的小步数量。
- δₛ：每次尝试写入模型的小更新。

每一步都会先临时应用，再评估 global loss。

**公式 (10)：小步接受规则**

Lᴳ(θₜ + δₛ) ≤ γᴳ

变量说明：

- θₜ：当前 guard 小步之前的模型参数。
- δₛ：当前候选小步更新。
- Lᴳ(θₜ + δₛ)：临时应用候选小步后的全局 loss。
- γᴳ：全局 loss 接受边界。

解释：如果公式 (10) 成立，当前小步被接受；如果不成立，当前小步回滚，并停止遗忘过程。这就是本文称为 trust-region style update 的原因：模型只能在全局效用仍处于允许范围内时继续移动。

分步接受比一次性接受完整更新更适合本文场景。一次性更新只能给出“成功或失败”的结果，而分步 guard 可以保留已经安全接受的前缀更新，并在效用接近边界时停止。accepted steps 和 attempted steps 因而成为额外诊断指标：如果某个设置经常提前停止，说明该方向或该范数已经接近全局效用边界；如果所有小步都被接受，则说明该更新在当前 guard 下较为温和。

### 3.8 面向有效遗忘的方向探测与 forget-loss guard

阶段二结果显示，retained-set curvature 可以保持 global utility，但有时会降低 forget-client loss。这说明方向可能错配。阶段三因此同时探测正负两个方向。

**公式 (11)：方向候选**

θ⁺ = θ* + Δθclip

θ⁻ = θ* - Δθclip

变量说明：

- θ⁺：正向更新候选。
- θ⁻：负向更新候选。
- θ*：遗忘前模型参数。
- Δθclip：经过范数裁剪后的更新方向。

算法选择更能提高 forget-client loss、且尽量满足 global guard 的方向。

最后加入 forget-loss guard。

**公式 (12)：forget-loss guard**

Lⱼ(θₜ + δₛ) ≥ Lⱼᵍᵘᵃʳᵈ - ε

变量说明：

- θₜ：当前 guard 小步之前的模型参数。
- Lⱼᵍᵘᵃʳᵈ：guard 使用的基线 forget loss。
- ε：允许的数值容忍项，例如 0 或 0.005。

解释：一个小步不能只因为 global loss 安全就被接受，它还必须避免让 forget-client loss 往错误方向移动。该 guard 可以阻止“看起来保护了全局效用，但遗忘目标失败”的更新。

最终，本文的接受逻辑由 direction probing、global guard 和 forget-loss guard 共同决定。direction probing 先判断正负方向中哪一个更可能提高 forget-client loss；global guard 检查该方向是否会破坏全局效用；forget-loss guard 则检查该方向是否真的满足遗忘目标。实验表中记录 stop reason，是为了区分两类失败：global_loss_guard 表示更新过大或效用损失过高，forget_loss_guard 表示更新虽然可能保护效用，但不满足遗忘目标。

## 4 实验设计与评价指标

### 4.1 指令微调数据集与基础语言模型

实验使用 Dolly 指令微调数据集 [Dolly]。该数据集包含 instruction-response 形式的样本，适合构造语言模型指令微调场景。本文将 Dolly 样本划分到多个联邦客户端中，每个客户端拥有本地数据子集，服务器不能直接把所有客户端数据视为一个集中式训练集。这样的设置符合本文关注的联邦微调问题：训练和遗忘都发生在多客户端数据分布之上。

阶段一到阶段三使用 Qwen2-0.5B [Qwen2]。选择该模型是因为它足够体现语言模型微调特点，同时仍能在单张 RTX 5090 32GB GPU 上完成多轮训练、HVP 和多组消融实验。阶段四进一步加入 Qwen2.5-1.5B [Qwen2.5]，用于观察模型参数规模增加后，forget-batch Hessian 和 retained-set curvature 的行为是否发生变化。

### 4.2 联邦训练配置与遗忘更新参数

本文统一采用 20 个客户端，客户端采样比例为 0.2，本地步数为 2，batch size 为 1，最大序列长度为 64，遗忘客户端固定为 client 0。固定 forget client 的好处是可以让四个阶段的结果具有可比性，避免客户端差异成为主要干扰因素；不足之处是在局限性中讨论，即未来仍需要更多 forget-client 设置。

遗忘阶段中，forget-client gradient 使用目标客户端本地样本计算，`unlearn_grad_sample_size` 设置为 8。retained-set Hessian 实验中，`hessian_ref_per_client` 设置为 1，即每个保留客户端贡献 1 个参考样本。由于总客户端数为 20，遗忘 1 个客户端后保留客户端为 19 个，因此 retained reference set 的大小为 19。LiSSA/HVP 相关参数采用递推深度 1、阻尼系数 0.01，遗忘步长 `unlearn_eta` 为 0.01。

### 4.3 四阶段实验流程与鲁棒性检查设计

实验分为四个阶段和一个鲁棒性检查。阶段一是云端最小闭环验证，使用较小数据比例和较少训练轮次，主要确认 FL、FedHDS、forget-batch Hessian、retained-set Hessian、guard、summary 和图表生成都能完整运行。阶段二扩大到 data_sample 0.4 和 20 rounds，作为主要 trade-off 实验，用于比较 forget-batch Hessian 和 retained-set guard 的效用与遗忘差异。

阶段三在阶段二基础上运行完整 9 组方向与 guard 消融，重点分析 retained-set update 为什么会出现方向错配，以及 forget-loss guard 是否能拒绝无效遗忘更新。阶段四构建小模型/大模型和小数据/大数据的 2 x 2 规模矩阵，把阶段三作为小模型小数据格子，并补充 Stage 4A、Stage 4B、Stage 4C 三个设置。最后，在大模型大数据设置下增加 seed 43 和 seed 44 的核心 5 组实验，用于检验 seed 42 结果是否偶然。

### 4.4 对比方法、评价指标与判定标准

本文对比 FL、FedHDS、FedHDS + forget-batch Hessian、FedHDS + retained-set Hessian + global guard，以及 FedHDS + retained-set Hessian + negative sign + forget-loss guard。FL 和 FedHDS 用于比较训练底座的 global utility；forget-batch Hessian 是直接遗忘 baseline；retained-set Hessian 是本文关注的效用保持路径；negative sign 与 forget-loss guard 用于修正 retained-set update 的方向错配问题。

主要指标包括 final global loss、forget-client loss before/after、update L2 norm、accepted/attempted guard steps、stop reason 和 unlearning time。本文将 global loss 接近 FedHDS baseline 或低于 guard threshold 视为效用保持；将 forget-client loss 不下降或上升视为方向有效的遗忘信号。需要强调的是，forget-client loss 只是 loss-based unlearning proxy，因此本文在结论中采用谨慎表述，不把它夸大为严格隐私删除证明。

## 5 实验结果与机制分析

### 5.1 阶段一：云端最小闭环与实验链路可行性验证

阶段一验证 FL、FedHDS、forget-batch Hessian、retained-set Hessian 与 guard 是否可以在云端 GPU 上完整运行。表 1 汇总了该最小闭环设置下的训练和遗忘结果。

**表 1. 阶段一最小云端验证结果。**
""")
    parts.append(stage1_table)
    parts.append("""

阶段一说明代码链路可运行，但该阶段规模较小，不用于最终优劣判断。其价值在于确认训练、遗忘、summary、paper table 和 figures 可以形成闭环。

从数值上看，阶段一中 FL 的 final global loss 为 1.6443，FedHDS 为 1.6657，两者差异不大，而且该阶段数据规模较小，因此不能据此判断训练底座优劣。更关键的是两个遗忘分支都成功运行：forget-batch Hessian 将 forget loss 从 2.3381 提高到 2.3489，retained-set guard 将 forget loss 从 2.3381 提高到 2.3460。虽然遗忘增益都不大，但说明 HVP、参数更新、guard 检查和结果记录可以在真实模型上完成。

阶段一还暴露了一个后续需要关注的现象：retained-set 方法运行时间明显长于 forget-batch Hessian。这是因为 retained-set curvature 需要在保留参考集上计算 HVP，并进行分步 guard 检查。该成本在小规模阶段已经可见，因此后续章节不会把 retained-set 方法描述为最快的遗忘方法，而是强调它在效用保持和可审计更新方面的优势。

### 5.2 阶段二：forget-batch 与 retained-set 的效用-遗忘 trade-off

阶段二将 data_sample 提高到 0.4、rounds 提高到 20，用于观察更正式设置下的效用和遗忘 trade-off。表 2 给出了阶段二的主要结果。

**表 2. 阶段二主要 trade-off 结果。**
""")
    parts.append(stage2_table)
    parts.append("""

阶段二的主要结论是：FedHDS 在更大设置下成为更好的 utility baseline；forget-batch Hessian 能提高 forget loss，但 global loss 代价较大；retained-set guard 能显著保护 global utility，但原方向可能降低 forget-client loss。这一问题直接引出阶段三的方向与 guard 消融。

更具体地说，阶段二中 FedHDS 的 final global loss 为 1.5250，低于 FL 的 1.5441，说明在 data_sample 0.4 和 20 rounds 设置下，FedHDS-style 过滤训练底座确实形成了更好的效用起点。forget-batch Hessian 把 forget loss 从 2.2581 提高到 2.2866，说明直接使用遗忘客户端 batch 作为 Hessian reference 时，更新方向更容易服务遗忘目标。但它的 final global loss 达到 1.6257，相比 FedHDS baseline 恶化明显，说明该方向对保留任务不够温和。

retained-set guard 的表现则相反。它的 final global loss 为 1.5482，明显接近 FedHDS baseline，也低于自动 guard 阈值 1.575，说明 retained reference 对效用保持有帮助。但它把 forget loss 从 2.2581 降到 2.2329，这意味着模型反而更适应遗忘客户端。这个现象非常关键：如果只看 global loss，会误以为 retained-set 方法更好；但结合 forget-client loss 后可以看到，它在原始方向上没有完成遗忘目标。因此阶段三必须进一步检查方向和 forget-loss guard。

### 5.3 阶段三：retained-set 更新方向错配与 guard 消融分析

阶段三保留阶段二基础设置，运行完整 9 组方向与 guard 消融，用于解释 retained-set update 的方向错配。完整消融结果如表 3 所示。

**表 3. 阶段三完整 9 组方向与 guard 消融。**
""")
    parts.append(table_md(zh_stage3_main_table(rows)))
    parts.append("""

阶段三表明，原始 positive/auto retained-set 方向虽然可以保持 global loss，但会降低 forget-client loss。启用 forget-loss guard 后，该无效小步会被拒绝，模型保持在 FedHDS baseline 状态。这说明 guard 并不是装饰性机制，而是真正阻止了“全局效用看似安全但遗忘目标失败”的更新。表 4 进一步聚焦 negative-sign update norm sweep。

从表 3 可以看出，retained_auto_direction_guard05 接受了 5/5 个小步，final global loss 为 1.5482，说明全局效用 guard 没有发现问题；但是 forget loss 从 2.2581 降到 2.2329，说明该方向对遗忘目标是错误的。加入 forget-loss guard 后，两个 tolerance 设置都在 0/1 步被拒绝，模型保持在 FedHDS baseline。这个结果说明 forget-loss guard 的价值不在于提高数值，而在于拒绝不应接受的更新。

forget-loss guard 的拒绝行为也使本文方法更容易审计。对于一个遗忘请求，系统不仅给出最终模型，还记录了为什么某个更新没有被接受。如果 stop reason 是 forget_loss_guard，说明问题出在遗忘目标方向；如果 stop reason 是 global_loss_guard，说明问题出在效用损失。这种区分对于联邦遗忘场景很重要，因为实际系统需要知道失败来自“忘不掉”还是“伤害太大”。

**表 4. 阶段三 negative-sign update norm sweep。**
""")
    parts.append(table_md(zh_negative_sweep_table(rows)))
    parts.append("""

negative-sign sweep 是阶段三的关键证据。max_norm=0.05 时，forget loss 从 2.2581 提高到 2.4105，global loss 为 1.5561；max_norm=0.10 时，forget loss 提高到 2.6104，global loss 为 1.5935；max_norm=0.15 时，guard 在 4/5 步停止，最终 update L2 为 0.1200。这说明 retained-set curvature 对方向敏感，必须结合 direction probing 和 forget-loss guard。

negative-sign sweep 给出了阶段三最重要的修正证据。max_norm=0.05 时，forget loss 提高 0.1524，global loss 从 1.5250 增加到 1.5561，仍处于较温和范围；max_norm=0.10 时，forget loss 提高 0.3523，global loss 增加到 1.5935，遗忘更强但效用代价也更高；max_norm=0.15 时，forget loss 提高最多，但 global guard 在 4/5 步截断更新，实际 update L2 只有 0.1200。这个结果说明，遗忘强度和全局效用之间存在连续 trade-off，而不是某个固定设置天然最优。

因此，阶段三的结论不是简单地选择最大 max_norm，而是说明 retained-set curvature 必须配合方向探测和受控步长。对于偏保守应用，可以选择 max_norm=0.05，以获得较小效用损失；对于需要更强遗忘信号的场景，可以考虑 max_norm=0.10；max_norm=0.15 更适合作为 guard 截断能力的证据，而不是默认推荐配置。

### 5.4 阶段四：模型规模与数据规模变化下的泛化验证

阶段四不引入新方法，而是用 2 x 2 矩阵验证阶段三得到的 corrected retained-negative update 是否在更大模型或更大数据下仍有解释性。阶段三作为小模型小数据格子，Stage 4A、4B、4C 补齐其余三个格子。表 5 汇总了 2 x 2 规模矩阵。

**表 5. 模型/数据规模泛化矩阵。**
""")
    parts.append(table_md(scale))
    for item in stage4_sets:
        parts.append(f"""

#### {item['tag']}：{item['title']} 设置下的完整结果

**表 {6 + ["Stage 4A", "Stage 4B", "Stage 4C"].index(str(item['tag']))}. {item['tag']} 完整结果。**
""")
        parts.append(table_md(zh_stage4_key_table(item["rows"])))  # type: ignore[arg-type]
    parts.append("""

表 6 至表 8 给出了 Stage 4A、Stage 4B 和 Stage 4C 的完整结果。规模矩阵支持三个观察。第一，小模型大数据设置中，forget-batch Hessian 遗忘更强，但 global loss 明显恶化；retained negative 也能提高 forget loss，同时 global loss 更接近 FedHDS。第二，大模型小数据设置中，forget-batch Hessian 不再提高 forget loss，而 retained negative 仍有小幅正向遗忘增益。第三，大模型大数据设置中，retained negative 的遗忘增益较小，但 final global loss 低于 FedHDS baseline，说明其效用行为更有利。

进一步分析 Stage 4A 可以看到，数据规模增加后 forget-client loss 的初始值从阶段三的 2.2581 提高到 2.4377，说明更大数据采样改变了目标客户端的局部 loss 状态。forget-batch Hessian 在该设置下遗忘增益很强，从 2.4377 提高到 2.8059，但 final global loss 上升到 2.0040，效用代价过大。retained negative norm010 的遗忘增益为 0.1770，弱于 forget-batch Hessian，但 global loss 为 1.5701，更接近 FedHDS baseline。这体现了本文方法的主要定位：不追求最强单点遗忘增益，而强调受控 trade-off。

Stage 4B 和 Stage 4C 则说明，大模型设置下 forget-batch Hessian 并不总是可靠。Stage 4B 中 forget-batch Hessian 使 forget loss 从 1.8812 降到 1.8702，Stage 4C 中也从 2.0880 降到 2.0828，方向都不是理想遗忘方向。相反，retained negative 设置在两个大模型格子中都给出正向或非负的遗忘变化，并且 global loss 接近或低于 FedHDS baseline。该结果说明 retained-set negative update 在模型规模增加后仍具有一定解释性。

### 5.5 大模型大数据设置下的随机种子鲁棒性检查

seed robustness check 不是新阶段，而是对 Stage 4C 的核心设置补充 seed 43 和 seed 44，用于判断 seed 42 的结果是否偶然。表 9 汇总了三个 seed 的核心对比。

**表 9. 大模型大数据设置下的 seed 鲁棒性检查。**
""")
    if stage4c_rows and seed_rows:
        parts.append(table_md(zh_seed_key_table(stage4c_rows, seed_rows)))
    parts.append("""

seed robustness 结果给出更谨慎的结论。seed 42 中 retained negative 提高 forget loss 且 global loss 低于 FedHDS；seed 43 中 global utility 与 FedHDS 接近，但 forget loss 出现极小下降；seed 44 中 forget-loss guard 直接拒绝 negative update，使模型保持在 FedHDS 状态。因此，本文不能声称 retained negative 在所有 seed 下都稳定提高 forget loss。更准确的结论是：该框架在 utility control 和 guard behavior 上更稳定，而 exact forgetting gain 对随机种子敏感。

从三个 seed 的对比看，稳定的是 guard 行为和效用控制，不稳定的是 forget-loss gain 的具体数值。seed 42 给出正向遗忘增益，seed 43 出现极小负变化，seed 44 则由 forget-loss guard 拒绝更新。这个结果反而强化了本文对 guard 的需求：如果没有 forget-loss guard，seed 44 中可能会接受一个不满足遗忘目标的更新；加入 guard 后，系统至少可以保持在 FedHDS 状态，避免错误更新进一步改变模型。

### 5.6 跨阶段结果讨论与方法定位

综合四个阶段，本文得到一条清晰主线：forget-batch Hessian 可以服务遗忘目标，但容易损害全局效用；retained-set curvature 更适合 utility preservation，但存在方向错配风险；direction probing、negative-sign correction 和 forget-loss guard 可以修正该风险；2 x 2 规模矩阵说明该 trade-off 在当前测试范围内仍有解释性；seed robustness 则提醒我们，遗忘增益不能过度外推。

跨阶段结果说明，本文方法最适合被理解为一种可审计的受控更新框架，而不是无条件最强的遗忘算法。forget-batch Hessian 在某些小模型或大数据设置下可以产生更大 forget-loss gain，但 global loss 代价也更明显；retained-set curvature 则更强调保留效用，但必须通过方向探测和 forget-loss guard 保证它没有沿错误方向移动。

因此，本文最终主张是谨慎的：retained-set curvature + negative direction + guard 可以在当前 Dolly/Qwen 实验范围内形成较稳定的 utility-preserving unlearning path。它的贡献在于把遗忘更新从一次性参数跳变变成可度量、可回滚、可解释的过程。每次更新都有 update norm、accepted steps、stop reason、global loss 和 forget loss 作为记录，这对联邦遗忘系统的实际部署和审计都更有价值。

## 6 方法局限性与适用边界

本文仍存在若干局限。

第一，实验数据集和模型家族仍然有限。本文主要使用 Dolly 指令微调数据集和 Qwen/Qwen2.5 模型家族。虽然阶段四已经加入模型规模和数据规模矩阵，但这仍属于同一数据集和同一模型家族下的受控泛化实验。不同指令数据集、不同模型架构、不同 tokenizer 和不同客户端分布都可能改变 Hessian 近似和 guard 行为。因此，本文结论应解释为当前 Dolly/Qwen 设置下的实验证据，而不是跨所有语言模型场景的普适结论。

第二，遗忘客户端设置仍然较单一。本文固定 forget client 为 client 0，这有利于让不同阶段之间保持可比性，但也限制了结论范围。不同客户端可能具有不同数据量、主题分布、样本难度和局部 loss 状态，遗忘难度也可能不同。未来工作需要在多个 forget client 上重复实验，并进一步分析客户端异质性对 retained-set curvature 的影响。

第三，seed robustness 检查仍然不够充分。本文补充了大模型大数据核心设置下的 seed 43 和 seed 44，用于判断 seed 42 是否偶然。结果显示 utility control 和 guard behavior 相对稳定，但 exact forgetting gain 对 seed 敏感。由于额外 seed 数量仍然有限，本文不能声称遗忘增益具有强统计稳定性。更严格的结论需要更多随机种子、更完整的九组实验和统计显著性分析。

第四，retained-set Hessian 的计算成本高于 forget-batch Hessian。forget-batch Hessian 只需要在目标客户端 batch 上计算曲率，而 retained-set curvature 需要构造保留参考集，并在保留样本上计算 HVP，还要执行分步 global guard 和 forget-loss guard。实验中 retained-set 方法的 unlearning time 明显更长。未来需要研究更高效的参考集选择、HVP 近似、缓存机制或低秩近似方法。

第五，本文遗忘评估主要基于 loss 指标。forget-client loss 是直接、可复现且便于跨阶段比较的代理指标，但它不能完全等价于隐私层面的删除证明。一个模型在目标客户端 loss 上升，并不必然意味着无法通过 membership inference、canary extraction 或其他攻击恢复目标数据影响。因此，后续工作应加入隐私攻击指标、生成质量指标和更细粒度的目标样本分析。

第六，本文的 guard 阈值和 max_norm 仍需人工设定。不同模型、不同数据采样比例和不同训练轮次下，global loss 的绝对值可能不同，固定阈值不一定适用。本文通过 auto guard 和 norm sweep 给出了一种可操作方案，但更理想的做法是根据训练过程、验证集波动和目标风险自动校准 guard threshold。

## 7 结论与未来工作

本文研究了基于 FedHDS-style 训练底座、保留集曲率和受控信赖域更新的联邦遗忘方法。论文围绕“如何在削弱目标客户端影响的同时保持保留客户端效用”这一问题展开，将遗忘客户端梯度和保留客户端曲率参考解耦，并通过 norm clipping、direction probing、stepped global guard 和 forget-loss guard 控制二阶更新。

实验结果可以总结为五点。第一，完整训练-遗忘-汇总链路可以在云端 GPU 上运行，说明本文实现的实验系统具备可复现的闭环能力。第二，阶段二结果显示 forget-batch Hessian 可以产生更直接的遗忘信号，但会带来明显 global loss 代价。第三，retained-set Hessian 更有利于 global utility preservation，但原始方向可能降低 forget-client loss，因此必须进行方向验证。第四，阶段三证明 negative-sign correction 与 forget-loss guard 能够修正方向错配，在 global loss 受控条件下提高 forget-client loss。第五，阶段四规模矩阵和 seed robustness check 表明，该方法在当前 Dolly/Qwen 设置下具有可解释的效用保持行为，但 exact forgetting gain 对模型规模、数据规模和随机种子敏感。

总体而言，本文方法不应被理解为所有设置下遗忘强度最强的算法，而应被定位为一种可审计、utility-preserving 的受控联邦遗忘框架。它的核心价值在于：每次遗忘更新都有明确的信号来源、曲率参考、更新范数、guard 接受步数和停止原因。相比直接应用二阶更新，这种受控流程更适合需要解释和审计的联邦学习场景。

未来工作可以从四个方向继续推进。第一，扩展到更多数据集、更多模型家族和更多 forget client，验证方法的泛化性。第二，引入 membership inference、canary extraction 等隐私攻击指标，使遗忘评估从 loss proxy 走向更强的隐私验证。第三，改进 retained-set HVP 的效率，降低二阶近似在大模型上的计算成本。第四，研究自动 guard calibration，使 global guard 和 forget-loss guard 能根据不同训练状态自适应设定阈值。
""")
    parts.append(appendix_figure_caption_list(zh=True))
    parts.append(ZH_REFERENCES)
    text = "\n".join(part for part in parts if part.strip()) + "\n"
    text = normalize_zh_figures(text)
    text = numeric_citations(text)
    text = polish_chinese(text)
    return replace_zh_references(text)


def main() -> None:
    paper = ROOT / "paper_draft.md"
    text = paper.read_text(encoding="utf-8")
    text = polish_english(text)
    paper.write_text(text, encoding="utf-8")
    (ROOT / "paper_draft_final.md").write_text(text, encoding="utf-8")
    markdown_to_docx(paper, ROOT / "paper_draft.docx")
    markdown_to_docx(ROOT / "paper_draft_final.md", ROOT / "paper_draft_final.docx")

    zh = make_chinese_paper(text)
    zh_path = ROOT / "paper_draft_zh.md"
    zh_path.write_text(zh, encoding="utf-8")
    try:
        markdown_to_docx(zh_path, ROOT / "paper_draft_zh.docx")
    except PermissionError:
        markdown_to_docx(zh_path, ROOT / "paper_draft_zh_updated.docx")


if __name__ == "__main__":
    main()
