from __future__ import annotations

import csv
import html
import os
import re
import shutil
import struct
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT / "formal_cloud_results" / "stage3_full_direction_guard_20260504"
FIGURES = STAGE3 / "figures"
STAGE4A = ROOT / "formal_cloud_results" / "stage4a_qwen05b_dsample06_full9_20260504"
STAGE4B = ROOT / "formal_cloud_results" / "stage4b_qwen15b_dsample04_full9_20260504"
STAGE4C = ROOT / "formal_cloud_results" / "stage4_large_model_data_20260504"
STAGE4_SEED = ROOT / "formal_cloud_results" / "stage4_seed_robustness_qwen15b_dsample06_core5_20260504"

STAGE4_CONFIGS = [
    ("Stage 4A", "Small model + large data", "Qwen2-0.5B", "0.6", STAGE4A),
    ("Stage 4B", "Large model + small data", "Qwen2.5-1.5B", "0.4", STAGE4B),
    ("Stage 4C", "Large model + large data", "Qwen2.5-1.5B", "0.6", STAGE4C),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f4(value: str | float | None, empty: str = "-") -> str:
    if value is None or value == "":
        return empty
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def f2(value: str | float | None, empty: str = "-") -> str:
    if value is None or value == "":
        return empty
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def gain(before: str, after: str) -> str:
    return f"{float(after) - float(before):+.4f}"


def by_group(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["group"]: row for row in rows}


def result_tables(rows: list[dict[str, str]]) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    g = by_group(rows)
    full_order = [
        "fl",
        "fedhds",
        "forget_batch_hessian",
        "retained_auto_direction_guard05",
        "retained_auto_forget_guard_tol0_guard05",
        "retained_auto_forget_guard_tol005_guard10",
        "retained_negative_norm005_guard05",
        "retained_negative_norm010_guard10",
        "retained_negative_norm015_guard10",
    ]
    full = [["组别", "Hessian/方向", "Forget loss", "Final global loss", "Update L2", "步数", "停止原因", "时间(s)"]]
    for name in full_order:
        row = g[name]
        if name in ("fl", "fedhds"):
            direction = "training baseline"
            forget = "-"
            steps = "-"
            stop = "-"
            time = "-"
        else:
            sign = row.get("unlearn_selected_update_sign") or row.get("unlearn_update_sign_mode") or "-"
            direction = f"{row.get('hessian_mode', '-')}, sign={sign}"
            forget = f"{f4(row['forget_client_loss_before_unlearning'])} -> {f4(row['forget_client_loss_after_unlearning'])} ({gain(row['forget_client_loss_before_unlearning'], row['forget_client_loss_after_unlearning'])})"
            steps = f"{int(float(row['unlearn_steps_accepted']))}/{int(float(row['unlearn_steps_attempted']))}"
            stop = row.get("unlearn_guard_stop_reason") or "-"
            time = f2(row.get("unlearning_time_sec"))
        full.append([
            name,
            direction,
            forget,
            f4(row.get("global_loss_after_unlearning") or row.get("final_global_loss")),
            f4(row.get("actual_param_delta_l2_norm")),
            steps,
            stop,
            time,
        ])

    direction = [["组别", "选择方向", "Forget loss", "Global loss", "步数", "停止原因", "说明"]]
    for name in [
        "retained_auto_direction_guard05",
        "retained_auto_forget_guard_tol0_guard05",
        "retained_auto_forget_guard_tol005_guard10",
    ]:
        row = g[name]
        direction.append([
            name,
            row.get("unlearn_selected_update_sign", "-"),
            f"{f4(row['forget_client_loss_before_unlearning'])} -> {f4(row['forget_client_loss_after_unlearning'])} ({gain(row['forget_client_loss_before_unlearning'], row['forget_client_loss_after_unlearning'])})",
            f"{f4(row['global_loss_before_unlearning'])} -> {f4(row['global_loss_after_unlearning'])}",
            f"{int(float(row['unlearn_steps_accepted']))}/{int(float(row['unlearn_steps_attempted']))}",
            row.get("unlearn_guard_stop_reason") or "-",
            "positive/auto 方向使 forget loss 下降" if name.endswith("guard05") and "forget_guard" not in name else "forget-loss guard 拒绝无效小步",
        ])

    neg = [["组别", "max_norm", "Forget loss", "Global loss", "Update L2", "步数", "停止原因"]]
    for name in [
        "retained_negative_norm005_guard05",
        "retained_negative_norm010_guard10",
        "retained_negative_norm015_guard10",
    ]:
        row = g[name]
        neg.append([
            name,
            f4(row["unlearn_max_update_norm"]),
            f"{f4(row['forget_client_loss_before_unlearning'])} -> {f4(row['forget_client_loss_after_unlearning'])} ({gain(row['forget_client_loss_before_unlearning'], row['forget_client_loss_after_unlearning'])})",
            f"{f4(row['global_loss_before_unlearning'])} -> {f4(row['global_loss_after_unlearning'])}",
            f4(row["actual_param_delta_l2_norm"]),
            f"{int(float(row['unlearn_steps_accepted']))}/{int(float(row['unlearn_steps_attempted']))}",
            row.get("unlearn_guard_stop_reason") or "-",
        ])
    return full, direction, neg


def table_md(table: list[list[str]]) -> str:
    header = "| " + " | ".join(table[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
    rows = ["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in table[1:]]
    return "\n".join([header, sep, *rows])


def final_loss(row: dict[str, str]) -> str:
    return row.get("global_loss_after_unlearning") or row.get("final_global_loss") or ""


def forget_text(row: dict[str, str]) -> str:
    before = row.get("forget_client_loss_before_unlearning", "")
    after = row.get("forget_client_loss_after_unlearning", "")
    if not before or not after:
        return "-"
    return f"{f4(before)} -> {f4(after)} ({gain(before, after)})"


def steps_text(row: dict[str, str]) -> str:
    accepted = row.get("unlearn_steps_accepted", "")
    attempted = row.get("unlearn_steps_attempted", "")
    if not accepted or not attempted:
        return "-"
    return f"{int(float(accepted))}/{int(float(attempted))}"


def note_text(row: dict[str, str]) -> str:
    hessian = row.get("hessian_mode", "")
    if not hessian:
        return "utility baseline"
    stop = row.get("unlearn_guard_stop_reason", "")
    if stop and stop != "completed":
        return f"guard stopped: {stop}"
    best_sign = row.get("unlearn_best_probe_update_sign", "")
    best_delta = row.get("unlearn_best_probe_direction_forget_delta", "")
    if best_sign:
        return f"best probe sign {best_sign}, delta {f4(best_delta)}"
    before = row.get("forget_client_loss_before_unlearning", "")
    after = row.get("forget_client_loss_after_unlearning", "")
    if before and after and float(after) > float(before):
        return "forget loss increased"
    return ""


def load_stage4_sets() -> list[dict[str, object]]:
    sets: list[dict[str, object]] = []
    for tag, title, model, data_sample, folder in STAGE4_CONFIGS:
        summary = folder / "summary.csv"
        if summary.exists():
            sets.append({
                "tag": tag,
                "title": title,
                "model": model,
                "data_sample": data_sample,
                "folder": folder,
                "rows": read_csv_rows(summary),
            })
    return sets


def load_seed_rows() -> list[dict[str, str]]:
    summary = STAGE4_SEED / "summary.csv"
    if not summary.exists():
        return []
    return read_csv_rows(summary)


def rows_for_seed(seed_rows: list[dict[str, str]], seed: str) -> list[dict[str, str]]:
    prefix = f"seed{seed}/"
    rows: list[dict[str, str]] = []
    for row in seed_rows:
        group = row.get("group", "")
        if group.startswith(prefix):
            copied = dict(row)
            copied["group"] = group[len(prefix):]
            rows.append(copied)
    return rows


def seed_robustness_table(stage4c_rows: list[dict[str, str]], seed_rows: list[dict[str, str]]) -> list[list[str]]:
    table = [[
        "Seed",
        "FedHDS global",
        "Forget-batch Hessian",
        "Retained auto guard",
        "Retained negative norm010",
        "Interpretation",
    ]]
    seed_sources = [
        ("42", stage4c_rows),
        ("43", rows_for_seed(seed_rows, "43")),
        ("44", rows_for_seed(seed_rows, "44")),
    ]
    interpretations = {
        "42": "retained negative improves forget loss and keeps global loss below FedHDS",
        "43": "utility remains close to FedHDS, but retained forgetting gain is seed-sensitive",
        "44": "forget-loss guard rejects the negative update and preserves the FedHDS state",
    }
    for seed, rows in seed_sources:
        if not rows:
            continue
        g = by_group(rows)
        fedhds = g.get("fedhds")
        fb = g.get("forget_batch_hessian")
        auto = g.get("retained_auto_forget_guard_tol0_guard10")
        neg = g.get("retained_negative_norm010_guard10")
        table.append([
            seed,
            f4(fedhds.get("final_global_loss") if fedhds else ""),
            f"{forget_text(fb)}, G={f4(final_loss(fb))}" if fb else "-",
            f"{forget_text(auto)}, G={f4(final_loss(auto))}, steps={steps_text(auto)}" if auto else "-",
            f"{forget_text(neg)}, G={f4(final_loss(neg))}, steps={steps_text(neg)}" if neg else "-",
            interpretations.get(seed, ""),
        ])
    return table


def seed_robustness_text(stage4c_rows: list[dict[str, str]], seed_rows: list[dict[str, str]]) -> str:
    if not seed_rows:
        return ""
    table = seed_robustness_table(stage4c_rows, seed_rows)
    return f"""阶段四稳定性补充：大模型大数据 seed robustness

本补充实验不是新的阶段五主实验，而是阶段四C的大模型大数据核心配置复现。它保持 Qwen2.5-1.5B、data_sample=0.6、rounds=20、20 clients、client fraction=0.2、forget_client_idx=0 等设置不变，只把 seed 从 42 扩展到 43 和 44，并只运行核心 5 组：

```text
fl
fedhds
forget_batch_hessian
retained_auto_forget_guard_tol0_guard10
retained_negative_norm010_guard10
```

{table_md(table)}

稳定性补充的主要结论：

1. seed=42 中，retained_negative_norm010_guard10 将 forget loss 从 2.0880 提高到 2.1078，并且 final global loss 1.6132 低于 FedHDS baseline 1.6201。
2. seed=43 中，retained_negative_norm010_guard10 的 global loss 1.6380 与 FedHDS baseline 1.6377 基本一致，但 forget loss 从 1.8483 到 1.8472，出现极小下降。这说明 forgetting gain 对 seed 敏感。
3. seed=44 中，retained_negative_norm010_guard10 被 forget-loss guard 在 0/1 步拒绝，模型保持在 FedHDS baseline：forget loss 1.8732 -> 1.8732，global loss 2.0818。这说明 guard 能阻止无效遗忘更新，但也意味着并非所有 seed 都会产生正向 forgetting gain。
4. 因此，seed robustness 不应写成“retained negative 在所有 seed 上都稳定提高 forget loss”。更准确的结论是：retained-set negative + guard 在大模型大数据设置下通常能保持 global utility 安全，并且 guard 能阻止错误方向更新；具体 forgetting gain 存在随机种子敏感性。

论文写法建议：

```text
The seed robustness check shows that the retained-set negative update remains utility-safe under the large-model/large-data setting, but its exact forgetting gain is seed-sensitive. In seed 42 the retained-negative update improves forget loss while reducing global loss below FedHDS. In seed 43 the update preserves global utility but does not improve forgetting. In seed 44 the forget-loss guard rejects the negative update and keeps the model at the FedHDS state. These results support the role of guard mechanisms and clarify the boundary of the forgetting claim.
```
"""


def paper_seed_section(stage4c_rows: list[dict[str, str]], seed_rows: list[dict[str, str]]) -> str:
    if not seed_rows:
        return ""
    table = seed_robustness_table(stage4c_rows, seed_rows)
    return f"""### 5.5 Seed Robustness Check on the Large-Model/Large-Data Setting

After the 2 x 2 scale matrix, we run a small robustness check on the most demanding Stage-4C setting: Qwen2.5-1.5B with data_sample 0.6 and 20 rounds. The goal is not to create a new experimental stage, but to test whether the seed=42 observation is an isolated accident. We keep the federated and unlearning configuration fixed and rerun only the core five settings for seed 43 and seed 44.

Table 6 compares seed 42, seed 43, and seed 44. Seed 42 is the original Stage-4C result, while seed 43 and seed 44 come from the robustness run.

{table_md(table)}

The robustness check gives a more nuanced picture than a simple success/failure statement. In seed 42, retained negative norm010 increases forget loss from 2.0880 to 2.1078 and obtains final global loss 1.6132, below the FedHDS baseline 1.6201. In seed 43, the same retained negative setting preserves utility, with final global loss 1.6380 close to the FedHDS baseline 1.6377, but the forget loss changes from 1.8483 to 1.8472, a very small decrease. In seed 44, the forget-loss guard rejects the retained negative update at 0/1 accepted steps, leaving the model at the FedHDS state with forget loss 1.8732 and global loss 2.0818.

These results support two claims and one boundary. The first supported claim is that retained-set negative updates remain utility-safe in the large-model/large-data setting: even when forgetting gain is absent, global loss stays close to the FedHDS baseline. The second supported claim is that the forget-loss guard is functionally important: in seed 44 it prevents an invalid update from being accepted. The boundary is that the exact forgetting gain is seed-sensitive. Therefore, the paper should not claim that retained negative unlearning always increases forget loss. A more accurate conclusion is that retained-set Hessian with negative direction and guard constraints provides an auditable and utility-safe update path, while the strength of forgetting depends on the local training trajectory and random seed.
"""


def compact_result_table(rows: list[dict[str, str]]) -> list[list[str]]:
    g = by_group(rows)
    order = [
        "fl",
        "fedhds",
        "forget_batch_hessian",
        "retained_auto_direction_guard05",
        "retained_auto_forget_guard_tol0_guard05",
        "retained_auto_forget_guard_tol0_guard10",
        "retained_auto_forget_guard_tol005_guard10",
        "retained_negative_norm005_guard05",
        "retained_negative_norm010_guard10",
        "retained_negative_norm015_guard10",
    ]
    table = [["Group", "Hessian/sign", "Forget loss", "Final global loss", "Update L2", "Steps", "Note"]]
    for name in order:
        if name not in g:
            continue
        row = g[name]
        if name in ("fl", "fedhds"):
            mode = "training baseline"
        else:
            sign = row.get("unlearn_selected_update_sign") or row.get("unlearn_update_sign_mode") or "-"
            mode = f"{row.get('hessian_mode', '-')}, sign={sign}"
        table.append([
            name,
            mode,
            forget_text(row),
            f4(final_loss(row)),
            f4(row.get("actual_param_delta_l2_norm")),
            steps_text(row),
            note_text(row),
        ])
    return table


def scale_matrix_table(stage3_rows: list[dict[str, str]], stage4_sets: list[dict[str, object]]) -> list[list[str]]:
    all_sets: list[dict[str, object]] = [{
        "tag": "Stage 3",
        "title": "Small model + small data",
        "model": "Qwen2-0.5B",
        "data_sample": "0.4",
        "rows": stage3_rows,
    }]
    all_sets.extend(stage4_sets)
    readings = {
        "Stage 3": "mechanism diagnosis and corrected negative direction",
        "Stage 4A": "data scaling keeps retained negative effective; forget-batch is stronger but costly",
        "Stage 4B": "model scaling weakens forget-batch; retained negative gives modest forgetting gain",
        "Stage 4C": "large model + large data preserves the retained-negative trade-off",
    }
    table = [[
        "Setting",
        "Model",
        "Data sample",
        "FedHDS global",
        "Forget-batch Hessian",
        "Retained negative norm010",
        "Main reading",
    ]]
    for item in all_sets:
        rows = item["rows"]  # type: ignore[index]
        g = by_group(rows)  # type: ignore[arg-type]
        fb = g["forget_batch_hessian"]
        rn = g["retained_negative_norm010_guard10"]
        tag = str(item["tag"])
        table.append([
            f"{tag}: {item['title']}",
            str(item["model"]),
            str(item["data_sample"]),
            f4(g["fedhds"]["final_global_loss"]),
            f"{forget_text(fb)}, G={f4(final_loss(fb))}",
            f"{forget_text(rn)}, G={f4(final_loss(rn))}",
            readings.get(tag, ""),
        ])
    return table


def stage4_text(stage3_rows: list[dict[str, str]], stage4_sets: list[dict[str, object]]) -> str:
    matrix = scale_matrix_table(stage3_rows, stage4_sets)
    sections = []
    for item in stage4_sets:
        rows = item["rows"]  # type: ignore[index]
        g = by_group(rows)  # type: ignore[arg-type]
        fb = g["forget_batch_hessian"]
        rn = g["retained_negative_norm010_guard10"]
        result_table = table_md(compact_result_table(rows))  # type: ignore[arg-type]
        sections.append(f"""### {item['tag']}: {item['title']}

Result directory:

```text
{item['folder']}
```

{result_table}

Key reading: FedHDS final global loss is {f4(g['fedhds']['final_global_loss'])}. The forget-batch Hessian changes forget loss {forget_text(fb)} with final global loss {f4(final_loss(fb))}. The retained negative norm010 setting changes forget loss {forget_text(rn)} with final global loss {f4(final_loss(rn))}. This setting is therefore used as the comparable retained-set negative configuration in the 2 x 2 scale matrix.
""")
    return f"""阶段四：模型规模和数据规模泛化验证

阶段四不再引入新的方法组件，而是验证阶段三得到的方向修正和 guard 机制是否能在更大模型或更大数据下保持可解释的 forgetting / utility trade-off。阶段四采用 2 x 2 矩阵设计：小模型/大模型与小数据/大数据交叉对照。

{table_md(matrix)}

阶段四的主要发现：

1. 小模型 + 大数据的 Stage 4A 中，forget-batch Hessian 可以大幅提高 forget loss，但 final global loss 明显恶化；retained-set negative update 同样提高 forget loss，同时 global loss 更接近 FedHDS。
2. 大模型 + 小数据的 Stage 4B 中，forget-batch Hessian 没有提高 forget loss，反而略微降低；retained-set negative update 小幅提高 forget loss，并保持 global loss 接近或略优于 FedHDS。
3. 大模型 + 大数据的 Stage 4C 中，retained-set negative update 继续提高 forget loss，并且 final global loss 低于 FedHDS baseline；forget-batch Hessian 仍没有提供稳定 forgetting gain。
4. 因此，阶段四支持更谨慎的结论：retained-set Hessian + negative direction + guard 不一定总是取得最大 forgetting gain，但比 naive forget-batch Hessian 更容易形成可控的 utility-preserving trade-off。

{chr(10).join(sections)}
"""


def paper_stage4_section(stage3_rows: list[dict[str, str]], stage4_sets: list[dict[str, object]]) -> str:
    matrix = scale_matrix_table(stage3_rows, stage4_sets)
    detail_tables = []
    for item in stage4_sets:
        rows = item["rows"]  # type: ignore[index]
        result_table = table_md(compact_result_table(rows))  # type: ignore[arg-type]
        detail_tables.append(f"""#### {item['tag']}: {item['title']}

{result_table}
""")
    return f"""### 5.4 Stage 4: Model and Data Scale Generalization

Stage 4 evaluates whether the Stage-3 direction correction and guard design remain meaningful when the data scale, model scale, or both are increased. It does not introduce a new algorithmic component. Instead, it forms a 2 x 2 scale matrix: Qwen2-0.5B versus Qwen2.5-1.5B, and data_sample 0.4 versus 0.6. Stage 3 provides the small-model/small-data cell, while Stage 4A, Stage 4B, and Stage 4C fill the remaining cells.

Table 5 summarizes the scale matrix using three comparable entries from each cell: the FedHDS utility baseline, the forget-batch Hessian baseline, and the retained-set negative norm010 configuration.

{table_md(matrix)}

The scale matrix supports three observations. First, increasing data while keeping the small model in Stage 4A preserves the main retained-set negative behavior: forget loss increases from 2.4377 to 2.6147 under norm010, while final global loss remains 1.5701, close to the FedHDS baseline 1.5423. In the same setting, forget-batch Hessian gives a stronger forgetting gain, 2.4377 to 2.8059, but its final global loss rises sharply to 2.0040. This is a clear utility-cost example.

Second, increasing model size while keeping data_sample 0.4 in Stage 4B changes the forget-batch behavior. Forget-batch Hessian no longer increases forget loss; it changes from 1.8812 to 1.8702. By contrast, retained negative norm010 increases forget loss from 1.8812 to 1.9206, with final global loss 1.6405, slightly lower than the FedHDS baseline 1.6438. This suggests that the retained-set negative direction remains useful even when the direct forget-batch curvature becomes weak or misaligned.

Third, increasing both model and data scale in Stage 4C keeps the retained-set negative update stable. Forget-batch Hessian changes forget loss from 2.0880 to 2.0828 and final global loss is 1.6291. Retained negative norm010 increases forget loss from 2.0880 to 2.1078 and obtains final global loss 1.6132, below the FedHDS baseline 1.6201. The forgetting gain is modest, but the utility behavior is favorable.

These results strengthen the Stage-3 interpretation. The key claim is not that retained-set Hessian always maximizes forget-client loss. Instead, the evidence shows that retained-set Hessian with negative direction and guard constraints gives a more controllable forgetting/utility trade-off across model and data scales than the naive forget-batch Hessian baseline.

The complete Stage-4 result tables are shown below.

{chr(10).join(detail_tables)}
"""


def stage3_text(rows: list[dict[str, str]]) -> str:
    g = by_group(rows)
    full, direction, neg = result_tables(rows)
    fb = g["forget_batch_hessian"]
    auto = g["retained_auto_direction_guard05"]
    n005 = g["retained_negative_norm005_guard05"]
    n010 = g["retained_negative_norm010_guard10"]
    n015 = g["retained_negative_norm015_guard10"]

    return f"""阶段三完整实验结果整理
最后更新：2026-05-04

一、当前状态

阶段三完整实验已经在 5090 云服务器上完成，并已下载、解压到本地。本次阶段三不再只是最小验证，而是完整的 9 组方向与 guard 消融实验。

本地结果目录：
F:\\FedHDS-zhaoge\\formal_cloud_results\\stage3_full_direction_guard_20260504

关键文件：
1. summary.csv：9 组实验汇总。
2. paper_table.csv / paper_table.md：论文表格。
3. figures：12 个图，包含 png 和 pdf。
4. 各组 final_results.json 和 logs：保留原始证据。

二、阶段三定位

阶段三用于解释阶段二中 retained-set guard 的失败模式：retained-set guard 能保护 global utility，但默认 positive/auto update sign 会让 forget-client loss 下降。阶段三因此集中验证三个问题：

1. retained-set Hessian update 是否存在方向不一致。
2. forget-loss guard 能否拒绝导致 forget loss 下降的无效小步。
3. negative sign 是否可以在 global guard 下显著提高 forget loss。

阶段三仍然使用 Qwen2-0.5B 和 Dolly，不是大模型大数据泛化实验。它的作用是完整方向消融和机制诊断。

三、实验设置

- model = Qwen2-0.5B
- dataset = Dolly
- data_sample = 0.4
- rounds = 20
- num_clients = 20
- client fraction k = 0.2
- local_step = 2
- batch_size = 1
- max_length = 64
- iid = 0
- forget_client_idx = 0
- lissa_depth = 1
- lissa_damping = 0.01
- unlearn_grad_sample_size = 8
- hessian_ref_per_client = 1
- retained reference set size = 19

FedHDS baseline final global loss = {f4(g['fedhds']['final_global_loss'])}。

四、完整 9 组结果表

{table_md(full)}

五、方向验证与 forget-loss guard

{table_md(direction)}

结果说明：

1. retained_auto_direction_guard05 的 selected update sign 为 positive，forget loss 从 {f4(auto['forget_client_loss_before_unlearning'])} 下降到 {f4(auto['forget_client_loss_after_unlearning'])}，下降 {gain(auto['forget_client_loss_before_unlearning'], auto['forget_client_loss_after_unlearning'])}。这复现了阶段二问题：global loss 可控，但 forgetting 目标没有实现。
2. 启用 forget-loss guard 后，tol=0 和 tol=0.005 两个配置都在第一个小步被拒绝，accepted steps 为 0/1，最终 global loss 和 forget loss 均保持 FedHDS baseline 状态。
3. 这说明 forget-loss guard 的作用是有效的：它能阻止“global utility 看起来安全，但 forget loss 下降”的无效 unlearning 更新。

六、negative sign norm sweep

{table_md(neg)}

结果说明：

1. max_norm=0.05 时，forget loss 从 {f4(n005['forget_client_loss_before_unlearning'])} 上升到 {f4(n005['forget_client_loss_after_unlearning'])}，提升 {gain(n005['forget_client_loss_before_unlearning'], n005['forget_client_loss_after_unlearning'])}；global loss 从 {f4(n005['global_loss_before_unlearning'])} 上升到 {f4(n005['global_loss_after_unlearning'])}，仍低于 guard=1.575。
2. max_norm=0.10 时，forget loss 提升到 {f4(n010['forget_client_loss_after_unlearning'])}，提升 {gain(n010['forget_client_loss_before_unlearning'], n010['forget_client_loss_after_unlearning'])}；global loss 为 {f4(n010['global_loss_after_unlearning'])}，仍低于 guard=1.625。
3. max_norm=0.15 时，forget loss 提升到 {f4(n015['forget_client_loss_after_unlearning'])}，但只接受 4/5 步，随后被 global_loss_guard 截停。最终实际 update L2 为 {f4(n015['actual_param_delta_l2_norm'])}，低于请求的 0.1500，说明 stepped guard 确实发生了截停。

七、阶段三核心结论

1. retained-set update 对 update sign 敏感。阶段二中 retained-set guard 的主要问题不是 guard 无效，而是默认 positive 方向与 forget objective 不一致。
2. forget-loss guard 可以拒绝使 forget loss 下降的小步，避免无效遗忘更新被接受。
3. negative sign 能显著提高 forget-client loss，同时 global loss 被 trust-region guard 控制在可接受范围内。
4. 在本次完整 9 组实验中，max_norm=0.05 是最保守配置，max_norm=0.10 提供更强 forgetting 且 global loss 仍可控，max_norm=0.15 则展示了 global guard 的截停作用。
5. 阶段三为论文提供了更完整的消融证据：方向验证、forget-loss guard、negative sign 强度和 global guard 截停都在同一套实验中得到记录。

八、论文写法建议

阶段三可以写成：

Stage-3 full direction and guard ablation shows that the retained-set curvature update is sign-sensitive. The original positive direction preserves global utility but reduces forget-client loss, revealing a direction mismatch. The forget-loss guard rejects such invalid steps. After switching to the negative sign, retained-set guarded unlearning substantially increases forget-client loss while keeping global loss within the trust-region guard.

对应中文表述：

阶段三完整方向与 guard 消融表明，retained-set curvature update 对更新方向敏感。原 positive 方向能够保持 global utility，但会降低 forget-client loss，说明该方向与遗忘目标不一致。forget-loss guard 能拒绝这类无效小步。切换到 negative sign 后，retained-set guarded unlearning 能显著提高 forget-client loss，同时 global loss 仍被 trust-region guard 控制。

九、结论边界

阶段三已经是完整的方向与 guard 消融，但仍不是多 seed、多 forget client 或大模型大数据泛化实验。因此论文中可以强调机制诊断和消融证据，但不应宣称已经完成大规模泛化验证。

十、相关图片

阶段三图像位于：
F:\\FedHDS-zhaoge\\formal_cloud_results\\stage3_full_direction_guard_20260504\\figures

包含：
1. forget_loss_before_after.png / .pdf
2. global_loss_final.png / .pdf
3. update_norm_vs_global_loss.png / .pdf
4. update_norm_vs_forget_gain.png / .pdf
5. guard_steps.png / .pdf
6. unlearning_time.png / .pdf
"""


def summary_text(
    rows: list[dict[str, str]],
    stage4_sets: list[dict[str, object]] | None = None,
    seed_rows: list[dict[str, str]] | None = None,
) -> str:
    if stage4_sets is None:
        stage4_sets = load_stage4_sets()
    if seed_rows is None:
        seed_rows = load_seed_rows()
    stage3 = stage3_text(rows)
    s4_text = stage4_text(rows, stage4_sets) if stage4_sets else ""
    stage4c_rows: list[dict[str, str]] = []
    for item in stage4_sets:
        if item.get("tag") == "Stage 4C":
            stage4c_rows = item["rows"]  # type: ignore[assignment]
            break
    seed_text = seed_robustness_text(stage4c_rows, seed_rows) if stage4c_rows and seed_rows else ""
    return f"""四阶段论文实验总结与写作总思路
最后更新：2026-05-04

一、论文总定位

当前小论文建议定位为：

FedHDS-assisted Federated Unlearning with Retained-Set Curvature and Guarded Trust-Region Updates

中文可以写成：

一种结合 FedHDS 训练底座、保留集曲率近似和受控 trust-region 更新的联邦遗忘方法。

论文主线不是宣称 retained-set Hessian 在所有指标上全面优于 forget-batch Hessian，而是提出并验证一条可运行、可控、能保护 global utility 的联邦遗忘链路。更准确的贡献是：

1. 构建 FedHDS + federated unlearning 的完整实验链路。
2. 提出 retained-set curvature approximation，用非遗忘客户端的 reference set 估计保留任务附近的曲率。
3. 引入 update norm clipping 与 stepped trust-region guard，使二阶 unlearning 更新可截停、可回滚、可按 global loss 控制。
4. 通过阶段三完整 9 组方向与 guard 消融发现 retained-set update 的 sign 敏感性，并用 forget-loss guard 与 negative sign 修正遗忘方向。
5. 通过阶段四 2 x 2 模型规模和数据规模泛化矩阵，验证该 trade-off 在更大数据和更大模型设置下仍具有可解释性。
6. 通过大模型大数据 seed robustness 补充，确认 guard 机制的 utility-safe 行为，同时明确 forgetting gain 存在 seed 敏感性。

二、四阶段实验角色

阶段一：云端最小正式闭环。证明 FL、FedHDS、forget-batch Hessian unlearning、retained-set Hessian + stepped guard 以及 summary/table/figures 工具链都能跑通。

阶段二：主要正式实验。在 data_sample=0.4、rounds=20 设置下形成论文主实验结果。它展示 FedHDS 训练底座的 utility 优势，也展示 forget-batch Hessian 和 retained-set guard 的 trade-off。

阶段三：完整方向与 guard 消融。它解释阶段二 retained-set guard 中 forget loss 下降的问题，验证 positive/auto 方向可能与遗忘目标不一致，forget-loss guard 能拒绝无效小步，negative sign 能显著提高 forget loss，同时 global loss 保持可控或被 global guard 截停。

阶段四：模型规模和数据规模泛化验证。它把阶段三的小模型小数据结果作为 2 x 2 矩阵的一个格子，并补充小模型大数据、大模型小数据、大模型大数据三组完整 9 组实验，用于验证方向修正和 guard 机制是否能跨规模保持可解释的 forgetting / utility trade-off。

稳定性补充：大模型大数据 seed robustness。它不是阶段五主实验，而是在 Stage 4C 的 Qwen2.5-1.5B + data_sample=0.6 设置下补充 seed=43/44 核心 5 组，验证 seed=42 结果的稳定性和边界。

三、阶段一结果摘要

阶段一配置：
- Qwen2-0.5B
- Dolly
- data_sample = 0.2
- rounds = 10
- num_clients = 20
- batch_size = 1
- max_length = 64
- forget_client_idx = 0

阶段一主结果：
- FL final global loss = 1.6443
- FedHDS final global loss = 1.6657
- forget-batch Hessian forget loss = 2.3381 -> 2.3489 (+0.0108)
- forget-batch Hessian final global loss = 1.6867
- retained-set guard forget loss = 2.3381 -> 2.3460 (+0.0079)
- retained-set guard final global loss = 1.6615

四、阶段二结果摘要

阶段二配置：
- Qwen2-0.5B
- Dolly
- data_sample = 0.4
- rounds = 20
- num_clients = 20
- batch_size = 1
- max_length = 64
- forget_client_idx = 0
- hessian_ref_per_client = 1
- global_guard = FedHDS loss + 0.05

阶段二主结果：
- FL final global loss = 1.5441
- FedHDS final global loss = 1.5250
- forget-batch Hessian forget loss = 2.2581 -> 2.2866 (+0.0285)
- forget-batch Hessian global loss = 1.5250 -> 1.6257 (+0.1007)
- retained-set auto guard forget loss = 2.2581 -> 2.2329 (-0.0252)
- retained-set auto guard global loss = 1.5250 -> 1.5482 (+0.0232)

阶段二结论：
1. FedHDS 在扩大设置下 final global loss 低于 FL，说明它是合理训练底座。
2. forget-batch Hessian 的 forgetting signal 更强，但 utility 损伤明显。
3. retained-set Hessian + global guard 能保护 global utility，但默认方向使 forget loss 下降，因此需要阶段三诊断。

五、阶段三结果摘要

{stage3}

十一、阶段四结果摘要

{s4_text}

十二、阶段四稳定性补充

{seed_text}

十三、论文结果叙述建议

建议结果章节按以下逻辑写：

1. 阶段一证明方法链路可运行。
2. 阶段二显示 FedHDS 在更长训练和更大采样下是更好的 utility baseline。
3. forget-batch Hessian 可以提升 forget loss，但明显增加 global loss。
4. retained-set guard 可以保护 global utility，但原 positive/auto 方向在阶段二会降低 forget loss。
5. 阶段三说明问题来自 update direction misalignment；forget-loss guard 能拒绝无效小步；negative sign 能在 global guard 下显著提升 forget loss。
6. 阶段四进一步说明：在小模型大数据、大模型小数据、大模型大数据设置下，retained-set negative + guard 通常比 naive forget-batch Hessian 提供更可控的 utility-preserving trade-off。
7. seed robustness 说明：大模型大数据下 global utility 控制较稳定，但具体 forgetting gain 对随机种子敏感，因此论文结论应强调 guard 和 trade-off，而不是绝对稳定遗忘增益。

十四、论文贡献边界

可以写：
- 方法链路完整可运行。
- retained-set curvature + clipping + stepped guard 能控制 unlearning 更新。
- retained-set guard 对 global utility 更友好。
- 方向修正后，forget loss 可以明显上升。
- 阶段四 2 x 2 矩阵提供了模型规模和数据规模泛化证据。
- seed robustness 提供了大模型大数据下的稳定性与边界证据。

不应写：
- retained-set Hessian 全面优于 forget-batch Hessian。
- 当前实验已经证明所有模型、所有数据集、所有隐私攻击场景下都有效。
- 阶段四已经覆盖多数据集、多模型家族或多 forget client。
- retained negative 在所有 seed 上都稳定提高 forget loss。

十五、后续工作和 limitation

1. 阶段三和阶段四虽然已经完成消融与规模泛化，但仍缺少多 forget client、多数据集和多模型家族验证。
2. retained-set Hessian 的计算成本高于 forget-batch Hessian。
3. retained reference size 和 guard threshold 对显存与结果较敏感。
4. seed robustness 已经显示 global utility 控制较稳定，但 forgetting gain 对随机种子敏感，后续需要更多 seed 和不同 forget client 验证。
"""


def update_paper_md(
    rows: list[dict[str, str]],
    stage4_sets: list[dict[str, object]] | None = None,
    seed_rows: list[dict[str, str]] | None = None,
) -> None:
    if stage4_sets is None:
        stage4_sets = load_stage4_sets()
    if seed_rows is None:
        seed_rows = load_seed_rows()
    path = ROOT / "paper_draft.md"
    text = path.read_text(encoding="utf-8")
    full, direction, neg = result_tables(rows)
    stage3_section = f"""### 5.3 Stage 3: Full Direction and Guard Ablation

Stage 3 is designed to explain the Stage-2 retained-set failure mode. It keeps the Stage-2 base configuration and runs a complete nine-setting direction and guard ablation. The goal is not to claim large-scale generalization, but to determine whether the retained-set update direction is misaligned, whether forget-loss guard can reject invalid forgetting steps, and whether a corrected negative direction can improve forgetting while preserving global utility.

Table 3 reports the full Stage-3 ablation.

{table_md(full)}

The first three rows reproduce the Stage-2 baseline setting in the same result directory. FedHDS obtains final global loss 1.5250, while FL obtains 1.5441. The forget-batch Hessian baseline increases forget-client loss from 2.2581 to 2.2866, but it also raises global loss to 1.6257. This confirms the Stage-2 trade-off: direct forget-batch curvature is aligned with forgetting, but it is costly for global utility.

The retained_auto_direction_guard05 row reproduces the retained-set failure mode. The selected positive direction keeps global loss controlled at 1.5482, but decreases forget-client loss from 2.2581 to 2.2329. The best probe forget delta is -0.0073, which indicates that the candidate direction does not improve forgetting. When forget-loss guard is enabled, both tolerance settings reject the first candidate step. The model remains at the FedHDS baseline with update L2 0 and 0/1 accepted steps. This is important guard behavior: the system avoids accepting an update that would look safe under global utility but fail the forgetting objective.

Table 4 focuses on the negative-sign norm sweep.

{table_md(neg)}

The negative-sign sweep provides the key correction evidence. With max_norm 0.05, forget-client loss increases from 2.2581 to 2.4105, while global loss remains 1.5561, below the guard 1.575. With max_norm 0.10, forget-client loss increases further to 2.6104, while global loss remains 1.5935, below the guard 1.625. With max_norm 0.15, forget-client loss increases to 2.6880, but the update is stopped after 4/5 accepted steps by the global loss guard. The final accepted update L2 is 0.1200 rather than the requested 0.1500, showing that the stepped guard actively truncates the update when utility approaches the guard boundary.

These results show that retained-set curvature is sign-sensitive. The original positive direction is not a reliable forgetting direction under the Stage-2 configuration. After switching to the negative direction and enabling forget-loss guard, the retained-set update can substantially increase forget-client loss while keeping global loss controlled. Among the three negative-sign settings, max_norm 0.05 is the most conservative and has the smallest global loss increase, while max_norm 0.10 gives a stronger forgetting gain with still-controlled global loss. The max_norm 0.15 case is useful as a guard-stopping example rather than the preferred setting.
"""
    stage4_section = paper_stage4_section(rows, stage4_sets) if stage4_sets else ""
    stage4c_rows: list[dict[str, str]] = []
    for item in stage4_sets:
        if item.get("tag") == "Stage 4C":
            stage4c_rows = item["rows"]  # type: ignore[assignment]
            break
    seed_section = paper_seed_section(stage4c_rows, seed_rows) if stage4c_rows and seed_rows else ""
    text = re.sub(
        r"### 5\.3 Stage 3:.*?(?=### 5\.[456] Cross-stage Discussion)",
        stage3_section + "\n" + stage4_section + "\n" + seed_section + "\n",
        text,
        flags=re.S,
    )
    text = re.sub(r"### 5\.[45] Cross-stage Discussion", "### 5.6 Cross-stage Discussion", text)
    text = text.replace(
        "Experiments on Dolly with Qwen2-0.5B are organized in three stages. The first stage verifies that the full FedHDS-unlearning pipeline can run end to end. The second stage shows the main trade-off: forget-batch Hessian increases forget-client loss but causes a larger global loss increase, while retained-set guard preserves global utility but suffers from direction misalignment. The third stage diagnoses this issue and shows that negative-sign correction with forget-loss guard can substantially increase forget-client loss while keeping global loss within the guard.",
        "Experiments on Dolly are organized in four stages. The first stage verifies that the full FedHDS-unlearning pipeline can run end to end with Qwen2-0.5B. The second stage shows the main trade-off: forget-batch Hessian increases forget-client loss but causes a larger global loss increase, while retained-set guard preserves global utility but suffers from direction misalignment. The third stage diagnoses this issue and shows that negative-sign correction with forget-loss guard can substantially increase forget-client loss while keeping global loss within the guard. The fourth stage builds a 2 x 2 model/data scale matrix with Qwen2-0.5B and Qwen2.5-1.5B under data_sample 0.4 and 0.6 to test whether the corrected trade-off generalizes across scale.",
    )
    text = text.replace(
        "We evaluate the method in three stages. Stage 1 verifies that the full pipeline can run on a cloud GPU. Stage 2 provides the main trade-off evidence under a larger setting: forget-batch Hessian produces a clearer forgetting signal but increases global loss substantially, whereas retained-set guard keeps global loss close to the FedHDS baseline but exposes an update-direction problem. Stage 3 then diagnoses this direction problem and shows that negative-sign correction with forget-loss guard can increase forget-client loss while keeping global loss controlled.",
        "We evaluate the method in four stages. Stage 1 verifies that the full pipeline can run on a cloud GPU. Stage 2 provides the main trade-off evidence under a larger setting: forget-batch Hessian produces a clearer forgetting signal but increases global loss substantially, whereas retained-set guard keeps global loss close to the FedHDS baseline but exposes an update-direction problem. Stage 3 then diagnoses this direction problem and shows that negative-sign correction with forget-loss guard can increase forget-client loss while keeping global loss controlled. Stage 4 evaluates model/data scale generalization through a 2 x 2 matrix.",
    )
    text = text.replace(
        "- We provide a cautious empirical interpretation: retained-set curvature is useful for utility preservation, but it requires direction validation and should not be presented as universally stronger than forget-batch Hessian.",
        "- We provide a cautious empirical interpretation: retained-set curvature is useful for utility preservation, but it requires direction validation and should not be presented as universally stronger than forget-batch Hessian.\n- We add a 2 x 2 model/data scale validation showing that the retained-set negative update remains a controllable trade-off across small/large model and small/large data settings.",
    )
    text = text.replace(
        "Experiments use Qwen2-0.5B on the Dolly dataset. We use this model size because it is large enough to represent a language-model fine-tuning setting, while still feasible for repeated Hessian-vector product experiments on a single RTX 5090 32GB GPU. The federated setting uses 20 clients, client fraction 0.2, local step 2, batch size 1, maximum sequence length 64, and forget client index 0.",
        "Experiments use Dolly as the federated instruction-tuning dataset. Stages 1-3 use Qwen2-0.5B because it is large enough to represent a language-model fine-tuning setting while still feasible for repeated Hessian-vector product experiments on a single RTX 5090 32GB GPU. Stage 4 additionally uses Qwen2.5-1.5B to evaluate model-scale generalization. The federated setting uses 20 clients, client fraction 0.2, local step 2, batch size 1, maximum sequence length 64, and forget client index 0.",
    )
    text = text.replace(
        "The experiments are organized into three stages.",
        "The experiments are organized into four stages."
    )
    text = text.replace(
        "Stage 3 is a full direction and guard ablation. It keeps the Stage-2 base configuration and tests update sign, forget-loss guard, and negative-sign norm sweep. This stage is not intended as a full large-scale experiment. Its purpose is to diagnose why the Stage-2 retained-set update preserves global utility but decreases forget-client loss, and to verify whether a corrected direction can satisfy both forgetting and utility constraints.",
        "Stage 3 is a full direction and guard ablation. It keeps the Stage-2 base configuration and tests update sign, forget-loss guard, and negative-sign norm sweep. Its purpose is to diagnose why the Stage-2 retained-set update preserves global utility but decreases forget-client loss, and to verify whether a corrected direction can satisfy both forgetting and utility constraints.\n\nStage 4 is a model/data scale generalization study. It uses Stage 3 as the small-model/small-data cell, then adds small-model/large-data, large-model/small-data, and large-model/large-data full nine-setting runs. This stage tests whether the retained-set negative direction and guard behavior remain meaningful when model size and data sample increase.",
    )
    text = text.replace(
        "Because Stage 3 is diagnostic but complete within the direction/guard design but complete within the direction/guard design, its main evidence is not a new FL/FedHDS comparison, but the relationship among update sign, forget-loss guard, forget-client loss change, and global guard behavior.",
        "Because Stage 3 is a complete direction/guard ablation, its main evidence is not a new FL/FedHDS comparison, but the relationship among update sign, forget-loss guard, forget-client loss change, and global guard behavior. Stage 4 then tests the same core retained-negative configuration under larger model and data settings."
    )
    text = text.replace(
        "Across the three stages, the results support a coherent interpretation of the method.",
        "Across the four stages, the results support a coherent interpretation of the method."
    )
    text = text.replace(
        "Fourth, the guard mechanisms make the unlearning update auditable. Update norm, clipping coefficient, accepted steps, rejected steps, stop reason, and post-step losses are all recorded. This is useful for a federated unlearning setting because an approximate unlearning update should not be a black-box parameter jump. The method provides an explicit record of why an update was accepted or rejected.\n\nThe final experimental message is therefore: retained-set curvature with trust-region guard is a utility-preserving unlearning framework, but it requires direction validation. With negative-sign correction and forget-loss guard, it can increase forget-client loss while keeping global loss within guard constraints.",
        "Fourth, Stage 4 shows that the corrected retained-set negative update is not limited to the original small-model/small-data setting. In the small-model/large-data setting, retained negative unlearning improves forget loss with much lower global utility cost than forget-batch Hessian. In the large-model settings, forget-batch Hessian no longer provides a stable forgetting gain, while retained negative unlearning still increases forget loss and keeps global loss close to or below the FedHDS baseline.\n\nFifth, the guard mechanisms make the unlearning update auditable. Update norm, clipping coefficient, accepted steps, rejected steps, stop reason, and post-step losses are all recorded. This is useful for a federated unlearning setting because an approximate unlearning update should not be a black-box parameter jump. The method provides an explicit record of why an update was accepted or rejected.\n\nThe final experimental message is therefore: retained-set curvature with trust-region guard is a utility-preserving unlearning framework, but it requires direction validation. With negative-sign correction and forget-loss guard, it can increase forget-client loss while keeping global loss within guard constraints. Stage 4 further shows that this interpretation remains meaningful when the data scale, model scale, or both are increased within the current Dolly/Qwen experimental scope."
    )
    text = text.replace(
        "First, the Stage-3 experiment is a complete direction and guard ablation, but it is still not a large-scale generalization study. It answers a specific question: whether the retained-set update direction is misaligned and whether negative-sign correction with forget-loss guard can fix the observed behavior. It does not yet provide multi-seed or multi-forget-client evidence. Therefore, Stage 3 should be interpreted as a diagnostic ablation, not as a full generalization study.",
        "First, the experiments still use a single dataset family and a fixed forget client. Stage 4 adds model/data scale validation, but it does not yet provide multi-dataset, multi-model-family, multi-forget-client, or privacy-attack evidence. Therefore, the scale results should be interpreted as controlled generalization within the current FedHDS/Dolly/Qwen setting rather than universal validation."
    )
    text = text.replace(
        "Fourth, the current experiments use Qwen2-0.5B and Dolly. This is suitable for a small-paper experimental setting and for validating the full unlearning chain, but it does not prove behavior on larger models, larger datasets, or more diverse client distributions. Future work should evaluate additional seeds, different forget clients, larger data samples, and eventually larger models such as 1.5B-scale Qwen variants.",
        "Fourth, the current experiments use Dolly and the Qwen/Qwen2.5 model family. This is suitable for validating the full unlearning chain and for a controlled scale study, but it does not prove behavior on other datasets, other model families, or more diverse client distributions. Future work should evaluate additional seeds, different forget clients, multiple datasets, and other model families."
    )
    text = text.replace(
        "The experiments lead to three main conclusions. First, the full pipeline is feasible: FL, FedHDS, forget-batch Hessian unlearning, retained-set Hessian unlearning, guard-based update control, summary generation, and plotting can run end to end on the cloud GPU setting. Second, the main Stage-2 experiment shows a clear trade-off. The forget-batch Hessian baseline produces a stronger forgetting signal, but it also causes a larger global loss increase. The retained-set guarded update preserves global utility more effectively, but its original direction can decrease forget-client loss. Third, Stage 3 explains and corrects this behavior. Retained-set updates are sign-sensitive; with forget-loss guard and negative-sign correction, forget-client loss can be substantially increased while global loss remains within guard constraints.",
        "The experiments lead to four main conclusions. First, the full pipeline is feasible: FL, FedHDS, forget-batch Hessian unlearning, retained-set Hessian unlearning, guard-based update control, summary generation, and plotting can run end to end on the cloud GPU setting. Second, the main Stage-2 experiment shows a clear trade-off. The forget-batch Hessian baseline produces a stronger forgetting signal, but it also causes a larger global loss increase. The retained-set guarded update preserves global utility more effectively, but its original direction can decrease forget-client loss. Third, Stage 3 explains and corrects this behavior. Retained-set updates are sign-sensitive; with forget-loss guard and negative-sign correction, forget-client loss can be substantially increased while global loss remains within guard constraints. Fourth, Stage 4 shows that this retained-negative guarded trade-off remains meaningful across a 2 x 2 model/data scale matrix, although the exact forgetting gain remains scale-dependent."
    )
    text = text.replace(
        "Future work should extend this validation to multiple seeds, multiple forget clients, larger data samples, and larger language models. Efficiency improvements for retained-set HVP computation are also important for practical deployment.",
        "Future work should extend this validation to multiple seeds, multiple forget clients, multiple datasets, other model families, and privacy-oriented attacks such as membership inference or canary extraction. Efficiency improvements for retained-set HVP computation are also important for practical deployment."
    )
    text = text.replace(
        "Stage 3 is a direction validation and small ablation.",
        "Stage 3 is a full direction and guard ablation.",
    )
    text = text.replace(
        "Stage 3 is diagnostic",
        "Stage 3 is diagnostic but complete within the direction/guard design",
    )
    text = text.replace(
        "First, the Stage-3 experiment is a direction-validation study rather than a complete large-scale experiment.",
        "First, the Stage-3 experiment is a complete direction and guard ablation, but it is still not a large-scale generalization study.",
    )
    text = text.replace(
        "Experiments on Dolly are organized in four stages. The first stage verifies that the full FedHDS-unlearning pipeline can run end to end with Qwen2-0.5B. The second stage shows the main trade-off: forget-batch Hessian increases forget-client loss but causes a larger global loss increase, while retained-set guard preserves global utility but suffers from direction misalignment. The third stage diagnoses this issue and shows that negative-sign correction with forget-loss guard can substantially increase forget-client loss while keeping global loss within the guard. The fourth stage builds a 2 x 2 model/data scale matrix with Qwen2-0.5B and Qwen2.5-1.5B under data_sample 0.4 and 0.6 to test whether the corrected trade-off generalizes across scale.",
        "Experiments on Dolly are organized in four stages plus a seed robustness check. The first stage verifies that the full FedHDS-unlearning pipeline can run end to end with Qwen2-0.5B. The second stage shows the main trade-off: forget-batch Hessian increases forget-client loss but causes a larger global loss increase, while retained-set guard preserves global utility but suffers from direction misalignment. The third stage diagnoses this issue and shows that negative-sign correction with forget-loss guard can substantially increase forget-client loss while keeping global loss within the guard. The fourth stage builds a 2 x 2 model/data scale matrix with Qwen2-0.5B and Qwen2.5-1.5B under data_sample 0.4 and 0.6 to test whether the corrected trade-off generalizes across scale. The seed check repeats the large-model/large-data core settings to distinguish stable utility protection from seed-sensitive forgetting gain.",
    )
    text = text.replace(
        "We evaluate the method in four stages. Stage 1 verifies that the full pipeline can run on a cloud GPU. Stage 2 provides the main trade-off evidence under a larger setting: forget-batch Hessian produces a clearer forgetting signal but increases global loss substantially, whereas retained-set guard keeps global loss close to the FedHDS baseline but exposes an update-direction problem. Stage 3 then diagnoses this direction problem and shows that negative-sign correction with forget-loss guard can increase forget-client loss while keeping global loss controlled. Stage 4 evaluates model/data scale generalization through a 2 x 2 matrix.",
        "We evaluate the method in four stages and one robustness check. Stage 1 verifies that the full pipeline can run on a cloud GPU. Stage 2 provides the main trade-off evidence under a larger setting: forget-batch Hessian produces a clearer forgetting signal but increases global loss substantially, whereas retained-set guard keeps global loss close to the FedHDS baseline but exposes an update-direction problem. Stage 3 then diagnoses this direction problem and shows that negative-sign correction with forget-loss guard can increase forget-client loss while keeping global loss controlled. Stage 4 evaluates model/data scale generalization through a 2 x 2 matrix. The seed robustness check reruns the large-model/large-data core settings to show that utility safety and guard behavior are more stable than the exact forgetting gain.",
    )
    scale_bullet = "- We add a 2 x 2 model/data scale validation showing that the retained-set negative update remains a controllable trade-off across small/large model and small/large data settings."
    seed_bullet = "- We add a seed robustness check on the large-model/large-data setting, showing stable utility control and meaningful guard behavior while acknowledging that exact forgetting gain is seed-sensitive."
    text = re.sub(
        rf"{re.escape(scale_bullet)}\n(?:{re.escape(scale_bullet)}\n)*",
        scale_bullet + "\n",
        text,
    )
    if seed_bullet not in text:
        text = text.replace(scale_bullet + "\n", scale_bullet + "\n" + seed_bullet + "\n")
    stage4_setup = "Stage 4 is a model/data scale generalization study. It uses Stage 3 as the small-model/small-data cell, then adds small-model/large-data, large-model/small-data, and large-model/large-data full nine-setting runs. This stage tests whether the retained-set negative direction and guard behavior remain meaningful when model size and data sample increase."
    seed_setup = "After Stage 4, a seed robustness check repeats the large-model/large-data core-five settings with seed 43 and seed 44. This check is reported as a robustness check rather than a separate experimental stage. Its purpose is to test whether the seed 42 Stage-4C result is an isolated accident and to identify which claims are stable across seeds."
    if seed_setup not in text:
        text = text.replace(stage4_setup, stage4_setup + "\n\n" + seed_setup)
    text = text.replace(
        "After Stage 4, a seed robustness check repeats the large-model/large-data core-five settings with seed 43 and seed 44. This check is not treated as a separate Stage 5. Its purpose is to test whether the seed 42 Stage-4C result is an isolated accident and to identify which claims are stable across seeds.",
        "",
    )
    text = re.sub(
        rf"(?:{re.escape(seed_setup)}\n\n)+",
        seed_setup + "\n\n",
        text,
    )
    text = text.replace(
        "Because Stage 3 is a complete direction/guard ablation, its main evidence is not a new FL/FedHDS comparison, but the relationship among update sign, forget-loss guard, forget-client loss change, and global guard behavior. Stage 4 then tests the same core retained-negative configuration under larger model and data settings.",
        "Because Stage 3 is a complete direction/guard ablation, its main evidence is not a new FL/FedHDS comparison, but the relationship among update sign, forget-loss guard, forget-client loss change, and global guard behavior. Stage 4 then tests the same core retained-negative configuration under larger model and data settings. The seed robustness check further separates two conclusions: global utility control and guard behavior are relatively stable, while the magnitude and sign of the forgetting gain can depend on the random seed.",
    )
    seed_metric_sentence = "The seed robustness check further separates two conclusions: global utility control and guard behavior are relatively stable, while the magnitude and sign of the forgetting gain can depend on the random seed."
    text = re.sub(rf"({re.escape(seed_metric_sentence)})(?:\s+\1)+", r"\1", text)
    seed_discussion = "Fifth, the seed robustness check prevents the Stage-4C result from being overclaimed. Seed 42 shows a positive retained-negative forgetting gain with global loss below FedHDS. Seed 43 preserves utility but gives a very small negative forgetting delta, and seed 44 shows the forget-loss guard rejecting the negative update and preserving the FedHDS state. The robust conclusion is therefore utility-safe and auditable behavior, not seed-independent forgetting improvement."
    if seed_discussion not in text:
        text = text.replace(
            "Fourth, Stage 4 shows that the corrected retained-set negative update is not limited to the original small-model/small-data setting. In the small-model/large-data setting, retained negative unlearning improves forget loss with much lower global utility cost than forget-batch Hessian. In the large-model settings, forget-batch Hessian no longer provides a stable forgetting gain, while retained negative unlearning still increases forget loss and keeps global loss close to or below the FedHDS baseline.\n\nFifth, the guard mechanisms make the unlearning update auditable.",
            "Fourth, Stage 4 shows that the corrected retained-set negative update is not limited to the original small-model/small-data setting. In the small-model/large-data setting, retained negative unlearning improves forget loss with much lower global utility cost than forget-batch Hessian. In the large-model settings, forget-batch Hessian no longer provides a stable forgetting gain, while retained negative unlearning still increases forget loss and keeps global loss close to or below the FedHDS baseline.\n\n" + seed_discussion + "\n\nSixth, the guard mechanisms make the unlearning update auditable.",
        )
    text = text.replace(
        "The final experimental message is therefore: retained-set curvature with trust-region guard is a utility-preserving unlearning framework, but it requires direction validation. With negative-sign correction and forget-loss guard, it can increase forget-client loss while keeping global loss within guard constraints. Stage 4 further shows that this interpretation remains meaningful when the data scale, model scale, or both are increased within the current Dolly/Qwen experimental scope.",
        "The final experimental message is therefore: retained-set curvature with trust-region guard is a utility-preserving unlearning framework, but it requires direction validation. With negative-sign correction and forget-loss guard, it can increase forget-client loss while keeping global loss within guard constraints. Stage 4 further shows that this interpretation remains meaningful when the data scale, model scale, or both are increased within the current Dolly/Qwen experimental scope. The seed robustness check adds an important boundary: the framework is more reliable as a guarded utility-preserving update path than as a guarantee of seed-independent forgetting gain.",
    )
    seed_boundary_sentence = "The seed robustness check adds an important boundary: the framework is more reliable as a guarded utility-preserving update path than as a guarantee of seed-independent forgetting gain."
    text = re.sub(rf"({re.escape(seed_boundary_sentence)})(?:\s+\1)+", r"\1", text)
    seed_limit = "Second, the seed robustness check is intentionally small. It repeats only the large-model/large-data core-five settings for two additional seeds. The results are valuable because they show stable utility control and useful guard rejection behavior, but they also show that exact forget-loss improvement is seed-sensitive. A stronger generalization claim would require more seeds and more forget clients."
    if seed_limit not in text:
        text = text.replace(
            "First, the experiments still use a single dataset family and a fixed forget client. Stage 4 adds model/data scale validation, but it does not yet provide multi-dataset, multi-model-family, multi-forget-client, or privacy-attack evidence. Therefore, the scale results should be interpreted as controlled generalization within the current FedHDS/Dolly/Qwen setting rather than universal validation.\n\nSecond, retained-set Hessian computation is more expensive than the forget-batch Hessian baseline.",
            "First, the experiments still use a single dataset family and a fixed forget client. Stage 4 adds model/data scale validation, but it does not yet provide multi-dataset, multi-model-family, multi-forget-client, or privacy-attack evidence. Therefore, the scale results should be interpreted as controlled generalization within the current FedHDS/Dolly/Qwen setting rather than universal validation.\n\n" + seed_limit + "\n\nThird, retained-set Hessian computation is more expensive than the forget-batch Hessian baseline.",
        )
        text = text.replace("Third, the method is sensitive to guard and reference settings.", "Fourth, the method is sensitive to guard and reference settings.")
        text = text.replace("Fourth, the current experiments use Dolly and the Qwen/Qwen2.5 model family.", "Fifth, the current experiments use Dolly and the Qwen/Qwen2.5 model family.")
    text = text.replace(
        "The experiments lead to four main conclusions. First, the full pipeline is feasible: FL, FedHDS, forget-batch Hessian unlearning, retained-set Hessian unlearning, guard-based update control, summary generation, and plotting can run end to end on the cloud GPU setting. Second, the main Stage-2 experiment shows a clear trade-off. The forget-batch Hessian baseline produces a stronger forgetting signal, but it also causes a larger global loss increase. The retained-set guarded update preserves global utility more effectively, but its original direction can decrease forget-client loss. Third, Stage 3 explains and corrects this behavior. Retained-set updates are sign-sensitive; with forget-loss guard and negative-sign correction, forget-client loss can be substantially increased while global loss remains within guard constraints. Fourth, Stage 4 shows that this retained-negative guarded trade-off remains meaningful across a 2 x 2 model/data scale matrix, although the exact forgetting gain remains scale-dependent.",
        "The experiments lead to five main conclusions. First, the full pipeline is feasible: FL, FedHDS, forget-batch Hessian unlearning, retained-set Hessian unlearning, guard-based update control, summary generation, and plotting can run end to end on the cloud GPU setting. Second, the main Stage-2 experiment shows a clear trade-off. The forget-batch Hessian baseline produces a stronger forgetting signal, but it also causes a larger global loss increase. The retained-set guarded update preserves global utility more effectively, but its original direction can decrease forget-client loss. Third, Stage 3 explains and corrects this behavior. Retained-set updates are sign-sensitive; with forget-loss guard and negative-sign correction, forget-client loss can be substantially increased while global loss remains within guard constraints. Fourth, Stage 4 shows that this retained-negative guarded trade-off remains meaningful across a 2 x 2 model/data scale matrix, although the exact forgetting gain remains scale-dependent. Fifth, the seed robustness check shows that utility control and guard rejection behavior are stable enough to support a cautious claim, while exact forget-loss improvement should be reported as seed-sensitive.",
    )
    for sentence in (seed_setup, seed_metric_sentence, seed_boundary_sentence):
        text = re.sub(rf"({re.escape(sentence)})(?:\s+\1)+", r"\1", text)
    path.write_text(text, encoding="utf-8")


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        sig = f.read(24)
    if sig[:8] != b"\x89PNG\r\n\x1a\n":
        return (900, 500)
    width, height = struct.unpack(">II", sig[16:24])
    return width, height


class DocxBuilder:
    def __init__(self, title: str):
        self.title = title
        self.body: list[str] = []
        self.rels: list[tuple[str, str, str]] = []
        self.media: list[tuple[Path, str]] = []
        self.rid = 1

    def inline_runs(self, text: str) -> str:
        runs: list[str] = []
        for part in re.split(r"(<sup>.*?</sup>)", text, flags=re.S):
            if not part:
                continue
            sup = re.fullmatch(r"<sup>(.*?)</sup>", part, flags=re.S)
            if sup:
                runs.append(
                    '<w:r><w:rPr><w:vertAlign w:val="superscript"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
                    f"<w:t>{html.escape(sup.group(1))}</w:t></w:r>"
                )
            else:
                runs.append(f"<w:r><w:t>{html.escape(part)}</w:t></w:r>")
        return "".join(runs) or "<w:r><w:t></w:t></w:r>"

    def p(self, text: str = "", style: str | None = None) -> None:
        style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
        runs = self.inline_runs(text.splitlines()[0] if text.splitlines() else "")
        self.body.append(f"<w:p><w:pPr>{style_xml}</w:pPr>{runs}</w:p>")

    def title_heading(self, text: str) -> None:
        self.p(text, "Title")

    def page_break(self) -> None:
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def toc(self, title: str, placeholder: str) -> None:
        instr = html.escape('TOC \\o "1-3" \\h \\z \\u')
        self.body.append(
            '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
            '<w:r><w:rPr><w:b/><w:sz w:val="32"/></w:rPr>'
            f"<w:t>{html.escape(title)}</w:t></w:r></w:p>"
        )
        self.body.append(
            '<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
            f'<w:r><w:instrText xml:space="preserve">{instr}</w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            f"<w:r><w:t>{html.escape(placeholder)}</w:t></w:r>"
            '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
        )

    def heading(self, text: str, level: int = 1) -> None:
        self.p(text, f"Heading{level}")

    def caption(self, text: str) -> None:
        self.body.append(
            '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
            '<w:r><w:rPr><w:b/><w:sz w:val="20"/></w:rPr>'
            f"<w:t>{html.escape(text)}</w:t></w:r></w:p>"
        )

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        ncols = max(len(row) for row in rows)
        cell_width = max(1000, min(2600, 9000 // max(ncols, 1)))
        xml = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="7F7F7F"/><w:left w:val="single" w:sz="4" w:space="0" w:color="7F7F7F"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="7F7F7F"/><w:right w:val="single" w:sz="4" w:space="0" w:color="7F7F7F"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/></w:tblBorders><w:tblLook w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>']
        for row_idx, row in enumerate(rows):
            header = row_idx == 0
            xml.append("<w:tr>")
            if header:
                xml.append("<w:trPr><w:tblHeader/></w:trPr>")
            for cell in row:
                shade = '<w:shd w:fill="D9EAF7"/>' if header else ""
                bold = "<w:b/>" if header else ""
                align = '<w:jc w:val="center"/>' if header else ""
                xml.append(
                    f"<w:tc><w:tcPr><w:tcW w:w=\"{cell_width}\" w:type=\"dxa\"/>"
                    f"<w:vAlign w:val=\"top\"/>{shade}</w:tcPr><w:p><w:pPr>{align}</w:pPr>"
                    f"<w:r><w:rPr>{bold}<w:sz w:val=\"19\"/></w:rPr><w:t>{html.escape(str(cell))}</w:t></w:r></w:p></w:tc>"
                )
            xml.append("</w:tr>")
        xml.append("</w:tbl>")
        self.body.append("".join(xml))

    def image(self, path: Path, caption: str) -> None:
        if not path.exists():
            return
        media_name = f"image{len(self.media) + 1}{path.suffix.lower()}"
        rid = f"rId{self.rid}"
        self.rid += 1
        self.media.append((path, media_name))
        self.rels.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"media/{media_name}"))
        width, height = png_size(path)
        max_cx = 5_800_000
        cx = max_cx
        cy = int(max_cx * height / max(width, 1))
        self.body.append(f"""
<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{self.rid + 100}" name="{html.escape(caption)}"/>
<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name="{html.escape(media_name)}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>""")
        self.caption(caption)

    def equation(self, math_xml: str) -> None:
        self.body.append(
            '<w:p><m:oMathPara><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>'
            f"<m:oMath>{math_xml}</m:oMath>"
            "</m:oMathPara></w:p>"
        )

    def equation_explanation(self, math_xml: str, explanation: str) -> None:
        self.body.append(
            '<w:p><w:r><w:t>- </w:t></w:r><m:oMath>'
            f"{math_xml}</m:oMath><w:r><w:t>{html.escape(explanation)}</w:t></w:r></w:p>"
        )

    def save(self, path: Path) -> None:
        document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<w:body>{''.join(self.body)}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body></w:document>"""
        styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="21"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
</w:styles>"""
        settings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:updateFields w:val="true"/></w:settings>"""
        rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        doc_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
        doc_rels.append('<Relationship Id="rStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
        doc_rels.append('<Relationship Id="rSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>')
        for rid, typ, target in self.rels:
            doc_rels.append(f'<Relationship Id="{rid}" Type="{typ}" Target="{target}"/>')
        doc_rels.append("</Relationships>")
        content_types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">']
        content_types.append('<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>')
        content_types.append('<Default Extension="xml" ContentType="application/xml"/>')
        content_types.append('<Default Extension="png" ContentType="image/png"/>')
        content_types.append('<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>')
        content_types.append('<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>')
        content_types.append('<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>')
        content_types.append("</Types>")
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", "".join(content_types))
            z.writestr("_rels/.rels", rels)
            z.writestr("word/document.xml", document)
            z.writestr("word/styles.xml", styles)
            z.writestr("word/settings.xml", settings)
            z.writestr("word/_rels/document.xml.rels", "".join(doc_rels))
            for src, name in self.media:
                z.write(src, f"word/media/{name}")


def add_text_to_doc(doc: DocxBuilder, text: str) -> None:
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            doc.p("")
            continue
        if re.match(r"^[一二三四五六七八九十]+、", line):
            doc.heading(line, 1)
        elif line.startswith("# "):
            doc.heading(line[2:], 1)
        elif line.startswith("## "):
            doc.heading(line[3:], 2)
        elif line.startswith("|"):
            continue
        else:
            doc.p(line)


def build_stage3_docx(text: str, rows: list[dict[str, str]]) -> None:
    full, direction, neg = result_tables(rows)
    doc = DocxBuilder("阶段三完整实验结果整理")
    doc.heading("阶段三完整实验结果整理", 1)
    add_text_to_doc(doc, text)
    doc.heading("阶段三完整 9 组结果表", 1)
    doc.table(full)
    doc.heading("方向验证与 forget-loss guard 表", 1)
    doc.table(direction)
    doc.heading("negative sign norm sweep 表", 1)
    doc.table(neg)
    doc.heading("阶段三图片", 1)
    for name in [
        "forget_loss_before_after.png",
        "global_loss_final.png",
        "update_norm_vs_global_loss.png",
        "update_norm_vs_forget_gain.png",
        "guard_steps.png",
        "unlearning_time.png",
    ]:
        doc.image(FIGURES / name, name)
    doc.save(ROOT / "readme_result_3.docx.docx")


def build_summary_docx(
    text: str,
    rows: list[dict[str, str]],
    stage4_sets: list[dict[str, object]] | None = None,
    seed_rows: list[dict[str, str]] | None = None,
) -> None:
    if stage4_sets is None:
        stage4_sets = load_stage4_sets()
    if seed_rows is None:
        seed_rows = load_seed_rows()
    full, direction, neg = result_tables(rows)
    doc = DocxBuilder("四阶段论文实验总结")
    doc.heading("四阶段论文实验总结与写作总思路", 1)
    add_text_to_doc(doc, text)
    doc.heading("阶段三完整 9 组结果表", 1)
    doc.table(full)
    if stage4_sets:
        doc.heading("阶段四 2 x 2 泛化矩阵表", 1)
        doc.table(scale_matrix_table(rows, stage4_sets))
        for item in stage4_sets:
            doc.heading(f"{item['tag']} 结果表", 1)
            doc.table(compact_result_table(item["rows"]))  # type: ignore[arg-type]
    stage4c_rows: list[dict[str, str]] = []
    for item in stage4_sets:
        if item.get("tag") == "Stage 4C":
            stage4c_rows = item["rows"]  # type: ignore[assignment]
            break
    if stage4c_rows and seed_rows:
        doc.heading("阶段四 seed robustness 表", 1)
        doc.table(seed_robustness_table(stage4c_rows, seed_rows))
    doc.heading("阶段三图片", 1)
    for folder, title in [
        (ROOT / "formal_cloud_results_formal_minimal_20260503_combined" / "figures", "阶段一图"),
        (ROOT / "formal_cloud_results" / "stage2_dsample04_round20_20260503" / "figures", "阶段二图"),
        (FIGURES, "阶段三图"),
        (STAGE4A / "figures", "阶段四A图"),
        (STAGE4B / "figures", "阶段四B图"),
        (STAGE4C / "figures", "阶段四C图"),
        (STAGE4_SEED / "figures", "阶段四seed汇总图"),
        (STAGE4_SEED / "seed43" / "figures", "阶段四seed43图"),
        (STAGE4_SEED / "seed44" / "figures", "阶段四seed44图"),
    ]:
        if folder.exists():
            for img in sorted(folder.glob("*.png")):
                doc.image(img, f"{title}: {img.name}")
    doc.save(ROOT / "Summary_of_the_three_stages_of_the_thesis.docx")
    doc.save(ROOT / "Summary_of_the_four_stages_of_the_thesis.docx")


def build_seed_docx(text: str, stage4c_rows: list[dict[str, str]], seed_rows: list[dict[str, str]]) -> None:
    doc = DocxBuilder("Stage 4 Seed Robustness")
    doc.heading("Stage 4 Seed Robustness Check", 1)
    add_text_to_doc(doc, text)
    doc.heading("Seed robustness table", 1)
    doc.table(seed_robustness_table(stage4c_rows, seed_rows))
    doc.heading("Seed robustness figures", 1)
    for folder, title in [
        (STAGE4_SEED / "figures", "seed summary"),
        (STAGE4_SEED / "seed43" / "figures", "seed43"),
        (STAGE4_SEED / "seed44" / "figures", "seed44"),
    ]:
        if folder.exists():
            for img in sorted(folder.glob("*.png")):
                doc.image(img, f"{title}: {img.name}")
    doc.save(ROOT / "readme_result4_seed_robustness.docx")


def split_md_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [re.sub(r"`([^`]+)`", r"\1", cell.strip()) for cell in line.split("|")]


def is_md_separator(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    if not stripped:
        return False
    return all(set(part.strip()) <= set("-: ") for part in stripped.split("|"))


def m_run(text: str) -> str:
    return f"<m:r><m:t>{html.escape(text)}</m:t></m:r>"


def m_sub(base: str, sub: str) -> str:
    return (
        "<m:sSub><m:e>"
        f"{m_run(base)}"
        "</m:e><m:sub>"
        f"{m_run(sub)}"
        "</m:sub></m:sSub>"
    )


def m_sup(base_xml: str, sup: str) -> str:
    return f"<m:sSup><m:e>{base_xml}</m:e><m:sup>{m_run(sup)}</m:sup></m:sSup>"


def m_subsup(base: str, sub: str, sup: str) -> str:
    return (
        "<m:sSubSup><m:e>"
        f"{m_run(base)}"
        "</m:e><m:sub>"
        f"{m_run(sub)}"
        "</m:sub><m:sup>"
        f"{m_run(sup)}"
        "</m:sup></m:sSubSup>"
    )


def m_frac(num_xml: str, den_xml: str) -> str:
    return f"<m:f><m:num>{num_xml}</m:num><m:den>{den_xml}</m:den></m:f>"


def formula_to_omml(line: str) -> str | None:
    cleaned = line.strip().strip("`")
    formulas: dict[str, str] = {
        "ΔLⱼ = Lⱼ(θᵤ) - Lⱼ(θ*)": "".join([
            m_sub("ΔL", "j"), m_run(" = "), m_sub("L", "j"), m_run("("),
            m_sub("θ", "u"), m_run(") - "), m_sub("L", "j"), m_run("(θ*)"),
        ]),
        "Lᴳ(θᵤ) ≤ γᴳ": "".join([
            m_sub("L", "G"), m_run("("), m_sub("θ", "u"), m_run(") <= "),
            m_sub("γ", "G"),
        ]),
        "gⱼ = ∇θ Lⱼ(θ*)": "".join([
            m_sub("g", "j"), m_run(" = ∇θ "), m_sub("L", "j"), m_run("(θ*)"),
        ]),
        "Dref = ⋃(k≠j) Sₖ,  Sₖ ⊂ Dₖ": "".join([
            m_sub("D", "ref"), m_run(" = "), m_sub("∪", "k≠j"), m_sub("S", "k"),
            m_run(",  "), m_sub("S", "k"), m_run(" ⊂ "), m_sub("D", "k"),
        ]),
        "v = Href⁻¹ gⱼ": "".join([
            m_run("v = "), m_sup(m_sub("H", "ref"), "-1"), m_run(" "), m_sub("g", "j"),
        ]),
        "vₜ₊₁ = gⱼ + (1 - λ)vₜ - Href vₜ": "".join([
            m_sub("v", "t+1"), m_run(" = "), m_sub("g", "j"), m_run(" + (1 - λ)"),
            m_sub("v", "t"), m_run(" - "), m_sub("H", "ref"), m_run(" "), m_sub("v", "t"),
        ]),
        "Δθraw = ηv / (K - 1)": "".join([
            m_sub("Δθ", "raw"), m_run(" = "),
            m_frac(m_run("η v"), m_run("K - 1")),
        ]),
        "Δθclip = Δθraw · min(1, τ / ||Δθraw||)": "".join([
            m_sub("Δθ", "clip"), m_run(" = "), m_sub("Δθ", "raw"), m_run(" · min(1, "),
            m_frac(m_run("τ"), "".join([m_run("||"), m_sub("Δθ", "raw"), m_run("||")])),
            m_run(")"),
        ]),
        "δₛ = Δθclip / S": "".join([
            m_sub("δ", "s"), m_run(" = "), m_frac(m_sub("Δθ", "clip"), m_run("S")),
        ]),
        "Lᴳ(θₜ + δₛ) ≤ γᴳ": "".join([
            m_sub("L", "G"), m_run("("), m_sub("θ", "t"), m_run(" + "), m_sub("δ", "s"),
            m_run(") <= "), m_sub("γ", "G"),
        ]),
        "θ⁺ = θ* + Δθclip": "".join([
            m_sup(m_run("θ"), "+"), m_run(" = θ* + "), m_sub("Δθ", "clip"),
        ]),
        "θ⁻ = θ* - Δθclip": "".join([
            m_sup(m_run("θ"), "-"), m_run(" = θ* - "), m_sub("Δθ", "clip"),
        ]),
        "Lⱼ(θₜ + δₛ) ≥ Lⱼᵍᵘᵃʳᵈ - ε": "".join([
            m_sub("L", "j"), m_run("("), m_sub("θ", "t"), m_run(" + "), m_sub("δ", "s"),
            m_run(") >= "), m_sup(m_sub("L", "j"), "guard"), m_run(" - ε"),
        ]),
    }
    return formulas.get(cleaned)


def symbol_to_omml(symbol: str) -> str | None:
    cleaned = symbol.strip().strip("`")
    symbols: dict[str, str] = {
        "K": m_run("K"),
        "k": m_run("k"),
        "Dₖ": m_sub("D", "k"),
        "θ*": m_run("θ*"),
        "θᵤ": m_sub("θ", "u"),
        "j": m_run("j"),
        "ΔLⱼ": m_sub("ΔL", "j"),
        "Lⱼ(θ*)": "".join([m_sub("L", "j"), m_run("(θ*)")]),
        "Lⱼ(θᵤ)": "".join([m_sub("L", "j"), m_run("("), m_sub("θ", "u"), m_run(")")]),
        "Lᴳ(θᵤ)": "".join([m_sub("L", "G"), m_run("("), m_sub("θ", "u"), m_run(")")]),
        "γᴳ": m_sub("γ", "G"),
        "gⱼ": m_sub("g", "j"),
        "Dref": m_sub("D", "ref"),
        "Sₖ": m_sub("S", "k"),
        "Href": m_sub("H", "ref"),
        "v": m_run("v"),
        "vₜ": m_sub("v", "t"),
        "vₜ₊₁": m_sub("v", "t+1"),
        "Href vₜ": "".join([m_sub("H", "ref"), m_run(" "), m_sub("v", "t")]),
        "λ": m_run("λ"),
        "η": m_run("η"),
        "K - 1": m_run("K - 1"),
        "Δθraw": m_sub("Δθ", "raw"),
        "||Δθraw||": "".join([m_run("||"), m_sub("Δθ", "raw"), m_run("||")]),
        "τ": m_run("τ"),
        "Δθclip": m_sub("Δθ", "clip"),
        "S": m_run("S"),
        "δₛ": m_sub("δ", "s"),
        "θₜ": m_sub("θ", "t"),
        "Lᴳ(θₜ + δₛ)": "".join([m_sub("L", "G"), m_run("("), m_sub("θ", "t"), m_run(" + "), m_sub("δ", "s"), m_run(")")]),
        "θ⁺": m_sup(m_run("θ"), "+"),
        "θ⁻": m_sup(m_run("θ"), "-"),
        "Lⱼᵍᵘᵃʳᵈ": m_sup(m_sub("L", "j"), "guard"),
        "ε": m_run("ε"),
    }
    return symbols.get(cleaned)


def variable_explanation_to_omml(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped.startswith("- "):
        return None
    item = stripped[2:].strip()
    for sep in ("：", ":"):
        if sep in item:
            symbol, explanation = item.split(sep, 1)
            math_xml = symbol_to_omml(symbol.strip())
            if math_xml:
                return math_xml, sep + explanation
    return None


def is_caption_line(line: str) -> str | None:
    stripped = line.strip()
    if not (stripped.startswith("**") and stripped.endswith("**")):
        return None
    text = stripped[2:-2].strip()
    if re.match(r"^(Table|Figure|表|图)\s*[A-Za-z0-9一二三四五六七八九十.：: -]", text):
        return text
    return None


def appendix_caption(stage: str, img: Path, index: int, zh: bool = False) -> str:
    stage_map_zh = {
        "Stage 1": "阶段一",
        "Stage 2": "阶段二",
        "Stage 3": "阶段三",
        "Stage 4A": "阶段四A",
        "Stage 4B": "阶段四B",
        "Stage 4C": "阶段四C",
        "Stage 4 seed summary": "阶段四 seed 汇总",
        "Stage 4 seed43": "阶段四 seed 43",
        "Stage 4 seed44": "阶段四 seed 44",
    }
    desc_en = {
        "forget_loss_before_after.png": "forget-client loss before and after unlearning",
        "global_loss_final.png": "final global loss comparison",
        "guard_steps.png": "accepted and attempted guard steps",
        "unlearning_time.png": "unlearning time comparison",
        "update_norm_vs_forget_gain.png": "update norm versus forget-loss gain",
        "update_norm_vs_global_loss.png": "update norm versus final global loss",
    }
    desc_zh = {
        "forget_loss_before_after.png": "遗忘前后 forget-client loss 对比",
        "global_loss_final.png": "final global loss 对比",
        "guard_steps.png": "guard attempted steps 与 accepted steps 对比",
        "unlearning_time.png": "unlearning time 对比",
        "update_norm_vs_forget_gain.png": "update norm 与 forget-loss gain 的关系",
        "update_norm_vs_global_loss.png": "update norm 与 final global loss 的关系",
    }
    if zh:
        stage_text = stage_map_zh.get(stage, stage)
        description = desc_zh.get(img.name, img.stem.replace("_", " "))
        return f"图 A{index}. {stage_text}：{description}。"
    description = desc_en.get(img.name, img.stem.replace("_", " "))
    return f"Figure A{index}. {stage}: {description}."


def markdown_to_docx(md_path: Path, out_path: Path) -> None:
    doc = DocxBuilder("Paper Draft")
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    zh = "_zh" in md_path.stem or md_path.name.endswith("_zh.md")
    inserted_toc = False
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            doc.title_heading(line[2:])
        elif line.startswith("## "):
            if not inserted_toc and re.match(r"##\s+1(?:\.|\s)", line.strip()):
                doc.toc("目录" if zh else "Table of Contents", "打开 Word 后更新域以生成目录。" if zh else "Update fields in Word to generate the table of contents.")
                doc.page_break()
                inserted_toc = True
            doc.heading(line[3:], 1)
        elif line.startswith("### "):
            doc.heading(line[4:], 2)
        elif image_match := image_pattern.match(line.strip()):
            caption = image_match.group(1).strip() or "Figure"
            img_path = Path(image_match.group(2).strip())
            if not img_path.is_absolute():
                img_path = (md_path.parent / img_path).resolve()
            doc.image(img_path, caption)
            if i + 1 < len(lines) and lines[i + 1].strip() == "":
                next_non_empty = i + 2
                if (
                    next_non_empty < len(lines)
                    and lines[next_non_empty].strip() in {f"*{caption}*", f"**{caption}**"}
                ):
                    i = next_non_empty
        elif equation_xml := formula_to_omml(line.strip()):
            doc.equation(equation_xml)
        elif equation_item := variable_explanation_to_omml(line.strip()):
            doc.equation_explanation(equation_item[0], equation_item[1])
        elif caption := is_caption_line(line.strip()):
            doc.caption(caption)
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                if not is_md_separator(lines[i]):
                    table_lines.append(lines[i])
                i += 1
            if table_lines:
                doc.table([split_md_table_row(item) for item in table_lines])
            continue
        elif line.strip():
            doc.p(re.sub(r"`([^`]+)`", r"\1", line.strip()))
        else:
            doc.p("")
        i += 1
    doc.heading("附录图" if zh else "Appendix Figures", 1)
    appendix_index = 1
    for folder, title in [
        (ROOT / "formal_cloud_results_formal_minimal_20260503_combined" / "figures", "Stage 1"),
        (ROOT / "formal_cloud_results" / "stage2_dsample04_round20_20260503" / "figures", "Stage 2"),
        (FIGURES, "Stage 3"),
        (STAGE4A / "figures", "Stage 4A"),
        (STAGE4B / "figures", "Stage 4B"),
        (STAGE4C / "figures", "Stage 4C"),
        (STAGE4_SEED / "figures", "Stage 4 seed summary"),
        (STAGE4_SEED / "seed43" / "figures", "Stage 4 seed43"),
        (STAGE4_SEED / "seed44" / "figures", "Stage 4 seed44"),
    ]:
        if folder.exists():
            for img in sorted(folder.glob("*.png")):
                doc.image(img, appendix_caption(title, img, appendix_index, zh=zh))
                appendix_index += 1
    doc.save(out_path)


def main() -> None:
    rows = read_csv_rows(STAGE3 / "summary.csv")
    stage4_sets = load_stage4_sets()
    seed_rows = load_seed_rows()
    s3_text = stage3_text(rows)
    (ROOT / "readme_result3.docx.txt").write_text(s3_text, encoding="utf-8-sig")
    build_stage3_docx(s3_text, rows)

    s4_text = stage4_text(rows, stage4_sets) if stage4_sets else ""
    if s4_text:
        (ROOT / "readme_result4.docx.txt").write_text(s4_text, encoding="utf-8-sig")
    stage4c_rows: list[dict[str, str]] = []
    for item in stage4_sets:
        if item.get("tag") == "Stage 4C":
            stage4c_rows = item["rows"]  # type: ignore[assignment]
            break
    seed_text = seed_robustness_text(stage4c_rows, seed_rows) if stage4c_rows and seed_rows else ""
    if seed_text:
        (ROOT / "readme_result4_seed_robustness.txt").write_text(seed_text, encoding="utf-8-sig")
        build_seed_docx(seed_text, stage4c_rows, seed_rows)

    s_text = summary_text(rows, stage4_sets, seed_rows)
    (ROOT / "Summary_of_the_three_stages_of_the_thesis.txt").write_text(s_text, encoding="utf-8-sig")
    (ROOT / "Summary_of_the_four_stages_of_the_thesis.txt").write_text(s_text, encoding="utf-8-sig")
    build_summary_docx(s_text, rows, stage4_sets, seed_rows)

    update_paper_md(rows, stage4_sets, seed_rows)
    markdown_to_docx(ROOT / "paper_draft.md", ROOT / "paper_draft.docx")

    outline = ROOT / "thesis_outline.md"
    outline_text = outline.read_text(encoding="utf-8", errors="replace")
    if "阶段三完整 9 组方向与 guard 消融" not in outline_text:
        outline_text += "\n\n## 2026-05-04 更新\n\n阶段三已更新为完整 9 组方向与 guard 消融实验。论文中应使用 `formal_cloud_results/stage3_full_direction_guard_20260504` 的 summary.csv、paper_table.md 和 figures，而不是旧的阶段三最小验证目录。\n"
    if "阶段四 2 x 2 模型和数据规模泛化矩阵" not in outline_text:
        outline_text += "\n\n## 2026-05-04 阶段四更新\n\n阶段四A/B/C结果已纳入论文：阶段四使用 2 x 2 模型和数据规模泛化矩阵，把阶段三作为小模型小数据格子，并补充小模型大数据、大模型小数据、大模型大数据三组完整 9 组实验。论文中应使用 `stage4a_qwen05b_dsample06_full9_20260504`、`stage4b_qwen15b_dsample04_full9_20260504` 和 `stage4_large_model_data_20260504` 的 summary.csv、paper_table.md 和 figures。\n"
    if "阶段四" in outline_text:
        outline.write_text(outline_text, encoding="utf-8")


if __name__ == "__main__":
    main()
