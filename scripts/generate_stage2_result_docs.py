from __future__ import annotations

import csv
import html
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = ROOT / "formal_cloud_results" / "stage2_dsample04_round20_20260503"
TXT_OUT = ROOT / "readme_result2.docx.txt"
DOCX_OUT = ROOT / "readme_result_2.docx.docx"


def read_summary() -> dict[str, dict[str, str]]:
    with (STAGE_DIR / "summary.csv").open(newline="", encoding="utf-8") as f:
        return {row["group"]: row for row in csv.DictReader(f)}


def f4(value: str | float | None) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.4f}"


def f2(value: str | float | None) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.2f}"


def delta(after: str, before: str) -> float:
    return float(after) - float(before)


def build_text(rows: dict[str, dict[str, str]]) -> str:
    fl = rows["fl"]
    fed = rows["fedhds"]
    fb = rows["forget_batch_hessian"]
    rg = rows["retained_guard_ref1_autoguard"]

    fb_forget_delta = delta(
        fb["forget_client_loss_after_unlearning"],
        fb["forget_client_loss_before_unlearning"],
    )
    fb_global_delta = delta(
        fb["global_loss_after_unlearning"],
        fb["global_loss_before_unlearning"],
    )
    rg_forget_delta = delta(
        rg["forget_client_loss_after_unlearning"],
        rg["forget_client_loss_before_unlearning"],
    )
    rg_global_delta = delta(
        rg["global_loss_after_unlearning"],
        rg["global_loss_before_unlearning"],
    )
    fed_vs_fl = float(fed["final_global_loss"]) - float(fl["final_global_loss"])
    guard_slack = float(rg["unlearn_global_loss_guard_max"]) - float(
        rg["global_loss_after_unlearning"]
    )

    lines = [
        "第二阶段云端正式实验结果整理",
        "",
        "一、当前状态",
        "",
        "第二阶段已经完成。本阶段在第一阶段最小闭环的基础上扩大训练设置，将 rounds 从 10 提高到 20，将 data_sample 从 0.2 提高到 0.4，目标是验证 FL/FedHDS 训练底座、forget-batch Hessian unlearning、以及 retained-set Hessian + auto guard 在更大数据采样和更长联邦训练轮数下的表现。",
        "",
        "本阶段最终结果已经下载并解压到本地：",
        "",
        "1. 主结果目录：",
        r"   F:\FedHDS-zhaoge\formal_cloud_results\stage2_dsample04_round20_20260503",
        "",
        "2. 云端压缩包备份：",
        r"   F:\FedHDS-zhaoge\stage2_dsample04_round20_20260503.tar.gz",
        "",
        "3. 关键文件：",
        "   - summary.csv",
        "   - paper_table.csv",
        "   - paper_table.md",
        "   - figures/*.png",
        "   - figures/*.pdf",
        "   - nohup.out",
        "   - logs/*.log",
        "   - 各组 final_results.json",
        "",
        "二、第二阶段实验设置",
        "",
        "本阶段使用 AutoDL/SeetaCloud 单卡 RTX 5090 32GB 云服务器，模型为 Qwen2-0.5B，本地云端路径为：",
        "",
        "/root/autodl-tmp/models/Qwen2-0.5B-ms",
        "",
        "数据集为 Dolly，本地云端路径为：",
        "",
        "/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl",
        "",
        "第二阶段正式配置如下：",
        "",
        "- num_clients = 20",
        "- client fraction k = 0.2",
        "- rounds = 20",
        "- local_step = 2",
        "- batch_size = 1",
        "- max_length = 64",
        "- data_sample = 0.4",
        "- iid = 0",
        "- forget_client_idx = 0",
        "- lissa_depth = 1",
        "- lissa_damping = 0.01",
        "- unlearn_grad_sample_size = 8",
        "- unlearn_eta = 0.01",
        "",
        "本阶段正式四组为：",
        "",
        "1. FL",
        "2. FedHDS",
        "3. FedHDS + forget-batch Hessian unlearning",
        "4. FedHDS + retained-set Hessian + auto global guard",
        "",
        "三、正式主结果摘要",
        "",
        f"FL 的 final global loss 为 {f4(fl['final_global_loss'])}。",
        "",
        f"FedHDS 的 final global loss 为 {f4(fed['final_global_loss'])}。相较 FL，FedHDS 低 {abs(fed_vs_fl):.4f}，说明在第二阶段更长训练和更大 data_sample 设置下，FedHDS 的全局效用优于普通 FL。",
        "",
        "FedHDS + forget-batch Hessian unlearning 的结果为：",
        "",
        f"- forget client loss: {f4(fb['forget_client_loss_before_unlearning'])} -> {f4(fb['forget_client_loss_after_unlearning'])}，变化 {fb_forget_delta:+.4f}",
        f"- global loss: {f4(fb['global_loss_before_unlearning'])} -> {f4(fb['global_loss_after_unlearning'])}，变化 {fb_global_delta:+.4f}",
        f"- actual update L2: {f4(fb['actual_param_delta_l2_norm'])}",
        f"- HVP L2 norm: {f2(fb['last_hvp_l2_norm'])}",
        f"- steps accepted/attempted: {fb['unlearn_steps_accepted']}/{fb['unlearn_steps_attempted']}",
        f"- unlearning time: {f2(fb['unlearning_time_sec'])} 秒",
        "",
        "该组 forget loss 上升，说明 forget-batch Hessian baseline 在当前设置下产生了更明确的遗忘方向；但 global loss 同时明显上升，说明它以全局效用损伤为代价换取遗忘效果。",
        "",
        "FedHDS + retained-set Hessian + auto guard 的结果为：",
        "",
        f"- hessian_ref_per_client = 1",
        f"- retained reference set size = {rg['retained_reference_set_size']}",
        f"- global_guard = {f4(rg['unlearn_global_loss_guard_max'])}",
        f"- forget client loss: {f4(rg['forget_client_loss_before_unlearning'])} -> {f4(rg['forget_client_loss_after_unlearning'])}，变化 {rg_forget_delta:+.4f}",
        f"- global loss: {f4(rg['global_loss_before_unlearning'])} -> {f4(rg['global_loss_after_unlearning'])}，变化 {rg_global_delta:+.4f}",
        f"- actual update L2: {f4(rg['actual_param_delta_l2_norm'])}",
        f"- HVP L2 norm: {f2(rg['last_hvp_l2_norm'])}",
        f"- steps accepted/attempted: {rg['unlearn_steps_accepted']}/{rg['unlearn_steps_attempted']}",
        f"- unlearning time: {f2(rg['unlearning_time_sec'])} 秒",
        f"- guard slack: {guard_slack:.4f}",
        "",
        "该组 5/5 个小步全部被接受，最终 global loss 仍低于自动 guard 阈值，说明 stepped guard 成功限制了全局效用漂移。但 forget client loss 从 2.2581 降到 2.2329，遗忘方向不理想，不能把这一组写成遗忘效果成功增强。",
        "",
        "四、论文表格",
        "",
        "| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
        f"| fedhds | eta=1.0000 | none | - | {f4(fed['final_global_loss'])} | - | - | - | utility baseline |",
        f"| fl | eta=1.0000 | none | - | {f4(fl['final_global_loss'])} | - | - | - | utility baseline |",
        f"| forget_batch_hessian | eta=0.0100 | forget-batch | {f4(fb['forget_client_loss_before_unlearning'])} -> {f4(fb['forget_client_loss_after_unlearning'])} | {f4(fb['final_global_loss'])} | {f4(fb['actual_param_delta_l2_norm'])} | {fb['unlearn_steps_accepted']}/{fb['unlearn_steps_attempted']} | {f2(fb['unlearning_time_sec'])} | forget loss increased |",
        f"| retained_guard_ref1_autoguard | eta=0.0100, max_norm=0.300, steps=5, global_guard={f4(rg['unlearn_global_loss_guard_max'])} | retained-set | {f4(rg['forget_client_loss_before_unlearning'])} -> {f4(rg['forget_client_loss_after_unlearning'])} | {f4(rg['final_global_loss'])} | {f4(rg['actual_param_delta_l2_norm'])} | {rg['unlearn_steps_accepted']}/{rg['unlearn_steps_attempted']} | {f2(rg['unlearning_time_sec'])} | utility preserved, weak forgetting |",
        "",
        "五、结果解读",
        "",
        "第二阶段最重要的变化是 FedHDS 在更大训练设置下成为更好的 utility baseline。第一阶段中 FedHDS 的 final global loss 略高于 FL；第二阶段中 FedHDS final global loss 为 1.5250，FL 为 1.5441，FedHDS 低约 0.0192。这说明当训练轮数和数据采样增加后，FedHDS 的聚类/过滤机制没有损伤全局训练，反而带来了更好的最终 global loss。",
        "",
        "forget-batch Hessian baseline 在遗忘指标上更强。forget loss 从 2.2581 上升到 2.2866，提升约 0.0285。但它把 global loss 从 1.5250 推高到 1.6257，恶化约 0.1007。该结果适合在论文中作为 trade-off baseline：遗忘方向明确，但 utility cost 较大。",
        "",
        "retained-set Hessian + auto guard 的主要价值是 utility control。该组把 global loss 控制在 1.5482，距离 FedHDS unlearning 前的 1.5250 只上升约 0.0232，也低于自动 guard 阈值 1.5750。但它没有实现有效遗忘，forget loss 反而下降约 0.0252。该组应写作“guarded retained-set update can preserve utility under a strict global-loss guard, but the current direction is too conservative or misaligned for strong forgetting”。",
        "",
        "从计算成本看，retained guard 仍明显更慢。forget-batch Hessian 约 1.64 秒完成，而 retained guard 约 131.13 秒。原因是 retained reference set curvature、5-step guard 评估以及每一步的 global loss 检查都增加了计算量。",
        "",
        "从更新幅度看，forget-batch Hessian 的 actual update L2 为 0.5077，retained guard 为 0.2486。retained guard 的更新更小，并且启用了 max_update_norm=0.3，虽然本次 clipping 未实际触发，但配置仍限制了可能的大步更新。",
        "",
        "六、retained guard 的优点",
        "",
        "第一，retained guard 在 data_sample=0.4、rounds=20 的第二阶段正式配置中稳定完成，没有 OOM 或中断。",
        "",
        "第二，auto global guard 能根据 FedHDS baseline loss 自动设定 guard=FedHDS loss+0.05，避免沿用固定阈值造成误拒绝或误接受。",
        "",
        "第三，global utility 明显更稳。forget-batch Hessian 的 global loss 恶化到 1.6257，而 retained guard 的 final global loss 为 1.5482，接近 FL 和 FedHDS baseline。",
        "",
        "第四，5/5 个小步均被接受，说明 stepped guard 在该阈值下给出了连续可记录、可解释的更新轨迹。",
        "",
        "七、retained guard 的缺点",
        "",
        "第一，当前第二阶段配置下遗忘效果不成立。forget loss 从 2.2581 降到 2.2329，方向与目标相反。",
        "",
        "第二，计算成本高。retained guard 约 131.13 秒，约为 forget-batch Hessian 的 80 倍。",
        "",
        "第三，方法对 Hessian reference、eta、guard 和 update norm 的组合较敏感。当前配置强调 utility preservation，但牺牲了 forgetting strength。",
        "",
        "第四，当前结果不能证明 retained-set Hessian 全面优于 forget-batch Hessian，只能证明它在全局效用约束方面更稳。",
        "",
        "八、论文写法建议",
        "",
        "第二阶段适合写成更清晰的 trade-off 实验：",
        "",
        "1. FedHDS 在更大训练设置下取得更低 final global loss，可作为主要 utility baseline。",
        "2. forget-batch Hessian 提升 forget loss，但明显损伤 global utility。",
        "3. retained-set Hessian + auto guard 显著控制 global utility，但当前 forgetting strength 不足。",
        "4. 因此本文方法更准确的定位是“utility-preserving guarded unlearning framework”，而不是“遗忘强度最强的 unlearning 方法”。",
        "",
        "推荐论文表述：",
        "",
        "Compared with the forget-batch Hessian baseline, the retained-set guarded update substantially reduces the global-loss increase, but its forgetting effect is weaker under the current stage-2 setting. This suggests that retained-set curvature and stepped guards are effective for utility preservation, while stronger or better-aligned forgetting objectives are needed for larger-scale improvements.",
        "",
        "九、图文件说明",
        "",
        "第二阶段已经生成 6 组 PNG/PDF 图，均已放入 Word 文档：",
        "",
        "1. global_loss_final：比较 FL、FedHDS、forget-batch Hessian、retained guard 的 final global loss。",
        "2. forget_loss_before_after：比较两种 unlearning 方法前后的 forget client loss。",
        "3. update_norm_vs_global_loss：展示 update L2 与 global loss 之间的关系。",
        "4. update_norm_vs_forget_gain：展示 update L2 与 forgetting gain 之间的关系。",
        "5. guard_steps：展示 retained guard 的 step 接受情况。",
        "6. unlearning_time：比较 forget-batch Hessian 与 retained guard 的 unlearning 时间。",
        "",
        "十、结论边界",
        "",
        "第二阶段不能写成 retained guard 在遗忘效果上超过 forget-batch baseline。更稳妥的结论是：",
        "",
        "1. 扩大 data_sample 和 rounds 后，FedHDS 的 global utility 优于 FL。",
        "2. forget-batch Hessian 可以提升 forget loss，但会带来明显 global utility 损伤。",
        "3. retained-set Hessian + auto guard 能显著约束 global loss，并在第二阶段稳定运行。",
        "4. retained guard 当前遗忘方向较弱，甚至出现 forget loss 下降，因此需要作为 limitation 或下一步优化方向说明。",
        "5. 论文贡献应聚焦在 retained-set curvature、global norm clipping、stepped trust-region guard 对 utility preservation 和可控更新流程的贡献。",
        "",
        "十一、下一步建议",
        "",
        "建议先把第二阶段作为正式结果写入论文主实验或补充实验。若需要进一步增强遗忘效果，可以在第三阶段优先做小范围参数验证：",
        "",
        "- 检查 Hessian update 方向符号与 forget objective 是否一致。",
        "- 尝试更大的 unlearn_eta 或更高 max_update_norm，但保留 auto global guard。",
        "- 尝试 retained reference set size 从 19 增加到 38 或 57，观察 OOM 和 forgetting gain。",
        "- 在保留 guard 的前提下加入 forget-loss minimum-gain check，避免 utility 被保护但 forgetting 方向无效。",
        "",
    ]
    return "\n".join(lines)


def xml_text(text: str) -> str:
    return html.escape(text, quote=False)


def paragraph(text: str, style: str | None = None, bold: bool = False) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    b = "<w:b/>" if bold else ""
    return (
        f"<w:p>{ppr}<w:r><w:rPr>{b}</w:rPr>"
        f'<w:t xml:space="preserve">{xml_text(text)}</w:t></w:r></w:p>'
    )


def table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    width = max(900, int(9000 / col_count))
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for _ in range(col_count))
    trs = []
    for i, row in enumerate(rows):
        cells = []
        for cell in row:
            bold = "<w:b/>" if i == 0 else ""
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
                f"<w:p><w:r><w:rPr>{bold}</w:rPr>"
                f'<w:t xml:space="preserve">{xml_text(cell)}</w:t></w:r></w:p></w:tc>'
            )
        trs.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        "<w:tbl>"
        '<w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblLook w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" '
        'w:noHBand="0" w:noVBand="1"/></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>"
    )


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        sig = f.read(24)
    if sig[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", sig[16:24])


def image_paragraph(rel_id: str, name: str, path: Path, docpr_id: int) -> str:
    px_w, px_h = png_size(path)
    width_in = min(5.9, px_w / 120)
    height_in = width_in * px_h / px_w
    if height_in > 4.4:
        height_in = 4.4
        width_in = height_in * px_w / px_h
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)
    return f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{docpr_id}" name="{xml_text(name)}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="{docpr_id}" name="{xml_text(path.name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def document_xml(body_items: list[str]) -> str:
    body = "\n".join(body_items)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1200" w:right="1000" w:bottom="1200" w:left="1000" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Calibri" w:eastAsia="SimSun" w:hAnsi="Calibri"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:pPr><w:jc w:val="center"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Calibri" w:eastAsia="SimHei" w:hAnsi="Calibri"/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Calibri" w:eastAsia="SimHei" w:hAnsi="Calibri"/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Calibri" w:eastAsia="SimHei" w:hAnsi="Calibri"/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr><w:tblBorders>
      <w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>
    </w:tblBorders></w:tblPr>
  </w:style>
</w:styles>
"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def doc_rels_xml(image_names: list[str]) -> str:
    rels = [
        '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    ]
    for i, image_name in enumerate(image_names, start=1):
        rels.append(
            f'<Relationship Id="rIdImg{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{xml_text(image_name)}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>\n"
    )


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>第二阶段云端正式实验结果整理</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex OOXML Writer</Application>
</Properties>
"""


def build_docx(rows: dict[str, dict[str, str]]) -> None:
    fl = rows["fl"]
    fed = rows["fedhds"]
    fb = rows["forget_batch_hessian"]
    rg = rows["retained_guard_ref1_autoguard"]

    paper_rows = [
        [
            "Source",
            "Setting",
            "Hessian",
            "Forget loss",
            "Final global loss",
            "Update L2",
            "Steps",
            "Time (s)",
            "Note",
        ],
        ["fedhds", "eta=1.0000", "none", "-", f4(fed["final_global_loss"]), "-", "-", "-", "utility baseline"],
        ["fl", "eta=1.0000", "none", "-", f4(fl["final_global_loss"]), "-", "-", "-", "utility baseline"],
        [
            "forget_batch_hessian",
            "eta=0.0100",
            "forget-batch",
            f"{f4(fb['forget_client_loss_before_unlearning'])} -> {f4(fb['forget_client_loss_after_unlearning'])}",
            f4(fb["final_global_loss"]),
            f4(fb["actual_param_delta_l2_norm"]),
            f"{fb['unlearn_steps_accepted']}/{fb['unlearn_steps_attempted']}",
            f2(fb["unlearning_time_sec"]),
            "forget loss increased",
        ],
        [
            "retained_guard_ref1_autoguard",
            f"eta=0.0100, max_norm=0.300, steps=5, global_guard={f4(rg['unlearn_global_loss_guard_max'])}",
            "retained-set",
            f"{f4(rg['forget_client_loss_before_unlearning'])} -> {f4(rg['forget_client_loss_after_unlearning'])}",
            f4(rg["final_global_loss"]),
            f4(rg["actual_param_delta_l2_norm"]),
            f"{rg['unlearn_steps_accepted']}/{rg['unlearn_steps_attempted']}",
            f2(rg["unlearning_time_sec"]),
            "utility preserved, weak forgetting",
        ],
    ]

    detail_rows = [
        ["Group", "Round-2 global", "Final global", "Forget before", "Forget after", "Hessian", "Ref size", "HVP L2"],
    ]
    for key in ["fl", "fedhds", "forget_batch_hessian", "retained_guard_ref1_autoguard"]:
        row = rows[key]
        detail_rows.append(
            [
                key,
                f4(row["round2_global_loss"]),
                f4(row["final_global_loss"]),
                f4(row.get("forget_client_loss_before_unlearning")),
                f4(row.get("forget_client_loss_after_unlearning")),
                row.get("hessian_mode") or "none",
                row.get("retained_reference_set_size") or "-",
                f2(row.get("last_hvp_l2_norm")),
            ]
        )

    figure_specs = [
        ("global_loss_final.png", "图 1：Final global loss 对比"),
        ("forget_loss_before_after.png", "图 2：Forget client loss 前后对比"),
        ("update_norm_vs_global_loss.png", "图 3：Update L2 与 final global loss"),
        ("update_norm_vs_forget_gain.png", "图 4：Update L2 与 forgetting gain"),
        ("guard_steps.png", "图 5：Retained guard step 接受情况"),
        ("unlearning_time.png", "图 6：Unlearning time 对比"),
    ]
    figure_paths = [STAGE_DIR / "figures" / name for name, _ in figure_specs]
    image_names = [f"image{i}.png" for i in range(1, len(figure_paths) + 1)]

    body: list[str] = [
        paragraph("第二阶段云端正式实验结果整理", "Title"),
        paragraph("一、实验状态", "Heading1"),
        paragraph("第二阶段已经完成，结果已下载并解压到本地 formal_cloud_results/stage2_dsample04_round20_20260503。该阶段在第一阶段基础上扩大 rounds=20、data_sample=0.4，用于观察更大训练设置下 utility、forgetting 和 guarded unlearning 的关系。"),
        paragraph("二、实验设置", "Heading1"),
        paragraph("模型：Qwen2-0.5B；数据集：Dolly；num_clients=20；k=0.2；rounds=20；local_step=2；batch_size=1；max_length=64；data_sample=0.4；iid=0；forget_client_idx=0；lissa_depth=1；lissa_damping=0.01；unlearn_eta=0.01。"),
        paragraph("三、论文主表", "Heading1"),
        table(paper_rows),
        paragraph("四、详细指标表", "Heading1"),
        table(detail_rows),
        paragraph("五、结果解读", "Heading1"),
        paragraph("FedHDS 的 final global loss 为 1.5250，低于 FL 的 1.5441，说明在第二阶段更大训练设置下 FedHDS 成为更好的 utility baseline。"),
        paragraph("forget-batch Hessian 的 forget loss 从 2.2581 上升到 2.2866，遗忘方向更明确；但 global loss 从 1.5250 上升到 1.6257，utility 损伤较大。"),
        paragraph("retained-set Hessian + auto guard 的 global loss 控制在 1.5482，低于 guard=1.5750，说明 utility control 有效；但 forget loss 从 2.2581 降到 2.2329，当前配置下遗忘方向不足。"),
        paragraph("因此第二阶段更适合写成 trade-off 结果：forget-batch baseline 提供更强 forgetting signal，retained guard 提供更稳定的 utility preservation。"),
        paragraph("六、论文图", "Heading1"),
    ]

    for idx, ((_, caption), path) in enumerate(zip(figure_specs, figure_paths), start=1):
        body.append(paragraph(caption, "Heading2"))
        body.append(image_paragraph(f"rIdImg{idx}", caption, path, idx))

    body.extend(
        [
            paragraph("七、结论边界", "Heading1"),
            paragraph("不能把第二阶段写成 retained guard 在遗忘效果上优于 forget-batch Hessian。更稳妥的结论是：FedHDS 在扩大训练设置下提升 global utility；forget-batch Hessian 提供更强遗忘但损伤 utility；retained-set Hessian + auto guard 能显著约束 global loss，但当前 forgetting strength 不足。"),
            paragraph("八、推荐论文表述", "Heading1"),
            paragraph("Compared with the forget-batch Hessian baseline, the retained-set guarded update substantially reduces the global-loss increase, but its forgetting effect is weaker under the current stage-2 setting. This suggests that retained-set curvature and stepped guards are effective for utility preservation, while stronger or better-aligned forgetting objectives are needed for larger-scale improvements."),
        ]
    )

    with zipfile.ZipFile(DOCX_OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml())
        z.writestr("_rels/.rels", root_rels_xml())
        z.writestr("word/document.xml", document_xml(body))
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/_rels/document.xml.rels", doc_rels_xml(image_names))
        z.writestr("docProps/core.xml", core_xml())
        z.writestr("docProps/app.xml", app_xml())
        for image_name, path in zip(image_names, figure_paths):
            z.write(path, f"word/media/{image_name}")


def main() -> None:
    rows = read_summary()
    TXT_OUT.write_text(build_text(rows), encoding="utf-8-sig")
    build_docx(rows)
    print(f"Wrote {TXT_OUT}")
    print(f"Wrote {DOCX_OUT}")


if __name__ == "__main__":
    main()
