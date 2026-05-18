from __future__ import annotations

import csv
import html
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = ROOT / "formal_cloud_results_formal_minimal_20260503_combined"
TXT_OUT = ROOT / "readme_result1.txt"
DOCX_OUT = ROOT / "readme_result_1.docx"


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


def step_count(value: str | None) -> str:
    if value in (None, ""):
        return "-"
    return str(int(float(value)))


def delta(after: str, before: str) -> float:
    return float(after) - float(before)


def build_text(rows: dict[str, dict[str, str]]) -> str:
    fl = rows["fl"]
    fed = rows["fedhds"]
    fb = rows["forget_batch_hessian"]
    rg = rows["retained_guard_ref1_guard170"]

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

    lines = [
        "第一阶段云端正式实验结果整理",
        "",
        "一、当前状态",
        "",
        "第一阶段已经完成。本阶段目标不是继续做参数 sweep，而是验证“小论文最小方法方案”在云端服务器上能否形成完整闭环：模型可加载、数据可读取、FL/FedHDS 可训练、forget-batch Hessian unlearning 可执行、retained-set Hessian + stepped guard 可执行，并且能够自动生成 summary、paper table 和 figures。",
        "",
        "本阶段最终结果已经下载到本地：",
        "",
        "1. 主结果目录：",
        r"   F:\FedHDS-zhaoge\formal_cloud_results_formal_minimal_20260503_combined",
        "",
        "2. 完整云端归档目录：",
        r"   F:\FedHDS-zhaoge\formal_cloud_results_autodl_20260503",
        "",
        "3. 关键文件：",
        "   - summary.csv",
        "   - paper_table.csv",
        "   - paper_table.md",
        "   - figures/*.png",
        "   - figures/*.pdf",
        "   - 各组 final_results.json",
        "   - 各组运行日志",
        "",
        "二、第一阶段实验设置",
        "",
        "本阶段使用 AutoDL 单卡 RTX 5090 32GB 云服务器，模型为 Qwen2-0.5B，本地路径为：",
        "",
        "/root/autodl-tmp/models/Qwen2-0.5B-ms",
        "",
        "数据集为 Dolly，本地路径为：",
        "",
        "/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl",
        "",
        "正式实验采用第一阶段最小云端配置：",
        "",
        "- num_clients = 20",
        "- client fraction k = 0.2",
        "- rounds = 10",
        "- local_step = 2",
        "- batch_size = 1",
        "- max_length = 64",
        "- data_sample = 0.2",
        "- iid = 0",
        "- forget_client_idx = 0",
        "- lissa_depth = 1",
        "- lissa_damping = 0.01",
        "- unlearn_eta = 0.01",
        "",
        "本阶段正式四组为：",
        "",
        "1. FL",
        "2. FedHDS",
        "3. FedHDS + forget-batch Hessian unlearning",
        "4. FedHDS + retained-set Hessian + stepped guard",
        "",
        "三、正式主结果摘要",
        "",
        f"FL 的 final global loss 为 {f4(fl['final_global_loss'])}。",
        "",
        f"FedHDS 的 final global loss 为 {f4(fed['final_global_loss'])}。",
        "",
        "FedHDS + forget-batch Hessian unlearning 的结果为：",
        "",
        f"- forget client loss: {f4(fb['forget_client_loss_before_unlearning'])} -> {f4(fb['forget_client_loss_after_unlearning'])}，变化 {fb_forget_delta:+.4f}",
        f"- global loss: {f4(fb['global_loss_before_unlearning'])} -> {f4(fb['global_loss_after_unlearning'])}，变化 {fb_global_delta:+.4f}",
        f"- actual update L2: {f4(fb['actual_param_delta_l2_norm'])}",
        f"- HVP L2 norm: {f2(fb['last_hvp_l2_norm'])}",
        f"- steps accepted/attempted: {step_count(fb['unlearn_steps_accepted'])}/{step_count(fb['unlearn_steps_attempted'])}",
        f"- unlearning time: {f2(fb['unlearning_time_sec'])} 秒",
        "",
        "FedHDS + retained-set Hessian + stepped guard 的最终可用配置为：",
        "",
        "- hessian_ref_per_client = 1",
        f"- retained reference set size = {step_count(rg['retained_reference_set_size'])}",
        "- unlearn_grad_sample_size = 8",
        "- max_update_norm = 0.3",
        "- unlearn_num_steps = 5",
        f"- global_guard = {f4(rg['unlearn_global_loss_guard_max'])}",
        "",
        f"在该配置下，forget client loss 从 {f4(rg['forget_client_loss_before_unlearning'])} 上升到 {f4(rg['forget_client_loss_after_unlearning'])}，变化 {rg_forget_delta:+.4f}；global loss 从 {f4(rg['global_loss_before_unlearning'])} 变为 {f4(rg['global_loss_after_unlearning'])}，变化 {rg_global_delta:+.4f}；actual update L2 为 {f4(rg['actual_param_delta_l2_norm'])}，5/5 个小步被接受，unlearning 耗时约 {f2(rg['unlearning_time_sec'])} 秒。",
        "",
        "四、论文表格",
        "",
        "| Source | Setting | Hessian | Forget loss | Final global loss | Update L2 | Steps | Time (s) | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
        f"| fl | eta=1.0000 | none | - | {f4(fl['final_global_loss'])} | - | - | - | utility baseline |",
        f"| fedhds | eta=1.0000 | none | - | {f4(fed['final_global_loss'])} | - | - | - | training backbone |",
        f"| forget_batch_hessian | eta=0.0100 | forget-batch | {f4(fb['forget_client_loss_before_unlearning'])} -> {f4(fb['forget_client_loss_after_unlearning'])} | {f4(fb['final_global_loss'])} | {f4(fb['actual_param_delta_l2_norm'])} | {step_count(fb['unlearn_steps_accepted'])}/{step_count(fb['unlearn_steps_attempted'])} | {f2(fb['unlearning_time_sec'])} | forget loss increased, global utility slightly worse |",
        f"| retained_guard_ref1_guard170 | eta=0.0100, max_norm=0.300, steps=5, global_guard={f4(rg['unlearn_global_loss_guard_max'])} | retained-set | {f4(rg['forget_client_loss_before_unlearning'])} -> {f4(rg['forget_client_loss_after_unlearning'])} | {f4(rg['final_global_loss'])} | {f4(rg['actual_param_delta_l2_norm'])} | {step_count(rg['unlearn_steps_accepted'])}/{step_count(rg['unlearn_steps_attempted'])} | {f2(rg['unlearning_time_sec'])} | forget loss increased slightly, global utility stable |",
        "",
        "五、结果解读",
        "",
        "第一阶段云端结果说明，当前代码链路已经能够在真实云端 GPU 环境中完整运行。FL、FedHDS、forget-batch Hessian unlearning、retained-set Hessian + stepped guard 均完成了训练或 unlearning 流程，并生成了可用于论文整理的表格和图。",
        "",
        "从遗忘效果看，forget-batch Hessian baseline 的 forget loss 提升幅度略大：2.3381 到 2.3489，提升约 0.0108。retained-set guard 的 forget loss 从 2.3381 到 2.3460，提升约 0.0079，略弱于 forget-batch baseline。",
        "",
        "从 global utility 看，forget-batch Hessian 会使 global loss 从 1.6657 上升到 1.6867，说明有轻微 utility 损伤。retained-set guard 的 global loss 从 1.6657 变为 1.6615，没有恶化，反而略低。这说明 stepped guard 在当前正式设置下确实起到了约束 global utility 的作用。",
        "",
        "从计算成本看，retained-set guard 明显更慢。forget-batch Hessian unlearning 耗时约 1.44 秒，而 retained-set guard 耗时约 71.13 秒。这说明 retained-set curvature 和 stepped guard 带来了更强的计算开销。",
        "",
        "六、retained guard 的优点",
        "",
        "第一，retained guard 能够在云端正式配置中稳定运行。原始 retained-set Hessian 使用 hessian_ref_per_client=4 时在 RTX 5090 32GB 上 OOM；调整为 hessian_ref_per_client=1 后成功完成。",
        "",
        "第二，stepped guard 能够把 retained-set Hessian 更新限制在可控范围内。最终配置下 5/5 个小步被接受，actual update L2 为 0.0427，说明更新幅度可被清楚记录和约束。",
        "",
        "第三，retained guard 对 global utility 更友好。在当前正式结果中，global loss 没有像 forget-batch baseline 那样上升，而是从 1.6657 略降到 1.6615。",
        "",
        "第四，retained-set Hessian 的方法逻辑更适合论文叙述。forget client gradient 定义“要移除什么”，retained reference set curvature 定义“系统保留任务附近的曲率结构”，再用 clipping 和 stepped guard 控制更新幅度。",
        "",
        "七、retained guard 的缺点",
        "",
        "第一，当前正式配置下 forgetting 提升较弱。forget loss 仅从 2.3381 上升到 2.3460，提升约 0.0079，不能宣称遗忘效果显著强于 baseline。",
        "",
        "第二，计算成本明显更高。retained guard 耗时约 71.13 秒，远高于 forget-batch Hessian 的 1.44 秒。",
        "",
        "第三，显存敏感。hessian_ref_per_client=4 会触发 CUDA OOM，说明 retained-set Hessian 的二阶计算在 32GB 显存上仍需要谨慎设置 reference size。",
        "",
        "第四，guard 阈值依赖当前实验 loss 尺度。本地小实验中 global_guard=0.8 合理，但云端正式实验中 unlearning 前 global loss 已经是 1.6657，继续使用 0.8 会导致所有小步被拒绝。最终改为 global_guard=1.7 后才得到可用结果。",
        "",
        "八、结论边界",
        "",
        "当前结果不能写成 retained-set Hessian 全面优于 forget-batch Hessian。更稳妥的结论是：",
        "",
        "1. retained-set curvature approximation 在云端正式实验中可运行。",
        "2. global norm clipping 和 stepped trust-region guard 能够把 retained-set Hessian 更新变成可截停、可回滚、可按 utility 阈值控制的过程。",
        "3. 在当前正式最小配置下，retained guard 能在不恶化 global loss 的情况下产生小幅 forget loss 提升。",
        "4. 但 retained guard 的 forgetting 提升弱于 forget-batch baseline，且计算成本明显更高。",
        "5. 因此该方法当前更适合作为“稳定性与可控性改进”来写，而不是作为“遗忘强度全面领先”的方法来写。",
        "",
        "九、论文写法建议",
        "",
        "方法部分可以强调三点：",
        "",
        "1. Retained-set curvature approximation：用 retained clients 构建参考集，估计系统保留任务附近的 Hessian 曲率。",
        "2. Global update norm clipping：在写回参数前约束整体 update L2，避免 retained Hessian 方向导致不可控大更新。",
        "3. Stepped trust-region guard：把 clipped update 切成多个小步，每步后检查 global loss，超过阈值则回滚并停止。",
        "",
        "实验部分可以采用以下消融叙述：",
        "",
        "1. FL vs FedHDS：说明训练底座和 global utility。",
        "2. forget-batch Hessian：作为 unlearning baseline，说明可产生 forget loss 提升，但有 global utility 代价。",
        "3. retained-set Hessian + guard：说明 retained curvature 方向在 guard 约束下可运行，且 global utility 更稳。",
        "4. OOM 与 guard 阈值修正：作为工程边界或 limitation 说明，不作为主方法贡献。",
        "",
        "十、图文件说明",
        "",
        "第一阶段已经生成 6 组 PNG/PDF 图，均已放入 Word 文档：",
        "",
        "1. global_loss_final：比较 FL、FedHDS、forget-batch Hessian、retained guard 的 final global loss。",
        "2. forget_loss_before_after：比较两种 unlearning 方法前后的 forget client loss。",
        "3. update_norm_vs_global_loss：展示 update L2 与 global loss 之间的关系。",
        "4. update_norm_vs_forget_gain：展示 update L2 与 forgetting gain 之间的关系。",
        "5. guard_steps：展示 retained guard 的 step 接受情况。",
        "6. unlearning_time：比较 forget-batch Hessian 与 retained guard 的 unlearning 时间。",
        "",
        "十一、下一步建议",
        "",
        "短期建议先进入写作整理阶段，固化第一阶段主表、方法段落、实验设置段落、结果分析和 limitation。除非论文需要更强实验支撑，否则不建议立刻继续做大量相近参数 sweep。",
        "",
        "如果要进入第二阶段，建议仍使用 Qwen2-0.5B，优先扩大 data_sample 或 rounds，并保留第一阶段已经验证过的 retained guard 安全配置：hessian_ref_per_client=1、max_update_norm=0.3、steps=5、global_guard 按训练后 global loss 尺度设置，而不是固定沿用本地 0.8。",
        "",
        "更大模型如 Qwen2.5-1.5B 或 Qwen2-1.5B 可作为第三阶段补充验证，不建议直接上 7B。",
        "",
    ]
    return "\n".join(lines)


def xml_text(text: str | float | int | None) -> str:
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def paragraph(text: str, style: str | None = None, bold: bool = False) -> str:
    pstyle = f'<w:pStyle w:val="{style}"/>' if style else ""
    b = "<w:b/>" if bold else ""
    ppr = f"<w:pPr>{pstyle}</w:pPr>" if pstyle else ""
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
  <dc:title>第一阶段云端正式实验结果整理</dc:title>
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
    rg = rows["retained_guard_ref1_guard170"]

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
        ["fl", "eta=1.0000", "none", "-", f4(fl["final_global_loss"]), "-", "-", "-", "utility baseline"],
        ["fedhds", "eta=1.0000", "none", "-", f4(fed["final_global_loss"]), "-", "-", "-", "training backbone"],
        [
            "forget_batch_hessian",
            "eta=0.0100",
            "forget-batch",
            f"{f4(fb['forget_client_loss_before_unlearning'])} -> {f4(fb['forget_client_loss_after_unlearning'])}",
            f4(fb["final_global_loss"]),
            f4(fb["actual_param_delta_l2_norm"]),
            f"{step_count(fb['unlearn_steps_accepted'])}/{step_count(fb['unlearn_steps_attempted'])}",
            f2(fb["unlearning_time_sec"]),
            "forget loss increased, global utility slightly worse",
        ],
        [
            "retained_guard_ref1_guard170",
            f"eta=0.0100, max_norm=0.300, steps=5, global_guard={f4(rg['unlearn_global_loss_guard_max'])}",
            "retained-set",
            f"{f4(rg['forget_client_loss_before_unlearning'])} -> {f4(rg['forget_client_loss_after_unlearning'])}",
            f4(rg["final_global_loss"]),
            f4(rg["actual_param_delta_l2_norm"]),
            f"{step_count(rg['unlearn_steps_accepted'])}/{step_count(rg['unlearn_steps_attempted'])}",
            f2(rg["unlearning_time_sec"]),
            "forget loss increased slightly, global utility stable",
        ],
    ]

    detail_rows = [
        ["Group", "Round-2 global", "Final global", "Forget before", "Forget after", "Hessian", "Ref size", "HVP L2"],
    ]
    for key in ["fl", "fedhds", "forget_batch_hessian", "retained_guard_ref1_guard170"]:
        row = rows[key]
        detail_rows.append(
            [
                key,
                f4(row["round2_global_loss"]),
                f4(row["final_global_loss"]),
                f4(row.get("forget_client_loss_before_unlearning")),
                f4(row.get("forget_client_loss_after_unlearning")),
                row.get("hessian_mode") or "none",
                step_count(row.get("retained_reference_set_size")) if row.get("retained_reference_set_size") else "-",
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
        paragraph("第一阶段云端正式实验结果整理", "Title"),
        paragraph("一、实验状态", "Heading1"),
        paragraph("第一阶段已经完成，结果已下载并解压到本地 formal_cloud_results_formal_minimal_20260503_combined。该阶段用于验证最小方法方案能否在云端服务器上完整闭环，包括 FL、FedHDS、forget-batch Hessian unlearning、retained-set Hessian + stepped guard，以及 summary、paper table 和 figures 的生成。"),
        paragraph("二、实验设置", "Heading1"),
        paragraph("模型：Qwen2-0.5B；数据集：Dolly；num_clients=20；k=0.2；rounds=10；local_step=2；batch_size=1；max_length=64；data_sample=0.2；iid=0；forget_client_idx=0；lissa_depth=1；lissa_damping=0.01；unlearn_eta=0.01。"),
        paragraph("三、论文主表", "Heading1"),
        table(paper_rows),
        paragraph("四、详细指标表", "Heading1"),
        table(detail_rows),
        paragraph("五、结果解读", "Heading1"),
        paragraph("FL 的 final global loss 为 1.6443，FedHDS 的 final global loss 为 1.6657。在第一阶段最小配置下，FedHDS 的 global loss 略高于 FL，但二者都完成了稳定训练。"),
        paragraph("forget-batch Hessian 的 forget loss 从 2.3381 上升到 2.3489，提升约 0.0108；global loss 从 1.6657 上升到 1.6867，说明有轻微 utility 损伤。"),
        paragraph("retained-set Hessian + stepped guard 的 forget loss 从 2.3381 上升到 2.3460，提升约 0.0079；global loss 从 1.6657 变为 1.6615，没有恶化，说明 guard 在当前设置下有效约束了 global utility。"),
        paragraph("因此第一阶段更适合写成最小闭环和可控性验证：forget-batch baseline 提供略强 forgetting signal，retained guard 提供更稳定的 utility preservation，但计算成本更高。"),
        paragraph("六、retained guard 的优点", "Heading1"),
        paragraph("第一，retained guard 能够在云端正式配置中稳定运行。原始 hessian_ref_per_client=4 在 RTX 5090 32GB 上 OOM，调整为 1 后成功完成。"),
        paragraph("第二，stepped guard 能够把 retained-set Hessian 更新限制在可控范围内，最终 5/5 个小步被接受，actual update L2 为 0.0427。"),
        paragraph("第三，global utility 更稳：forget-batch Hessian 使 global loss 从 1.6657 上升到 1.6867，而 retained guard 使 global loss 从 1.6657 变为 1.6615。"),
        paragraph("七、retained guard 的缺点", "Heading1"),
        paragraph("第一，forgetting 提升较弱，forget loss 提升约 0.0079，低于 forget-batch baseline 的约 0.0108。"),
        paragraph("第二，计算成本明显更高，retained guard 约 71.13 秒，远高于 forget-batch Hessian 的 1.44 秒。"),
        paragraph("第三，显存敏感，hessian_ref_per_client=4 会在 32GB GPU 上触发 CUDA OOM。"),
        paragraph("八、论文图", "Heading1"),
    ]

    for idx, ((_, caption), path) in enumerate(zip(figure_specs, figure_paths), start=1):
        body.append(paragraph(caption, "Heading2"))
        body.append(image_paragraph(f"rIdImg{idx}", caption, path, idx))

    body.extend(
        [
            paragraph("九、结论边界", "Heading1"),
            paragraph("不能把第一阶段写成 retained-set Hessian 全面优于 forget-batch Hessian。更稳妥的结论是：retained-set curvature approximation 在云端正式实验中可运行；global norm clipping 和 stepped trust-region guard 能够把 retained-set Hessian 更新变成可截停、可回滚、可按 utility 阈值控制的过程；在当前最小配置下，该方法能在不恶化 global loss 的情况下产生小幅 forget loss 提升，但 forgetting 提升弱于 forget-batch baseline，且计算成本更高。"),
            paragraph("十、推荐论文表述", "Heading1"),
            paragraph("Compared with the forget-batch Hessian baseline, the retained-set guarded update preserves global utility better under the stage-1 setting, but its forgetting gain is smaller and its computational cost is higher. This supports positioning the method as a controllable, utility-preserving unlearning framework rather than a forgetting-strength-dominant method."),
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
