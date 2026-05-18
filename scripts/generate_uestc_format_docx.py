from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

from update_stage3_full_reports import (
    FIGURES,
    ROOT,
    STAGE4A,
    STAGE4B,
    STAGE4C,
    STAGE4_SEED,
    appendix_caption,
    formula_to_omml,
    is_caption_line,
    is_md_separator,
    png_size,
    split_md_table_row,
    variable_explanation_to_omml,
)


PAPER_W = 11906
PAPER_H = 16838
MARGIN_3CM = 1701
HEADER_FOOTER_2CM = 1134
CONTENT_WIDTH_EMU = 5_400_000

APPENDIX_FIGURE_SELECTION = {
    "Stage 1": ["guard_steps.png", "unlearning_time.png", "update_norm_vs_forget_gain.png"],
    "Stage 2": ["guard_steps.png", "unlearning_time.png", "update_norm_vs_forget_gain.png", "update_norm_vs_global_loss.png"],
    "Stage 3": ["global_loss_final.png", "forget_loss_before_after.png", "update_norm_vs_global_loss.png"],
    "Stage 4A": ["guard_steps.png", "unlearning_time.png", "update_norm_vs_forget_gain.png"],
    "Stage 4B": ["guard_steps.png", "unlearning_time.png", "update_norm_vs_forget_gain.png"],
    "Stage 4C": ["guard_steps.png", "unlearning_time.png", "update_norm_vs_forget_gain.png"],
    "Stage 4 seed summary": ["global_loss_final.png", "guard_steps.png", "update_norm_vs_global_loss.png"],
    "Stage 4 seed43": ["guard_steps.png"],
    "Stage 4 seed44": ["guard_steps.png"],
}


def append_selected_appendix_figures(doc, zh: bool) -> None:
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
            selected = APPENDIX_FIGURE_SELECTION.get(title, [])
            images = [folder / name for name in selected if (folder / name).exists()] if selected else sorted(folder.glob("*.png"))
            for img in images:
                doc.image(img, appendix_caption(title, img, appendix_index, zh=zh))
                appendix_index += 1

APPENDIX_FIGURE_SELECTION = {
    "Stage 1": [
        "guard_steps.png",
        "unlearning_time.png",
        "update_norm_vs_forget_gain.png",
    ],
    "Stage 2": [
        "guard_steps.png",
        "unlearning_time.png",
        "update_norm_vs_forget_gain.png",
        "update_norm_vs_global_loss.png",
    ],
    "Stage 3": [
        "global_loss_final.png",
        "forget_loss_before_after.png",
        "update_norm_vs_global_loss.png",
    ],
    "Stage 4A": [
        "guard_steps.png",
        "unlearning_time.png",
        "update_norm_vs_forget_gain.png",
    ],
    "Stage 4B": [
        "guard_steps.png",
        "unlearning_time.png",
        "update_norm_vs_forget_gain.png",
    ],
    "Stage 4C": [
        "guard_steps.png",
        "unlearning_time.png",
        "update_norm_vs_forget_gain.png",
    ],
    "Stage 4 seed summary": [
        "global_loss_final.png",
        "guard_steps.png",
        "update_norm_vs_global_loss.png",
    ],
    "Stage 4 seed43": [
        "guard_steps.png",
    ],
    "Stage 4 seed44": [
        "guard_steps.png",
    ],
}


def _rpr(size: int = 24, bold: bool = False, superscript: bool = False) -> str:
    parts = [
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>',
        f'<w:sz w:val="{size}"/>',
        f'<w:szCs w:val="{size}"/>',
    ]
    if bold:
        parts.insert(0, "<w:b/>")
    if superscript:
        parts.append('<w:vertAlign w:val="superscript"/>')
    return "<w:rPr>" + "".join(parts) + "</w:rPr>"


def _text_run(text: str, size: int = 24, bold: bool = False, superscript: bool = False) -> str:
    return (
        f"<w:r>{_rpr(size=size, bold=bold, superscript=superscript)}"
        f'<w:t xml:space="preserve">{html.escape(text)}</w:t></w:r>'
    )


class UESTCDocxBuilder:
    def __init__(self, title: str):
        self.title = title
        self.body: list[str] = []
        self.rels: list[tuple[str, str, str]] = []
        self.media: list[tuple[Path, str]] = []
        self.rid = 1

    def inline_runs(self, text: str, size: int = 24, bold: bool = False) -> str:
        runs: list[str] = []
        for part in re.split(r"(<sup>.*?</sup>)", text, flags=re.S):
            if not part:
                continue
            sup = re.fullmatch(r"<sup>(.*?)</sup>", part, flags=re.S)
            if sup:
                runs.append(_text_run(sup.group(1), size=18, superscript=True))
            else:
                runs.append(_text_run(part, size=size, bold=bold))
        return "".join(runs) or _text_run("")

    def p(self, text: str = "", style: str = "Normal", size: int = 24) -> None:
        one_line = text.splitlines()[0] if text.splitlines() else ""
        self.body.append(
            f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
            f"{self.inline_runs(one_line, size=size)}</w:p>"
        )

    def page_break(self) -> None:
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def title_heading(self, text: str) -> None:
        self.body.append(
            '<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>'
            f"{self.inline_runs(text, size=32, bold=True)}</w:p>"
        )

    def toc(self, title: str, placeholder: str) -> None:
        instr = html.escape('TOC \\o "1-3" \\h \\z \\u')
        self.body.append(
            '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
            f"{self.inline_runs(title, size=30, bold=True)}</w:p>"
        )
        self.body.append(
            '<w:p><w:pPr><w:pStyle w:val="BodyNoIndent"/></w:pPr>'
            '<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
            f'<w:r><w:instrText xml:space="preserve">{instr}</w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            f"{self.inline_runs(placeholder)}"
            '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
        )

    def heading(self, text: str, level: int = 1) -> None:
        style = f"Heading{min(max(level, 1), 3)}"
        self.body.append(
            f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
            f"{self.inline_runs(text, size=30 if level == 1 else 28, bold=True)}</w:p>"
        )

    def caption(self, text: str) -> None:
        self.body.append(
            '<w:p><w:pPr><w:pStyle w:val="Caption"/></w:pPr>'
            f"{self.inline_runs(text, size=21, bold=False)}</w:p>"
        )

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        ncols = max(len(row) for row in rows)
        cell_width = max(900, min(2600, 8500 // max(ncols, 1)))
        xml = [
            '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
            '<w:tblLayout w:type="autofit"/>'
            '<w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
            '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
            '<w:tblBorders>'
            '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:left w:val="nil"/>'
            '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:right w:val="nil"/>'
            '<w:insideH w:val="nil"/>'
            '<w:insideV w:val="nil"/>'
            '</w:tblBorders>'
            '<w:tblLook w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" '
            'w:noHBand="1" w:noVBand="1"/></w:tblPr>'
        ]
        for row_idx, row in enumerate(rows):
            is_header = row_idx == 0
            xml.append("<w:tr>")
            if is_header:
                xml.append("<w:trPr><w:tblHeader/></w:trPr>")
            for cell in row:
                header_border = (
                    '<w:tcBorders><w:bottom w:val="single" w:sz="8" w:space="0" '
                    'w:color="000000"/></w:tcBorders>'
                    if is_header
                    else ""
                )
                bold = "<w:b/>" if is_header else ""
                xml.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{cell_width}" w:type="dxa"/>'
                    f'<w:vAlign w:val="center"/>{header_border}</w:tcPr>'
                    '<w:p><w:pPr><w:pStyle w:val="TableText"/><w:jc w:val="center"/></w:pPr>'
                    '<w:r><w:rPr>'
                    f'{bold}<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>'
                    '<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                    f'<w:t xml:space="preserve">{html.escape(str(cell))}</w:t></w:r></w:p></w:tc>'
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
        self.rels.append(
            (
                rid,
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                f"media/{media_name}",
            )
        )
        width, height = png_size(path)
        cx = CONTENT_WIDTH_EMU
        cy = int(CONTENT_WIDTH_EMU * height / max(width, 1))
        self.body.append(
            '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            '<wp:inline distT="0" distB="0" distL="0" distR="0" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{self.rid + 100}" name="{html.escape(caption)}"/>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="0" name="{html.escape(media_name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
            '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
        )
        self.caption(caption)

    def equation(self, math_xml: str) -> None:
        self.body.append(
            '<w:p><m:oMathPara><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>'
            f"<m:oMath>{math_xml}</m:oMath></m:oMathPara></w:p>"
        )

    def equation_explanation(self, math_xml: str, explanation: str) -> None:
        self.body.append(
            '<w:p><w:pPr><w:pStyle w:val="BodyNoIndent"/></w:pPr>'
            f'{_text_run("- ")}<m:oMath>{math_xml}</m:oMath>'
            f"{_text_run(explanation)}</w:p>"
        )

    def _styles_xml(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:jc w:val="both"/><w:spacing w:before="0" w:after="0" w:line="360" w:lineRule="auto"/><w:ind w:firstLine="480"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="BodyNoIndent"><w:name w:val="BodyNoIndent"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="both"/><w:spacing w:before="0" w:after="0" w:line="360" w:lineRule="auto"/><w:ind w:firstLine="0"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="360" w:line="360" w:lineRule="auto"/><w:ind w:firstLine="0"/></w:pPr><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:keepNext/><w:outlineLvl w:val="0"/><w:spacing w:before="480" w:after="360" w:line="360" w:lineRule="auto"/><w:ind w:firstLine="0"/></w:pPr><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="1"/><w:spacing w:before="360" w:after="120" w:line="360" w:lineRule="auto"/><w:ind w:firstLine="0"/></w:pPr><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="2"/><w:spacing w:before="240" w:after="120" w:line="360" w:lineRule="auto"/><w:ind w:firstLine="0"/></w:pPr><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="240" w:line="300" w:lineRule="auto"/><w:ind w:firstLine="0"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="TableText"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="0" w:line="300" w:lineRule="auto"/><w:ind w:firstLine="0"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:style>
</w:styles>"""

    def save(self, path: Path) -> None:
        document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<w:body>{''.join(self.body)}<w:sectPr><w:pgSz w:w="{PAPER_W}" w:h="{PAPER_H}"/><w:pgMar w:top="{MARGIN_3CM}" w:right="{MARGIN_3CM}" w:bottom="{MARGIN_3CM}" w:left="{MARGIN_3CM}" w:header="{HEADER_FOOTER_2CM}" w:footer="{HEADER_FOOTER_2CM}" w:gutter="0"/></w:sectPr></w:body></w:document>"""
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
            z.writestr("word/styles.xml", self._styles_xml())
            z.writestr("word/settings.xml", settings)
            z.writestr("word/_rels/document.xml.rels", "".join(doc_rels))
            for src, name in self.media:
                z.write(src, f"word/media/{name}")


def markdown_to_uestc_docx(md_path: Path, out_path: Path) -> None:
    doc = UESTCDocxBuilder("UESTC formatted paper draft")
    lines = md_path.read_text(encoding="utf-8").splitlines()
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    zh = "_zh" in md_path.stem or md_path.name.endswith("_zh.md")
    inserted_toc = False
    inserted_appendix = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if line.startswith("# "):
            doc.title_heading(line[2:].strip())
        elif line.startswith("## "):
            if not inserted_toc and re.match(r"##\s+1(?:\.|\s)", stripped):
                doc.toc(
                    "目录" if zh else "Table of Contents",
                    "打开 Word 后更新域以生成目录。" if zh else "Update fields in Word to generate the table of contents.",
                )
                doc.page_break()
                inserted_toc = True
            heading_text = line[3:].strip()
            doc.heading(heading_text, 1)
            if heading_text == "Appendix A. Supplementary Figures":
                append_selected_appendix_figures(doc, zh)
                inserted_appendix = True
        elif line.startswith("### "):
            doc.heading(line[4:].strip(), 2)
        elif line.startswith("#### "):
            doc.heading(line[5:].strip(), 3)
        elif image_match := image_pattern.match(stripped):
            caption = image_match.group(1).strip() or "图"
            img_path = Path(image_match.group(2).strip())
            if not img_path.is_absolute():
                img_path = (md_path.parent / img_path).resolve()
            doc.image(img_path, caption)
            if i + 2 < len(lines) and lines[i + 1].strip() == "":
                next_caption = lines[i + 2].strip()
                if next_caption in {f"*{caption}*", f"**{caption}**"}:
                    i += 2
        elif equation_xml := formula_to_omml(stripped):
            doc.equation(equation_xml)
        elif equation_item := variable_explanation_to_omml(stripped):
            doc.equation_explanation(equation_item[0], equation_item[1])
        elif caption := is_caption_line(stripped):
            doc.caption(caption)
        elif line.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].startswith("|"):
                if not is_md_separator(lines[i]):
                    table_lines.append(lines[i])
                i += 1
            if table_lines:
                doc.table([split_md_table_row(item) for item in table_lines])
            continue
        elif stripped:
            doc.p(re.sub(r"`([^`]+)`", r"\1", stripped))
        else:
            doc.p("", style="BodyNoIndent")
        i += 1

    if not inserted_appendix:
        doc.heading("Appendix Figures", 1)
        append_selected_appendix_figures(doc, zh)

    doc.save(out_path)


def main() -> None:
    markdown_to_uestc_docx(ROOT / "paper_draft_zh.md", ROOT / "paper_draft_zh_uestc_template.docx")
    english_md = ROOT / "paper_draft_final.md"
    if english_md.exists():
        markdown_to_uestc_docx(english_md, ROOT / "paper_draft_final_uestc_template.docx")


if __name__ == "__main__":
    main()
