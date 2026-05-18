from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


DOCX_PATH = Path("paper_draft_submission_v2_uestc_template.docx")
MD_PATH = Path("paper_draft_submission_v2.md")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("w", W_NS)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")


def qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def text_of(elem: ET.Element) -> str:
    return "".join(t.text or "" for t in elem.findall(".//w:t", {"w": W_NS})).strip()


def style_of_paragraph(para: ET.Element) -> str | None:
    node = para.find("./w:pPr/w:pStyle", {"w": W_NS})
    if node is None:
        return None
    return node.get(qn("val"))


def set_paragraph_text(para: ET.Element, text: str) -> None:
    ppr = para.find(qn("pPr"))
    for child in list(para):
        if child.tag != qn("pPr"):
            para.remove(child)
    run = ET.SubElement(para, qn("r"))
    text_node = ET.SubElement(run, qn("t"))
    text_node.set(f"{{{XML_NS}}}space", "preserve")
    text_node.text = text
    if ppr is not None and para[0] is not ppr:
        para.remove(ppr)
        para.insert(0, ppr)


def make_paragraph(text: str, style_id: str | None = None) -> ET.Element:
    p = ET.Element(qn("p"))
    if style_id:
        ppr = ET.SubElement(p, qn("pPr"))
        pstyle = ET.SubElement(ppr, qn("pStyle"))
        pstyle.set(qn("val"), style_id)
    run = ET.SubElement(p, qn("r"))
    text_node = ET.SubElement(run, qn("t"))
    text_node.set(f"{{{XML_NS}}}space", "preserve")
    text_node.text = text
    return p


def make_table(headers: list[str], rows: list[list[str]]) -> ET.Element:
    tbl = ET.Element(qn("tbl"))
    tbl_pr = ET.SubElement(tbl, qn("tblPr"))
    tbl_style = ET.SubElement(tbl_pr, qn("tblStyle"))
    tbl_style.set(qn("val"), "TableGrid")
    tbl_w = ET.SubElement(tbl_pr, qn("tblW"))
    tbl_w.set(qn("w"), "0")
    tbl_w.set(qn("type"), "auto")
    borders = ET.SubElement(tbl_pr, qn("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = ET.SubElement(borders, qn(edge))
        b.set(qn("val"), "single")
        b.set(qn("sz"), "8")
        b.set(qn("space"), "0")
        b.set(qn("color"), "auto")
    tbl_look = ET.SubElement(tbl_pr, qn("tblLook"))
    tbl_look.set(qn("val"), "04A0")
    tbl_look.set(qn("firstRow"), "1")
    tbl_look.set(qn("lastRow"), "0")
    tbl_look.set(qn("firstColumn"), "1")
    tbl_look.set(qn("lastColumn"), "0")
    tbl_look.set(qn("noHBand"), "0")
    tbl_look.set(qn("noVBand"), "1")

    def add_row(values: list[str]) -> None:
        tr = ET.SubElement(tbl, qn("tr"))
        for value in values:
            tc = ET.SubElement(tr, qn("tc"))
            tc_pr = ET.SubElement(tc, qn("tcPr"))
            tc_w = ET.SubElement(tc_pr, qn("tcW"))
            tc_w.set(qn("w"), "0")
            tc_w.set(qn("type"), "auto")
            tc.append(make_paragraph(value))

    add_row(headers)
    for row in rows:
        add_row(row)
    return tbl


def clean_text(text: str) -> str:
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = re.sub(r"<sup>\[(.*?)\]</sup>", r"[\1]", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def parse_markdown_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    def parse_row(line: str) -> list[str]:
        parts = [clean_text(x) for x in line.strip().strip("|").split("|")]
        return [p for p in parts]

    headers = parse_row(lines[0])
    rows = [parse_row(line) for line in lines[2:]]
    return headers, rows


def parse_markdown_blocks(text: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].replace("\ufeff", "")
        line = raw.strip()
        if not line:
            i += 1
            continue
        if line.startswith("!["):
            i += 1
            continue
        if line.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append(("table", parse_markdown_table(table_lines)))
            continue
        if line.startswith("## "):
            blocks.append(("h2", clean_text(line[3:])))
            i += 1
            continue
        if line.startswith("### "):
            blocks.append(("h3", clean_text(line[4:])))
            i += 1
            continue
        if re.match(r"^\d+\.\s", line) or line.startswith("- "):
            blocks.append(("p", clean_text(line)))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                break
            if nxt.startswith(("## ", "### ", "|", "![")) or re.match(r"^\d+\.\s", nxt) or nxt.startswith("- "):
                break
            paragraph.append(nxt)
            i += 1
        blocks.append(("p", clean_text(" ".join(paragraph))))
    return blocks


def extract_section(md_text: str, start_heading: str, end_heading: str) -> str:
    start = md_text.index(start_heading)
    end = md_text.index(end_heading, start)
    return md_text[start:end].strip()


def convert_blocks_to_xml(
    blocks: list[tuple[str, object]],
    h2_style: str | None,
    h3_style: str | None,
) -> list[ET.Element]:
    elems: list[ET.Element] = []
    for kind, payload in blocks:
        if kind == "h2":
            elems.append(make_paragraph(payload, h2_style))
        elif kind == "h3":
            elems.append(make_paragraph(payload, h3_style))
        elif kind == "p":
            elems.append(make_paragraph(payload))
        elif kind == "table":
            headers, rows = payload  # type: ignore[misc]
            elems.append(make_table(headers, rows))
    return elems


def find_body_child_index(body: ET.Element, target_text: str) -> int:
    for idx, child in enumerate(list(body)):
        if child.tag == qn("p") and text_of(child) == target_text:
            return idx
    raise ValueError(f"Could not find paragraph with text: {target_text}")


def replace_children(body: ET.Element, start_idx: int, end_idx: int, new_children: list[ET.Element]) -> None:
    current = list(body)
    for child in current[start_idx:end_idx]:
        body.remove(child)
    for offset, child in enumerate(new_children):
        body.insert(start_idx + offset, child)


def main() -> None:
    if not DOCX_PATH.exists():
        raise FileNotFoundError(DOCX_PATH)
    if not MD_PATH.exists():
        raise FileNotFoundError(MD_PATH)

    md_text = MD_PATH.read_text(encoding="utf-8")
    keyword_match = re.search(r"(?m)^Keywords:\s*(.+)$", md_text)
    keywords_text = f"Keywords: {keyword_match.group(1).strip()}" if keyword_match else None
    section_1_to_4 = extract_section(
        md_text,
        "## 1. Introduction",
        "## 5. Experimental Results and Mechanism Analysis",
    )
    section_5_to_7 = extract_section(
        md_text,
        "## 5. Experimental Results and Mechanism Analysis",
        "## References",
    )

    with zipfile.ZipFile(DOCX_PATH, "r") as zin:
        xml_bytes = zin.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    body = root.find(qn("body"))
    if body is None:
        raise RuntimeError("word/document.xml does not contain w:body")

    title_para = next((p for p in body.findall(qn("p")) if "Guarded Federated Unlearning" in text_of(p)), None)
    if title_para is not None:
        set_paragraph_text(title_para, "Guarded Federated Unlearning via Retained-Set Curvature")
    if keywords_text is not None:
        keywords_para = next((p for p in body.findall(qn("p")) if text_of(p).startswith("Keywords:")), None)
        if keywords_para is not None:
            set_paragraph_text(keywords_para, keywords_text)

    h2_style = None
    h3_style = None
    for para in body.findall(qn("p")):
        txt = text_of(para)
        if txt == "1. Introduction":
            h2_style = style_of_paragraph(para)
        elif txt == "2.1 Federated Unlearning":
            h3_style = style_of_paragraph(para)
        if h2_style and h3_style:
            break

    blocks_1_to_4 = parse_markdown_blocks(section_1_to_4)
    blocks_5_to_7 = parse_markdown_blocks(section_5_to_7)
    xml_1_to_4 = convert_blocks_to_xml(blocks_1_to_4, h2_style, h3_style)
    xml_5_to_7 = convert_blocks_to_xml(blocks_5_to_7, h2_style, h3_style)

    start_1 = find_body_child_index(body, "1. Introduction")
    end_1 = find_body_child_index(body, "5. Experimental Results and Mechanism Analysis")
    replace_children(body, start_1, end_1, xml_1_to_4)

    start_5 = find_body_child_index(body, "5. Experimental Results and Mechanism Analysis")
    end_5 = find_body_child_index(body, "References")
    replace_children(body, start_5, end_5, xml_5_to_7)

    new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DOCX_PATH.with_name(f"{DOCX_PATH.stem}.before_word_sync_{stamp}.bak.docx")
    shutil.copy2(DOCX_PATH, backup)

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".docx", dir=str(DOCX_PATH.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(DOCX_PATH, "r") as zin, zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = new_xml
                zout.writestr(item, data)
        os.replace(tmp_path, DOCX_PATH)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"Updated {DOCX_PATH}")
    print(f"Backup  {backup}")

    repair_script = DOCX_PATH.with_name("repair_word_formatting.py")
    reference_docx = DOCX_PATH.with_name("paper_ZhaoFengyuan_FedHDS.docx")
    if repair_script.exists() and reference_docx.exists():
        subprocess.run([sys.executable, str(repair_script)], cwd=str(DOCX_PATH.parent), check=True)


if __name__ == "__main__":
    main()
