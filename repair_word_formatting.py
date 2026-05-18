from __future__ import annotations

import copy
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


TARGET = Path("paper_draft_submission_v2_uestc_template.docx")
REFERENCE = Path("paper_ZhaoFengyuan_FedHDS.docx")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"

ET.register_namespace("w", W_NS)
ET.register_namespace("m", M_NS)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")


def qn(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def text_of(elem: ET.Element) -> str:
    return "".join(t.text or "" for t in elem.findall(".//w:t", {"w": W_NS})).strip()


def has_math(elem: ET.Element) -> bool:
    return elem.find(".//m:oMath", {"m": M_NS, "w": W_NS}) is not None or elem.find(
        ".//m:oMathPara", {"m": M_NS, "w": W_NS}
    ) is not None


def is_blank_paragraph(elem: ET.Element) -> bool:
    return elem.tag == qn(W_NS, "p") and not text_of(elem) and not has_math(elem)


def is_formula_caption(elem: ET.Element) -> bool:
    return elem.tag == qn(W_NS, "p") and bool(re.match(r"^Formula \(\d+\)\.", text_of(elem)))


def formula_number(elem: ET.Element) -> int:
    m = re.match(r"^Formula \((\d+)\)\.", text_of(elem))
    if not m:
        raise ValueError("not a formula caption")
    return int(m.group(1))


def is_heading_like(text: str) -> bool:
    return bool(
        re.match(
            r"^(\d+(\.\d+)*\s)|^Algorithm\b|^Table\b|^Figure\b|^References$|^Appendix\b",
            text,
        )
    )


def is_equation_placeholder(elem: ET.Element) -> bool:
    if elem.tag != qn(W_NS, "p") or has_math(elem):
        return False
    txt = text_of(elem)
    if not txt or is_formula_caption(elem) or is_heading_like(txt):
        return False
    equation_markers = ("=", "≤", "≥", "||")
    return len(txt) <= 160 and not txt.endswith(".") and any(marker in txt for marker in equation_markers)


def clone(elem: ET.Element) -> ET.Element:
    return copy.deepcopy(elem)


def load_document(path: Path) -> tuple[ET.Element, ET.Element, list[ET.Element]]:
    with zipfile.ZipFile(path, "r") as z:
        root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(qn(W_NS, "body"))
    if body is None:
        raise RuntimeError(f"{path} has no w:body")
    children = list(body)
    return root, body, children


def collect_reference_assets(ref_children: list[ET.Element]) -> tuple[ET.Element, dict[int, list[ET.Element]]]:
    table_caption = "Table. Qualitative comparison of federated unlearning approaches."
    ref_table: ET.Element | None = None
    formula_math: dict[int, list[ET.Element]] = {}

    i = 0
    while i < len(ref_children):
        child = ref_children[i]
        txt = text_of(child)
        if txt == table_caption:
            j = i + 1
            while j < len(ref_children):
                if ref_children[j].tag == qn(W_NS, "tbl"):
                    ref_table = clone(ref_children[j])
                    break
                j += 1
        if is_formula_caption(child):
            num = formula_number(child)
            seq: list[ET.Element] = []
            j = i + 1
            while j < len(ref_children):
                nxt = ref_children[j]
                nxt_txt = text_of(nxt)
                nxt_math = has_math(nxt)
                if not nxt_txt and not nxt_math:
                    j += 1
                    continue
                if nxt_math:
                    seq.append(clone(nxt))
                    j += 1
                    continue
                break
            formula_math[num] = seq
        i += 1

    if ref_table is None:
        raise RuntimeError("Could not find reference table")
    return ref_table, formula_math


def rebuild_target(
    tgt_children: list[ET.Element],
    ref_table: ET.Element,
    formula_math: dict[int, list[ET.Element]],
) -> list[ET.Element]:
    table_caption = "Table. Qualitative comparison of federated unlearning approaches."
    out: list[ET.Element] = []
    i = 0
    replace_next_tbl = False

    while i < len(tgt_children):
        child = tgt_children[i]
        txt = text_of(child)

        if txt == table_caption:
            out.append(clone(child))
            replace_next_tbl = True
            i += 1
            continue

        if replace_next_tbl and child.tag == qn(W_NS, "tbl"):
            out.append(clone(ref_table))
            replace_next_tbl = False
            i += 1
            continue

        if is_formula_caption(child):
            out.append(clone(child))
            num = formula_number(child)
            ref_seq = formula_math.get(num, [])
            i += 1
            while i < len(tgt_children):
                nxt = tgt_children[i]
                if is_blank_paragraph(nxt) or has_math(nxt) or is_equation_placeholder(nxt):
                    i += 1
                    continue
                break
            for math_para in ref_seq:
                out.append(clone(math_para))
            continue

        out.append(clone(child))
        i += 1

    return out


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)
    if not REFERENCE.exists():
        raise FileNotFoundError(REFERENCE)

    tgt_root, tgt_body, tgt_children = load_document(TARGET)
    _, _, ref_children = load_document(REFERENCE)

    ref_table, formula_math = collect_reference_assets(ref_children)
    rebuilt_children = rebuild_target(tgt_children, ref_table, formula_math)

    # Replace the body contents while preserving the final sectPr.
    for child in list(tgt_body):
        tgt_body.remove(child)
    for child in rebuilt_children:
        tgt_body.append(child)

    xml_bytes = ET.tostring(tgt_root, encoding="utf-8", xml_declaration=True)

    backup = TARGET.with_name(f"{TARGET.stem}.before_format_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.docx")
    shutil.copy2(TARGET, backup)

    fd, tmp_name = tempfile.mkstemp(suffix=".docx", dir=str(TARGET.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    repaired_path = TARGET.with_name(f"{TARGET.stem}.repaired.docx")
    try:
        with zipfile.ZipFile(TARGET, "r") as zin, zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = xml_bytes
                zout.writestr(item, data)
        try:
            os.replace(tmp_path, TARGET)
            print(f"Updated {TARGET}")
            print(f"Backup  {backup}")
            return
        except PermissionError:
            shutil.copy2(tmp_path, repaired_path)
            print(f"Target is locked; wrote repaired copy to {repaired_path}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print(f"Backup  {backup}")


if __name__ == "__main__":
    main()
