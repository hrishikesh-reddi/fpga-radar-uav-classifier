#!/usr/bin/env python3
"""
Convert Technical_Report.md → Technical_Report.pdf using fpdf2.
Handles: headings (H1-H4), tables, code blocks, bullet lists,
         bold/italic inline text, and embedded PNG images.
"""

import os
import re
import sys

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH    = os.path.join(REPORT_DIR, "Technical_Report.md")
PDF_PATH   = os.path.join(REPORT_DIR, "Technical_Report.pdf")

# ── Attempt fpdf2 import ────────────────────────────────────────────────
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    print("[ERROR] fpdf2 not installed. Run: pip install fpdf2")
    sys.exit(1)

# ── Colour palette ──────────────────────────────────────────────────────
DARK_BLUE   = (15,  40,  80)
MID_BLUE    = (30,  80, 160)
LIGHT_BLUE  = (220, 235, 255)
BLACK       = (20,  20,  20)
GRAY_RULE   = (160, 160, 160)
CODE_BG     = (240, 240, 240)
TABLE_HDR   = (15,  40,  80)
TABLE_ROW_A = (245, 248, 255)
TABLE_ROW_B = (255, 255, 255)

PAGE_W = 210
PAGE_H = 297
MARGIN = 18
COL_W  = PAGE_W - 2 * MARGIN


class ReportPDF(FPDF):
    def header(self):
        # Top colour bar
        self.set_fill_color(*DARK_BLUE)
        self.rect(0, 0, PAGE_W, 8, style="F")
        self.ln(9)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY_RULE)
        self.cell(0, 6, f"BITS Pilani AMD/Xilinx FPGA Hackathon 2026  ·  Page {self.page_no()}", align="C")


def sanitize(text: str) -> str:
    """Replace Unicode chars unsupported by fpdf2 latin-1 core fonts."""
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "--",  # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2192": "->",  # rightwards arrow
        "\u2264": "<=",  # less-than-or-equal
        "\u2265": ">=",  # greater-than-or-equal
        "\u2248": "~",   # approximately equal
        "\u00b5": "us",  # micro sign
        "\u2713": "OK",  # check mark
        "\u2714": "OK",  # heavy check mark
        "\u2705": "[OK]", # green check
        "\u2715": "X",   # multiplication X
        "\u00d7": "x",   # multiplication sign
        "\u2022": "*",   # bullet
        "\u2026": "...", # ellipsis
        "\u00b0": "deg", # degree sign
        "\u00b7": ".",   # middle dot
        "\u25cf": "*",   # black circle
        "\u2212": "-",   # minus sign
        "\u00e9": "e",   # e-acute
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Strip anything still outside latin-1
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text

def strip_bold_italic(text: str) -> str:
    """Remove markdown ** and * from text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"`(.+?)`",       r"\1", text)
    return sanitize(text)


def parse_inline(pdf: FPDF, text: str, base_size: float, base_color: tuple):
    """Render inline markdown (bold, italic, code) as mixed-style cells."""
    # Split on bold / italic / code markers
    pattern = re.compile(r"(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)")
    last = 0
    for m in pattern.finditer(text):
        # Plain text before
        plain = text[last:m.start()]
        if plain:
            pdf.set_font("Helvetica", size=base_size)
            pdf.set_text_color(*base_color)
            pdf.write(5, sanitize(plain))
        # Matched segment
        raw = m.group(0)
        if raw.startswith("**"):
            pdf.set_font("Helvetica", "B", base_size)
            pdf.set_text_color(*base_color)
            pdf.write(5, sanitize(m.group(2)))
        elif raw.startswith("`"):
            pdf.set_font("Courier", size=base_size - 1)
            pdf.set_text_color(120, 0, 0)
            pdf.write(5, sanitize(m.group(4)))
        else:
            pdf.set_font("Helvetica", "I", base_size)
            pdf.set_text_color(*base_color)
            pdf.write(5, sanitize(m.group(3)))
        last = m.end()
    # Trailing plain text
    if last < len(text):
        pdf.set_font("Helvetica", size=base_size)
        pdf.set_text_color(*base_color)
        pdf.write(5, sanitize(text[last:]))


def build_pdf():
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(MARGIN, 6, MARGIN)
    pdf.add_page()
    pdf.set_text_shaping(False)

    with open(MD_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        # ── H1 ──────────────────────────────────────────────────────────
        if line.startswith("# ") and not line.startswith("## "):
            text = line[2:].strip()
            pdf.set_fill_color(*DARK_BLUE)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(COL_W, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            pdf.ln(3)
            i += 1
            continue

        # ── H2 ──────────────────────────────────────────────────────────
        if line.startswith("## "):
            text = line[3:].strip()
            pdf.ln(3)
            pdf.set_fill_color(*MID_BLUE)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(COL_W, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            pdf.ln(2)
            i += 1
            continue

        # ── H3 ──────────────────────────────────────────────────────────
        if line.startswith("### "):
            text = line[4:].strip()
            pdf.ln(2)
            pdf.set_text_color(*DARK_BLUE)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(COL_W, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            # Underline rule
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.set_draw_color(*MID_BLUE)
            pdf.set_line_width(0.4)
            pdf.line(MARGIN, y, MARGIN + COL_W, y)
            pdf.ln(1)
            i += 1
            continue

        # ── H4 ──────────────────────────────────────────────────────────
        if line.startswith("#### "):
            text = line[5:].strip()
            pdf.ln(1)
            pdf.set_text_color(*MID_BLUE)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(COL_W, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            i += 1
            continue

        # ── Horizontal rule ─────────────────────────────────────────────
        if line.strip() in ("---", "***", "___"):
            pdf.ln(2)
            pdf.set_draw_color(*GRAY_RULE)
            pdf.set_line_width(0.3)
            y = pdf.get_y()
            pdf.line(MARGIN, y, MARGIN + COL_W, y)
            pdf.ln(3)
            i += 1
            continue

        # ── Code block ──────────────────────────────────────────────────
        if line.strip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            i += 1  # skip closing ```
            pdf.ln(2)
            pdf.set_fill_color(*CODE_BG)
            pdf.set_draw_color(200, 200, 200)
            pdf.set_line_width(0.2)
            block_h = len(code_lines) * 4.5 + 4
            x0 = pdf.get_x()
            y0 = pdf.get_y()
            pdf.rect(x0, y0, COL_W, block_h, style="FD")
            pdf.set_font("Courier", size=7)
            pdf.set_text_color(*BLACK)
            for cl in code_lines:
                pdf.set_x(MARGIN + 2)
                pdf.cell(COL_W - 4, 4.5, cl[:120], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            continue

        # ── Table ────────────────────────────────────────────────────────
        if line.strip().startswith("|"):
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row_text = lines[i].strip()
                # Skip separator rows like |---|---|
                if re.match(r"^\|[\s\-:|\s]+\|$", row_text):
                    i += 1
                    continue
                cells = [c.strip() for c in row_text.strip("|").split("|")]
                table_rows.append(cells)
                i += 1

            if not table_rows:
                continue

            pdf.ln(2)
            num_cols = max(len(r) for r in table_rows)
            col_w_each = COL_W / num_cols

            for ridx, row in enumerate(table_rows):
                is_header = (ridx == 0)
                fill_color = TABLE_HDR if is_header else (TABLE_ROW_A if ridx % 2 == 0 else TABLE_ROW_B)
                text_color = (255, 255, 255) if is_header else BLACK
                font_style = "B" if is_header else ""

                pdf.set_fill_color(*fill_color)
                pdf.set_text_color(*text_color)
                pdf.set_font("Helvetica", font_style, 8)
                pdf.set_draw_color(200, 200, 200)
                pdf.set_line_width(0.1)

                # Fixed row height, text truncated if needed
                row_h = 5.5
                for j in range(num_cols):
                    cell_text = strip_bold_italic(row[j]) if j < len(row) else ""
                    # Truncate so it fits
                    while pdf.get_string_width(cell_text) > col_w_each - 2 and len(cell_text) > 3:
                        cell_text = cell_text[:-4] + "…"
                    pdf.cell(col_w_each, row_h, cell_text, border=1, fill=True,
                              new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.ln(row_h)

            pdf.ln(2)
            continue

        # ── Image reference ──────────────────────────────────────────────
        m = re.match(r"!\[(.+?)\]\((.+?)\)", line.strip())
        if m:
            alt  = m.group(1)
            path = m.group(2)
            img_path = os.path.join(REPORT_DIR, path)
            if os.path.exists(img_path):
                # Check if enough space remains; if not, new page
                if pdf.get_y() + 80 > PAGE_H - 20:
                    pdf.add_page()
                pdf.ln(2)
                img_w = min(COL_W, 150)
                x_img = MARGIN + (COL_W - img_w) / 2
                pdf.image(img_path, x=x_img, w=img_w)
                # Caption
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(*GRAY_RULE)
                pdf.cell(COL_W, 5, alt, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
                pdf.ln(2)
            else:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(*GRAY_RULE)
                pdf.cell(COL_W, 5, f"[Image not found: {path}]",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            i += 1
            continue

        # ── Bullet list ──────────────────────────────────────────────────
        m_bullet = re.match(r"^(\s*)[*\-]\s+(.*)", line)
        if m_bullet:
            indent_level = len(m_bullet.group(1)) // 2
            content = m_bullet.group(2)
            indent_mm = MARGIN + 4 + indent_level * 4
            bullet_sym = "•" if indent_level == 0 else "–"
            pdf.set_x(indent_mm)
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(*BLACK)
            pdf.cell(4, 5, bullet_sym)
            pdf.set_x(indent_mm + 4)
            # Write inline with bold/italic support
            parse_inline(pdf, content, 9, BLACK)
            pdf.ln(5)
            i += 1
            continue

        # ── Blockquote / note ────────────────────────────────────────────
        if line.strip().startswith(">"):
            text = re.sub(r"^>\s*", "", line.strip())
            pdf.ln(1)
            pdf.set_fill_color(230, 240, 255)
            pdf.set_x(MARGIN + 3)
            pdf.set_font("Helvetica", "I", 8.5)
            pdf.set_text_color(60, 60, 120)
            pdf.multi_cell(COL_W - 3, 5, strip_bold_italic(text), fill=True)
            pdf.ln(1)
            i += 1
            continue

        # ── Blank line ───────────────────────────────────────────────────
        if not line.strip():
            pdf.ln(2)
            i += 1
            continue

        # ── Regular paragraph ────────────────────────────────────────────
        pdf.set_x(MARGIN)
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(*BLACK)
        parse_inline(pdf, sanitize(line.rstrip()), 9, BLACK)
        pdf.ln(5)
        i += 1

    pdf.output(PDF_PATH)
    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"[OK] PDF written: {PDF_PATH}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build_pdf()
