"""Render the three formatted V4.1 paper tables as single-page vector PDFs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle


TABLES = {
    1: (
        "table_1_main_results.pdf", (198, 119, 126, 124, 76, 55, 62, 84), 8,
        "All methods use 100 common holdout markets. Brackets show 95% confidence "
        "intervals from resampling available training seeds and holdout anchors. "
        "Single-seed intervals reflect market uncertainty.",
    ),
    2: (
        "table_2_matched_credit.pdf", (208, 170, 164), 5,
        "Differences are paired by training seed and holdout market against standalone. "
        "Brackets show 95% bootstrap confidence intervals. Positive relative wealth "
        "and negative cost gap favor the listed method.",
    ),
    3: (
        "table_3_compute.pdf", (198, 84, 69, 72, 82, 70, 72, 72), 8,
        "Relative values use shared-global RCPO as the reference. Training timing "
        "and memory ratios require matching hardware, timing boundaries and "
        "interaction budgets. N/A indicates an inapplicable or unavailable value.",
    ),
}
INK = colors.HexColor("#172934")
MUTED = colors.HexColor("#52636E")
TEAL = colors.HexColor("#14675E")
LINE = colors.HexColor("#D9E3E4")
STRIPE = colors.HexColor("#F3F7F7")


def parse_paper_tables(path: Path) -> dict[int, tuple[str, list[list[str]]]]:
    parsed = {}
    for section in re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.MULTILINE):
        title = re.match(r"Table ([123])\. (.+)", section)
        if title is None:
            continue
        number = int(title[1])
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.splitlines()
            if line.startswith("|")
        ]
        if number in parsed or len(rows) != TABLES[number][2] + 2:
            raise ValueError(f"Unexpected number of Table {number} rows in {path}")
        headers, _, *body = rows
        expected_columns = len(TABLES[number][1])
        if any(len(row) != expected_columns for row in [headers, *body]):
            raise ValueError(f"Unexpected number of Table {number} columns in {path}")
        parsed[number] = (title[2], [headers, *body])
    if parsed.keys() != TABLES.keys():
        raise ValueError(f"Expected Tables 1, 2 and 3 in {path}")
    return parsed


def render(folder: Path, selection: str, seed_label: str) -> None:
    for number, (title, cells) in parse_paper_tables(folder / "paper_tables.md").items():
        filename, widths, _, caption = TABLES[number]
        margin = 29
        table_width = sum(widths)
        header_styles = [
            ParagraphStyle(
                "header", fontName="Paper-Bold", fontSize=9, leading=11.5,
                textColor=colors.white, alignment=alignment,
            )
            for alignment in (TA_LEFT, TA_RIGHT)
        ]
        headings = [
            Paragraph(escape(label).replace("⁻³", "<super>-3</super>"), header_styles[int(index != 0)])
            for index, label in enumerate(cells[0])
        ]
        table = Table([headings, *cells[1:]], colWidths=widths, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), (colors.white, STRIPE)),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK),
            ("FONTNAME", (0, 1), (-1, -1), "Paper-Regular"),
            ("FONTSIZE", (0, 1), (-1, -1), 9.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, LINE),
        ]))
        for row in cells[1:]:
            for value, width in zip(row, widths, strict=True):
                if pdfmetrics.stringWidth(value, "Paper-Regular", 9.5) > width - 18:
                    raise ValueError(f"Table {number} cell exceeds available width: {value!r}")
        _, table_height = table.wrap(table_width, 1000)
        notes = Paragraph(escape(caption), ParagraphStyle(
            "notes", fontName="Paper-Regular", fontSize=9, leading=13, textColor=MUTED,
        ))
        _, note_height = notes.wrap(table_width, 1000)
        page_height = 86 + table_height + 13 + note_height + margin
        pdf = Canvas(str(folder / filename), pagesize=(table_width + 2 * margin, page_height))
        pdf.setTitle(f"V4.1 Table {number}. {title} ({selection}, {seed_label})")
        pdf.setFillColor(TEAL)
        pdf.setFont("Paper-Bold", 9)
        pdf.drawString(margin, page_height - 29, f"V4.1  /  TABLE {number}")
        pdf.setFillColor(MUTED)
        pdf.setFont("Paper-Regular", 9)
        pdf.drawRightString(margin + table_width, page_height - 29, f"{selection}  /  {seed_label}")
        pdf.setFillColor(INK)
        pdf.setFont("Paper-Bold", 18)
        pdf.drawString(margin, page_height - 60, title)
        table_bottom = page_height - 86 - table_height
        table.drawOn(pdf, margin, table_bottom)
        notes.drawOn(pdf, margin, table_bottom - 13 - note_height)
        pdf.showPage()
        pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("evaluation/v4.1"))
    parser.add_argument(
        "--font-dir", type=Path, default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
        help="Directory with arial.ttf and arialbd.ttf for PDF embedding.",
    )
    args = parser.parse_args()
    for name, font in (("Paper-Regular", "arial.ttf"), ("Paper-Bold", "arialbd.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(args.font_dir / font)))
    for directory, selection in (
        ("best_return", "Best return"), ("best_feasible", "Feasibility-selected"),
    ):
        for label, seed_label in (
            ("average_seeds", "Seeds 0-2"), ("seed0", "Seed 0"),
            ("seed1", "Seed 1"), ("seed2", "Seed 2"),
        ):
            render(args.root / directory / label, selection, seed_label)
    print("Wrote 24 table PDFs.")


if __name__ == "__main__":
    main()
