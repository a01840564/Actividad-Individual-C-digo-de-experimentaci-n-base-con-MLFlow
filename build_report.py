"""Exporta el informe Markdown a PDF con Arial 11 e interlineado 1.5."""

import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "TC5061_Experimentacion_MLFlow_HernandezAndres.md"


def inline(text):
    pattern = r"(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))"
    parts = []
    for part in re.split(pattern, text):
        if part.startswith("`") and part.endswith("`"):
            parts.append(escape(part[1:-1]))
        elif part.startswith("**") and part.endswith("**"):
            parts.append(f"<b>{escape(part[2:-2])}</b>")
        elif match := re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part):
            label, url = match.groups()
            parts.append(f'<link href="{escape(url, quote=True)}" color="#184c72">{escape(label)}</link>')
        else:
            parts.append(escape(part))
    return "".join(parts)


def paragraph_text(lines):
    result = ""
    for index, line in enumerate(lines):
        result += inline(line.strip())
        if index < len(lines) - 1:
            result += "<br/>" if line.endswith("  ") else " "
    return result


def make_table(lines, style):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append([Paragraph(inline(cell), style) for cell in cells])
    widths = [105, 30, 85, 65, 72, 55, 74] if len(rows[0]) == 7 else None
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf4")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#8ca3b7")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for index, line in enumerate(lines[2:], 1):
        if "05_balanced" in line:
            commands.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor("#e8f3eb")))
    table.setStyle(TableStyle(commands))
    table.spaceAfter = 5
    return table


def build_story(markdown, styles):
    lines = re.sub(r"<!--[\s\S]*?-->", "", markdown.read_text()).splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if "page-break-after" in line:
            story.append(PageBreak())
            index += 1
        elif line.startswith("#"):
            story.append(Paragraph(inline(line.lstrip("# ")), styles["heading"]))
            index += 1
        elif line.startswith("```"):
            index += 1
            block = []
            while index < len(lines) and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            story.append(Paragraph("<br/>".join(escape(item) for item in block), styles["body"]))
            index += 1
        elif line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].startswith("|"):
                block.append(lines[index])
                index += 1
            story.append(make_table(block, styles["cell"]))
        elif match := re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip()):
            path = markdown.parent / match.group(2)
            if not path.is_file():
                raise FileNotFoundError(path)
            picture = Image(str(path))
            scale = min(510 / picture.imageWidth, 555 / picture.imageHeight)
            picture.drawWidth = picture.imageWidth * scale
            picture.drawHeight = picture.imageHeight * scale
            story.append(picture)
            index += 1
        else:
            block = []
            while index < len(lines) and lines[index].strip():
                if lines[index].startswith(("#", "```", "|", "![", "<div")):
                    break
                block.append(lines[index])
                index += 1
            story.append(Paragraph(paragraph_text(block), styles["body"]))
    return story


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/TC5061_Experimentacion_MLFlow_HernandezAndres.pdf")
    parser.add_argument("--font-dir", type=Path, default=Path("/System/Library/Fonts/Supplemental"))
    args = parser.parse_args()
    for label, filename in [("Arial", "Arial.ttf"), ("Arial-Bold", "Arial Bold.ttf")]:
        path = args.font_dir / filename
        if not path.is_file():
            parser.error(f"No se encuentra {path}. Indica la carpeta de Arial con --font-dir.")
        pdfmetrics.registerFont(TTFont(label, str(path)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial", boldItalic="Arial-Bold")
    styles = {
        "body": ParagraphStyle("body", fontName="Arial", fontSize=11, leading=16.5, spaceAfter=5),
        "heading": ParagraphStyle("heading", fontName="Arial-Bold", fontSize=11, leading=16.5,
                                  spaceBefore=5, spaceAfter=5, keepWithNext=True),
        "cell": ParagraphStyle("cell", fontName="Arial", fontSize=11, leading=16.5),
    }
    def footer(canvas, document):
        canvas.setFont("Arial", 11)
        canvas.setFillColor(colors.HexColor("#526476"))
        canvas.drawString(50, 24, "TC5061 | Experimentación con MLflow")
        canvas.drawRightString(562, 24, str(document.page))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(args.output), pagesize=letter,
                                 leftMargin=50, rightMargin=50, topMargin=35, bottomMargin=40,
                                 title="TC5061 - Experimentación con MLflow",
                                 author="José Andrés Hernández Guerra")
    document.build(build_story(args.markdown.resolve(), styles),
                   onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
