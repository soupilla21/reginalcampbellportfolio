#!/usr/bin/env python3
"""
Generate a 1200x630 social card for a governance artifact.

Distinct from the article card variant (see public/assets/articles/*-card.jpg):
amber accent bar and "GOVERNANCE ARTIFACT" eyebrow badge instead of the plain
white bar, plus a small corner marker, so artifact cards are recognizable
against article cards in a LinkedIn feed. Same dark gradient background and
Inter/JetBrains Mono typography as the article pipeline for site consistency.

Pipeline: fill SVG template -> rsvg-convert to PNG -> PIL to JPEG.
Mirrors the established article card pipeline (rsvg-convert -> PIL -> JPEG).

Usage:
    python3 scripts/generate-artifact-card.py \
        --title "Pipeline Verification Placeholder" \
        --description "Throwaway artifact used to verify the Phase A/B/C pipeline." \
        --version "0.1" \
        --eyebrow-suffix "" \
        --out public/assets/artifacts/pipeline-verification-placeholder-card.jpg
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

TEMPLATE_PATH = Path(__file__).parent / "social-cards" / "artifact-card-template.svg"

TITLE_FONT_SIZE = 54
TITLE_LINE_HEIGHT = 66
TITLE_START_Y = 236
TITLE_MAX_CHARS_PER_LINE = 20
TITLE_MAX_LINES = 3
DESC_MAX_CHARS = 78


def wrap_title(title: str) -> list[str]:
    lines = textwrap.wrap(title, width=TITLE_MAX_CHARS_PER_LINE, break_long_words=False)
    if len(lines) > TITLE_MAX_LINES:
        lines = lines[:TITLE_MAX_LINES]
        lines[-1] = lines[-1].rstrip() + "…"
    return lines


def build_title_svg(lines: list[str]) -> str:
    # First line gets the lightest weight to echo the article card's
    # two-tone headline treatment; the rest are full white bold.
    parts = []
    for i, line in enumerate(lines):
        y = TITLE_START_Y + i * TITLE_LINE_HEIGHT
        fill = "#e5e5e5" if i == 0 and len(lines) > 1 else "#ffffff"
        parts.append(
            f'<text x="80" y="{y}" font-family="Inter" font-weight="800" '
            f'font-size="{TITLE_FONT_SIZE}" fill="{fill}">{escape(line)}</text>'
        )
    return "\n  ".join(parts)


def truncate_description(description: str) -> str:
    if len(description) <= DESC_MAX_CHARS:
        return description
    cut = description[: DESC_MAX_CHARS - 1].rstrip().rstrip(".")
    return cut + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--eyebrow-suffix", default="", help="Optional short suffix after the eyebrow badge, e.g. '· RISK ASSESSMENT'")
    parser.add_argument("--out", required=True, help="Output JPEG path")
    args = parser.parse_args()

    template = TEMPLATE_PATH.read_text()

    title_lines = wrap_title(args.title)
    title_svg = build_title_svg(title_lines)
    desc_y = TITLE_START_Y + len(title_lines) * TITLE_LINE_HEIGHT + 44

    svg = template
    svg = svg.replace("__TITLE_LINES__", title_svg)
    svg = svg.replace("__DESC_Y__", str(desc_y))
    svg = svg.replace("__DESCRIPTION__", escape(truncate_description(args.description)))
    svg = svg.replace("__VERSION__", escape(args.version))
    svg = svg.replace("__EYEBROW_SUFFIX__", escape(args.eyebrow_suffix))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_svg = out_path.with_suffix(".tmp.svg")
    tmp_png = out_path.with_suffix(".tmp.png")
    tmp_svg.write_text(svg)

    try:
        subprocess.run(
            ["rsvg-convert", "-w", "1200", "-h", "630", "-o", str(tmp_png), str(tmp_svg)],
            check=True,
        )
        img = Image.open(tmp_png).convert("RGB")
        img.save(out_path, "JPEG", quality=90)
    finally:
        tmp_svg.unlink(missing_ok=True)
        tmp_png.unlink(missing_ok=True)

    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
