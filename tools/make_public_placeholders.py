#!/usr/bin/env python3
"""Generate neutral placeholders with the aspect ratios of withheld figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def fitted_size(width: int, height: int, limit: int = 1400) -> tuple[int, int]:
    scale = min(1.0, limit / max(width, height))
    return max(320, round(width * scale)), max(220, round(height * scale))


def font_for(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_placeholder(source: Path, destination: Path) -> None:
    with Image.open(source) as original:
        width, height = fitted_size(*original.size)
    canvas = Image.new("RGB", (width, height), "#F4F4F4")
    draw = ImageDraw.Draw(canvas)
    border = max(3, min(width, height) // 90)
    draw.rectangle(
        (border, border, width - border - 1, height - border - 1),
        outline="#858585",
        width=border,
    )
    headline = "Image omitted from public access copy"
    filename = destination.name
    title_font = font_for(max(18, min(width, height) // 12))
    file_font = font_for(max(14, min(width, height) // 18))
    title_box = draw.textbbox((0, 0), headline, font=title_font)
    file_box = draw.textbbox((0, 0), filename, font=file_font)
    title_width = title_box[2] - title_box[0]
    title_height = title_box[3] - title_box[1]
    file_width = file_box[2] - file_box[0]
    gap = max(14, min(width, height) // 20)
    top = (height - title_height - gap - (file_box[3] - file_box[1])) // 2
    draw.text(((width - title_width) // 2, top), headline, fill="#444444", font=title_font)
    draw.text(
        ((width - file_width) // 2, top + title_height + gap),
        filename,
        fill="#666666",
        font=file_font,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        canvas.save(destination, format="JPEG", quality=88, optimize=True)
    else:
        canvas.save(destination, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-figures", type=Path, required=True)
    parser.add_argument("--output-figures", type=Path, required=True)
    args = parser.parse_args()
    for source in sorted(args.source_figures.rglob("*")):
        if source.is_file() and source.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            destination = args.output_figures / source.relative_to(args.source_figures)
            make_placeholder(source, destination)


if __name__ == "__main__":
    main()
