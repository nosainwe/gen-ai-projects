"""Sketch preprocessing, a rough structure check and comparison sheets.

Bringing Sketches to Life with Generative AI
Author: Nosa Peter Inwe
Course: Computer Vision and Machine Learning II, Université Paris-Est Créteil (UPEC)
Professor: Arezoo Ghodsifard

The Scribble ControlNet I use expects white strokes on a black background, so I
convert my pen-on-paper photos into that format here. I only transform drawings I
supply; nothing in this file makes up a sketch.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageFont
import numpy as np


def prepare_sketch(path: str | Path, size: int = 512, threshold: int = 205) -> Image.Image:
    """Turn one photographed sketch into a square white-on-black control map.

    I pad the photo onto a white square instead of stretching it, so the car or
    sword keeps its proportions. Any pixel darker than `threshold` counts as ink.
    """
    if size < 64 or size % 8:
        # Stable Diffusion's VAE downsamples by 8, so the side must divide by 8.
        raise ValueError("size must be >=64 and divisible by 8")
    if not 1 <= threshold <= 254:
        raise ValueError("threshold must be in [1, 254]")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Sketch not found: {path}")
    with Image.open(path) as raw:
        # Phone photos often carry a rotation flag; I apply it before anything else.
        image = ImageOps.exif_transpose(raw).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    grey = np.asarray(ImageOps.grayscale(canvas), dtype=np.uint8)
    strokes = (grey < threshold).astype(np.uint8) * 255   # dark ink -> white stroke
    return Image.fromarray(strokes, "L").convert("RGB")


def ink_fraction(control: Image.Image) -> float:
    """Share of white pixels in a control map.

    I use it as a sanity check on the photo, not as a quality score: a value near 0
    means my lines vanished, and a high value usually means a shadow became "ink".
    """
    return float(np.mean(np.asarray(control.convert("L")) > 127))


def sketch_edge_recall(control: Image.Image, generated: Image.Image,
                       tolerance_pixels: int = 5, edge_threshold: int = 65) -> float:
    """Fraction of my sketch strokes that land within a few pixels of an edge in the output.

    It's a rough helper for comparing strengths, nothing more. A busy texture
    (brick, foliage) creates edges everywhere and can push the number up without
    the shape being right, so my own visual ratings carry the real judgement.
    """
    if not 1 <= tolerance_pixels <= 20:
        raise ValueError("tolerance_pixels must be 1..20")
    if not 1 <= edge_threshold <= 254:
        raise ValueError("edge_threshold must be 1..254")
    size = control.size
    target = ImageOps.grayscale(generated.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    found = target.filter(ImageFilter.FIND_EDGES)
    edges = Image.fromarray((np.asarray(found) >= edge_threshold).astype(np.uint8) * 255, "L")
    # A max filter grows each edge pixel, which gives me the "near an edge" tolerance.
    nearby = edges.filter(ImageFilter.MaxFilter(2 * tolerance_pixels + 1))
    strokes = np.asarray(control.convert("L")) > 127
    if strokes.sum() == 0:
        return 0.0
    return float(np.mean((np.asarray(nearby) > 127)[strokes]))


def make_comparison(sketch_path: str | Path, control: Image.Image,
                    images: list[tuple[str, Image.Image]], out: str | Path) -> None:
    """Save my original sketch, its control map and the generated images in one labelled row."""
    with Image.open(sketch_path) as raw:
        original = ImageOps.exif_transpose(raw).convert("RGB")
    cells = [original, control.convert("RGB")] + [im.convert("RGB") for _, im in images]
    labels = ["My hand sketch", "Scribble control map"] + [label for label, _ in images]
    tile, top = 360, 52
    sheet = Image.new("RGB", (tile * len(cells), tile + top), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=22)   # Pillow >= 10.1; large enough to read in my report
    except TypeError:
        font = ImageFont.load_default()
    for ix, (im, label) in enumerate(zip(cells, labels)):
        im = ImageOps.contain(im, (tile - 20, tile - 20), Image.Resampling.LANCZOS)
        sheet.paste(im, (ix * tile + (tile - im.width) // 2, top + (tile - im.height) // 2))
        draw.text((ix * tile + 10, 14), label, fill="black", font=font)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
