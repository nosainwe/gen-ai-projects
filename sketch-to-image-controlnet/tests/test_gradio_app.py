"""Tests for my web app's logic, with a fake model. Gradio itself is not needed here.

Author: Nosa Peter Inwe · Computer Vision and Machine Learning II, UPEC · Professor Arezoo Ghodsifard
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pytest

from src.gradio_app import SketchApp, flatten
from src.run_experiments import load_settings
from src.sketch_pipeline import control_map_from_image, prepare_sketch

CONFIG = Path(__file__).resolve().parents[1] / "experiments.json"


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"images": [Image.new("RGB", (512, 512), (95, 118, 154))]})()


def make_app():
    pipe = FakePipe()
    app = SketchApp(load_settings(CONFIG), pipe_factory=lambda: pipe, generator_factory=lambda seed: ("gen", seed))
    return app, pipe


def canvas_drawing(with_line=True):
    """What the Gradio sketchpad sends: transparent everywhere except my strokes."""
    layer = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    if with_line:
        ImageDraw.Draw(layer).line((80, 256, 430, 256), fill=(0, 0, 0, 255), width=6)
    return {"background": None, "layers": [layer], "composite": layer}


def paper_photo():
    img = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(img).rectangle((150, 120, 490, 360), outline="black", width=5)
    return img


def test_flatten_puts_transparent_canvas_on_white():
    flat = flatten(canvas_drawing(with_line=False))
    assert flat.mode == "RGB" and flat.getpixel((10, 10)) == (255, 255, 255)


def test_flatten_accepts_arrays_and_none():
    assert flatten(None) is None
    assert flatten({"composite": None}) is None
    assert flatten(np.full((32, 32, 3), 255, np.uint8)).size == (32, 32)


def test_in_memory_map_matches_file_version(tmp_path):
    path = tmp_path / "s.png"
    paper_photo().save(path)
    a = np.asarray(prepare_sketch(path))
    b = np.asarray(control_map_from_image(paper_photo()))
    assert (a == b).all()


def test_empty_canvas_falls_back_to_upload():
    app, _ = make_app()
    image, control = app.pick_sketch(canvas_drawing(with_line=False), paper_photo(), 205)
    assert image.size == (640, 480) and control.size == (512, 512)


def test_drawing_wins_over_upload():
    app, _ = make_app()
    image, _ = app.pick_sketch(canvas_drawing(), paper_photo(), 205)
    assert image.size == (512, 512)


def test_run_passes_my_settings_to_the_model():
    app, pipe = make_app()
    control, image, caption = app.run(canvas_drawing(), None, "  a red car  ", 0.65, 7, 20, 205)
    call = pipe.calls[0]
    assert call["prompt"] == "a red car"
    assert call["controlnet_conditioning_scale"] == pytest.approx(0.65)
    assert call["num_inference_steps"] == 20 and call["generator"] == ("gen", 7)
    assert call["image"] is control and image.size == (512, 512)
    assert "scale 0.65" in caption


def test_model_loads_once():
    loads = []
    app = SketchApp(load_settings(CONFIG), pipe_factory=lambda: loads.append(1) or FakePipe(),
                    generator_factory=lambda s: None)
    for _ in range(2):
        app.run(canvas_drawing(), None, "a car", 0.65, 1, 10, 205)
    assert loads == [1]


@pytest.mark.parametrize("drawn, prompt, strength, message", [
    (None, "a car", 0.65, "Draw something"),
    ("line", "   ", 0.65, "prompt"),
    ("line", "a car", 2.5, "between 0 and 2"),
])
def test_run_rejects_bad_input(drawn, prompt, strength, message):
    app, pipe = make_app()
    sketch = canvas_drawing() if drawn else None
    with pytest.raises(ValueError, match=message):
        app.run(sketch, None, prompt, strength, 1, 10, 205)
    assert pipe.calls == []


def test_run_rejects_a_flooded_control_map():
    app, pipe = make_app()
    dark = Image.new("RGB", (512, 512), (40, 40, 40))   # a shadowy photo: everything reads as ink
    with pytest.raises(ValueError, match="too much"):
        app.run(None, dark, "a car", 0.65, 1, 10, 205)
    assert pipe.calls == []


def test_preview_messages():
    app, _ = make_app()
    control, msg = app.preview(canvas_drawing(), None, 205)
    assert control is not None and "usable" in msg
    assert app.preview(None, None, 205) == (None, "Draw something or upload a sketch.")
