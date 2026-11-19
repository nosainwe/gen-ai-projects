"""My offline tests: they check preprocessing, config validation and bookkeeping.

Bringing Sketches to Life with Generative AI
Author: Nosa Peter Inwe
Course: Computer Vision and Machine Learning II, Université Paris-Est Créteil (UPEC)
Professor: Arezoo Ghodsifard

None of these tests downloads a model. Where I need a "pipeline" I use a fake one
that returns a flat grey image, so the tests prove the code flow works, not that
the real model draws good cars.
"""
from pathlib import Path
import csv, json
from PIL import Image, ImageDraw
import pytest
from src.sketch_pipeline import prepare_sketch, ink_fraction, make_comparison, sketch_edge_recall
from src.run_experiments import load_settings, prepare_inputs, run_generation

CONFIG = Path(__file__).resolve().parents[1] / "experiments.json"


def drawing(p):
    """A simple stand-in drawing (a rectangle) for tests only, never a submitted sketch."""
    im = Image.new("RGB", (150, 100), "white")
    ImageDraw.Draw(im).rectangle((20, 20, 130, 85), outline="black", width=4)
    im.save(p)
    return p


class FakePipe:
    """Records every call and returns a grey image instead of running diffusion."""
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"images": [Image.new("RGB", (512, 512), (95, 118, 154))]})()


def write_config(tmp_path, s):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(s))
    return p


def test_controls_resize_and_strokes(tmp_path):
    im = prepare_sketch(drawing(tmp_path / "a.png"), 128)
    assert im.size == (128, 128) and im.mode == "RGB"
    assert .003 < ink_fraction(im) < .30


def test_blank_returns_zero(tmp_path):
    p = tmp_path / "b.png"
    Image.new("RGB", (100, 100), "white").save(p)
    assert ink_fraction(prepare_sketch(p, 128)) == 0


@pytest.mark.parametrize("bad", [10, 127, 130])
def test_invalid_resolution(tmp_path, bad):
    with pytest.raises(ValueError):
        prepare_sketch(drawing(tmp_path / "a.png"), bad)


@pytest.mark.parametrize("th", [0, 255])
def test_invalid_threshold(tmp_path, th):
    with pytest.raises(ValueError):
        prepare_sketch(drawing(tmp_path / "a.png"), 128, th)


def test_missing_sketch(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_sketch(tmp_path / "no.png", 128)


def test_make_sheet(tmp_path):
    p = drawing(tmp_path / "a.png")
    c = prepare_sketch(p, 128)
    o = tmp_path / "sheet.png"
    make_comparison(p, c, [("a", c)], o)
    assert Image.open(o).size == (1080, 412)


def test_metric_no_ink():
    blank = Image.new("RGB", (128, 128), "black")
    assert sketch_edge_recall(blank, blank) == 0


def test_metric_close_to_edge(tmp_path):
    c = prepare_sketch(drawing(tmp_path / "a.png"), 128)
    # White strokes on black have clear edges, so a map compared with itself scores well.
    assert sketch_edge_recall(c, c) > 0.1
    with pytest.raises(ValueError):
        sketch_edge_recall(c, c, 0)


def test_config_valid():
    s = load_settings(CONFIG)
    assert [e["label"] for e in s["experiments"]] == ["car", "house", "sword"]
    assert all(x["control_strengths"] == [0.0, 0.65, 1.1] for x in s["experiments"])
    assert all(len(x["prompt_variants"]) == 2 for x in s["experiments"])


def test_duplicate_input_rejected(tmp_path):
    s = json.loads(CONFIG.read_text())
    s["experiments"][1]["sketch"] = s["experiments"][0]["sketch"]
    with pytest.raises(ValueError):
        load_settings(write_config(tmp_path, s))


@pytest.mark.parametrize("bad", ["main", "has space", ""])
def test_bad_variant_name_rejected(tmp_path, bad):
    s = json.loads(CONFIG.read_text())
    s["experiments"][0]["prompt_variants"][0]["name"] = bad
    with pytest.raises(ValueError):
        load_settings(write_config(tmp_path, s))


def test_missing_drawings_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_inputs(load_settings(CONFIG), tmp_path, tmp_path / "outputs")


def setup_run(tmp_path, s):
    dat = tmp_path / "data" / "sketches"
    dat.mkdir(parents=True)
    for e in s["experiments"]:
        drawing(dat / e["sketch"])
    out = tmp_path / "outputs"
    return out, prepare_inputs(s, tmp_path, out)


def test_full_run_with_fake_pipe(tmp_path):
    s = load_settings(CONFIG)
    out, controls = setup_run(tmp_path, s)
    pipe = FakePipe()
    rows = run_generation(s, tmp_path, out, controls, pipe, "test-stub", lambda seed: seed)
    # 3 sketches x (3 strengths + 2 prompt variants) = 15; the main prompt at 0.65 is reused.
    assert len(rows) == 15 and len(pipe.calls) == 15
    assert [c["controlnet_conditioning_scale"] for c in pipe.calls[:5]] == [0.0, .65, 1.1, .65, .65]
    assert all(c["generator"] == s["seed"] for c in pipe.calls)
    assert pipe.calls[3]["prompt"] == s["experiments"][0]["prompt_variants"][0]["prompt"]
    for e in s["experiments"]:
        stem = Path(e["sketch"]).stem
        assert (out / f"{stem}_comparison.png").is_file()
        assert (out / f"{stem}_prompts.png").is_file()
    assert len(json.loads((out / "run_manifest.json").read_text())) == 15
    review = list(csv.DictReader((out / "human_review_template.csv").open()))
    assert len(review) == 15 and {r["experiment"] for r in review} == {"control_strength", "prompt"}


def test_variant_strength_outside_ablation_adds_main_run(tmp_path):
    s = load_settings(CONFIG)
    s["prompt_variant_strength"] = 0.9   # not one of the ablation strengths
    out, controls = setup_run(tmp_path, s)
    rows = run_generation(s, tmp_path, out, controls, FakePipe(), "test-stub", lambda seed: seed)
    assert len(rows) == 18   # one extra main-prompt image per sketch at 0.9


def test_no_variants_still_works(tmp_path):
    s = load_settings(CONFIG)
    for e in s["experiments"]:
        e.pop("prompt_variants")
    out, controls = setup_run(tmp_path, s)
    rows = run_generation(s, tmp_path, out, controls, FakePipe(), "test-stub", lambda seed: seed)
    assert len(rows) == 9 and not list(out.glob("*_prompts.png"))
