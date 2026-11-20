"""My experiment runner: sketch -> Scribble ControlNet + Stable Diffusion 1.5 -> images.

Bringing Sketches to Life with Generative AI
Author: Nosa Peter Inwe
Course: Computer Vision and Machine Learning II, Université Paris-Est Créteil (UPEC)
Professor: Arezoo Ghodsifard

I run two experiments per sketch, always with the same seed:
  1. Control-strength ablation: one prompt at ControlNet scales 0.00, 0.65 and 1.10.
     At 0.00 the sketch has no influence, so that image is my prompt-only baseline.
  2. Prompt variation: two extra prompts at the shared scale 0.65, so I can see
     whether the structure survives a change of style, material or setting.

Usage (from the project root):
  python -m src.run_experiments --dry-run    # checks my sketches, saves control maps, no download
  python -m src.run_experiments --generate   # real inference on a CUDA GPU
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from src.sketch_pipeline import prepare_sketch, ink_fraction, sketch_edge_recall, make_comparison


def load_settings(path: Path) -> dict:
    """Read experiments.json and stop early on anything that would spoil the comparison."""
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("model_base", "controlnet_model", "seed", "size", "num_inference_steps",
                "guidance_scale", "experiments")
    for key in required:
        if key not in s:
            raise ValueError(f"Missing configuration: {key}")
    if not isinstance(s["experiments"], list) or len(s["experiments"]) < 3:
        raise ValueError("I need at least THREE distinct sketch entries")
    if int(s["size"]) < 64 or int(s["size"]) % 8:
        raise ValueError("size must be >=64 and divisible by 8")
    if int(s["num_inference_steps"]) < 1:
        raise ValueError("num_inference_steps must be positive")
    if float(s["guidance_scale"]) < 0:
        raise ValueError("guidance_scale must not be negative")
    if not 0 <= float(s.get("prompt_variant_strength", 0.65)) <= 2:
        raise ValueError("prompt_variant_strength must be 0..2")
    if len({e.get("sketch") for e in s["experiments"]}) != len(s["experiments"]):
        raise ValueError("Use a different input file for each sketch entry")
    for e in s["experiments"]:
        if not e.get("sketch") or not e.get("prompt") or not e.get("control_strengths"):
            raise ValueError("Each experiment needs sketch, prompt and control_strengths")
        if Path(e["sketch"]).name != e["sketch"] or not e["sketch"].lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError("Use a PNG/JPG filename, not a path, in sketch")
        strengths = [float(v) for v in e["control_strengths"]]
        if not all(0 <= v <= 2 for v in strengths):
            raise ValueError("Control strength must be 0..2")
        if len(set(strengths)) != len(strengths):
            raise ValueError("Repeated control strength")
        variants = e.get("prompt_variants", [])
        names = [v.get("name") for v in variants]
        if any(not n or not str(n).isidentifier() for n in names) or len(set(names)) != len(names) or "main" in names:
            raise ValueError("Each prompt variant needs a unique simple name (letters, digits, _), not 'main'")
        if any(not v.get("prompt") for v in variants):
            raise ValueError("Each prompt variant needs a prompt")
    return s


def sha256(path: Path) -> str:
    """Fingerprint of my input file, so anyone can confirm which sketch produced which image."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_inputs(s: dict, root: Path, out: Path) -> dict:
    """Build and save a control map for every sketch; refuse to continue if one is missing."""
    out.mkdir(parents=True, exist_ok=True)
    controls, missing = {}, []
    for e in s["experiments"]:
        p = root / "data" / "sketches" / e["sketch"]
        if not p.is_file():
            missing.append(e["sketch"])
            continue
        im = prepare_sketch(p, int(s["size"]), int(s.get("threshold", 205)))
        frac = ink_fraction(im)
        if not .002 <= frac <= .40:
            # Almost no ink, or a shadow flooding the page: fix the photo or threshold first.
            raise ValueError(f"{e['sketch']}: stroke fraction {frac:.4f} outside 0.002..0.40; "
                             "check the photograph or the threshold")
        control_path = out / (Path(e["sketch"]).stem + "_control.png")
        im.save(control_path)
        controls[e["sketch"]] = (p, im, frac)
        print(f"Ready: {p.name}; ink fraction={frac:.3f}; control={control_path.name}")
    if missing:
        raise FileNotFoundError("Missing hand sketches: " + ", ".join(missing))
    return controls


def load_pipeline(s: dict, allow_cpu: bool = False):
    """Load the pretrained models from Hugging Face. I do not train anything from scratch."""
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not allow_cpu:
        raise RuntimeError("No CUDA GPU found. Use a GPU (e.g. a Colab T4); --allow-cpu works but is very slow")
    dtype = torch.float16 if has_cuda else torch.float32
    controlnet = ControlNetModel.from_pretrained(s["controlnet_model"], torch_dtype=dtype, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        s["model_base"], controlnet=controlnet, torch_dtype=dtype, use_safetensors=True)
    # UniPC gets a clean image in ~25 steps, which keeps the whole run short.
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if has_cuda:
        pipe.enable_model_cpu_offload()   # keeps only the active sub-model on the GPU
    else:
        pipe.to("cpu")
    return pipe, ("cuda" if has_cuda else "cpu")


def packages() -> dict:
    """Exact library versions for my reproducibility record."""
    ans = {"python": platform.python_version()}
    for p in ("torch", "diffusers", "transformers", "accelerate", "Pillow", "numpy"):
        try:
            ans[p] = version(p)
        except PackageNotFoundError:
            ans[p] = "not installed"
    return ans


def export_rows(rows: list[dict], out: Path) -> None:
    """Write the manifest, a summary CSV and a blank sheet for my own ratings."""
    (out / "run_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = ["sketch", "experiment", "prompt_variant", "control_strength", "edge_recall_proxy", "result",
            "prompt", "seed", "steps", "guidance_scale", "model_base", "controlnet_model"]
    with (out / "run_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (out / "human_review_template.csv").open("w", newline="", encoding="utf-8") as f:
        keys2 = ["sketch", "experiment", "prompt_variant", "control_strength", "structure_1_to_5",
                 "image_quality_1_to_5", "prompt_fit_1_to_5", "specific_strength", "specific_failure"]
        w = csv.DictWriter(f, fieldnames=keys2, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_generation(s: dict, root: Path, out: Path, controls: dict, pipe, device: str, generator_factory) -> list[dict]:
    """Generate every image for both experiments and save comparison sheets as I go."""
    rows = []
    seed = int(s["seed"])
    variant_strength = float(s.get("prompt_variant_strength", 0.65))

    def generate(e, control, prompt, strength):
        # A fresh generator with the same seed gives every run the same starting noise,
        # so the only differences come from the setting I am testing.
        return pipe(prompt=prompt, negative_prompt=s.get("negative_prompt"), image=control,
                    width=int(s["size"]), height=int(s["size"]),
                    num_inference_steps=int(s["num_inference_steps"]),
                    guidance_scale=float(s["guidance_scale"]),
                    controlnet_conditioning_scale=strength,
                    generator=generator_factory(seed)).images[0]

    def record(e, sketch, control, frac, image, name, experiment, variant, prompt, strength):
        image.save(out / name)
        score = sketch_edge_recall(control, image, int(s.get("edge_tolerance_pixels", 5)))
        rows.append({"sketch": e["sketch"], "experiment": experiment, "prompt_variant": variant,
                     "result": f"outputs/{name}", "control_strength": strength,
                     "edge_recall_proxy": round(score, 4), "prompt": prompt,
                     "negative_prompt": s.get("negative_prompt"), "seed": seed,
                     "steps": int(s["num_inference_steps"]), "guidance_scale": float(s["guidance_scale"]),
                     "threshold": int(s.get("threshold", 205)), "ink_fraction": round(frac, 5),
                     "source_sha256": sha256(sketch), "model_base": s["model_base"],
                     "controlnet_model": s["controlnet_model"], "device": device,
                     "created_utc": datetime.now(timezone.utc).isoformat()})
        export_rows(rows, out)   # I rewrite the records after every image, so a crash loses nothing
        print("Saved", name, "edge proxy", f"{score:.3f}")

    for e in s["experiments"]:
        sketch, control, frac = controls[e["sketch"]]
        stem = Path(e["sketch"]).stem

        # Experiment 1: same prompt, three control strengths.
        cells, main_at_variant_strength = [], None
        for strength in [float(v) for v in e["control_strengths"]]:
            image = generate(e, control, e["prompt"], strength)
            record(e, sketch, control, frac, image, f"{stem}_control_{strength:.2f}.png",
                   "control_strength", "main", e["prompt"], strength)
            cells.append((f"Scale {strength:.2f}" if strength else "Baseline (no control)", image))
            if abs(strength - variant_strength) < 1e-9:
                main_at_variant_strength = image
        make_comparison(sketch, control, cells, out / f"{stem}_comparison.png")

        # Experiment 2: same sketch, seed and strength, different prompts.
        variants = e.get("prompt_variants", [])
        if variants:
            if main_at_variant_strength is None:
                main_at_variant_strength = generate(e, control, e["prompt"], variant_strength)
                record(e, sketch, control, frac, main_at_variant_strength,
                       f"{stem}_prompt_main.png", "prompt", "main", e["prompt"], variant_strength)
            prompt_cells = [(f"Main prompt ({variant_strength:.2f})", main_at_variant_strength)]
            for v in variants:
                image = generate(e, control, v["prompt"], variant_strength)
                record(e, sketch, control, frac, image, f"{stem}_prompt_{v['name']}.png",
                       "prompt", v["name"], v["prompt"], variant_strength)
                prompt_cells.append((f"Prompt: {v['name']}", image))
            make_comparison(sketch, control, prompt_cells, out / f"{stem}_prompts.png")
    return rows


def main():
    ap = argparse.ArgumentParser(description="My sketch-to-image experiments")
    ap.add_argument("--config", type=Path, default=Path("experiments.json"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Check my sketches and save control maps; no download")
    mode.add_argument("--generate", action="store_true", help="Run real inference with the pretrained models")
    ap.add_argument("--allow-cpu", action="store_true", help="CPU generation works but is very slow")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    s = load_settings(config)
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    controls = prepare_inputs(s, root, out)
    if args.dry_run:
        print("Dry run only: no models loaded and no images generated.")
        return
    pipe, device = load_pipeline(s, args.allow_cpu)
    import torch
    (out / "environment.json").write_text(json.dumps({
        "packages": packages(), "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "configuration": s}, indent=2), encoding="utf-8")
    rows = run_generation(s, root, out, controls, pipe, device,
                          lambda seed: torch.Generator(device="cpu").manual_seed(seed))
    print(f"Finished {len(rows)} images. Next: look at them and fill in human_review_template.csv.")


if __name__ == "__main__":
    main()
