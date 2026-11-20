"""My web app: draw a sketch in the browser (or upload a photo of one) and generate.

Bringing Sketches to Life with Generative AI
Author: Nosa Peter Inwe
Course: Computer Vision and Machine Learning II, Université Paris-Est Créteil (UPEC)
Professor: Arezoo Ghodsifard

It runs the same pretrained pipeline as my experiments (Stable Diffusion 1.5 plus the
Scribble ControlNet) and the same preprocessing, so what I see here matches my report.
I run it on a Colab T4 with notebooks/02_web_app.ipynb, which gives me a public link.

The logic sits in SketchApp so I can test it with a fake model; Gradio only appears
inside build_demo, which keeps my unit tests free of the web framework.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .sketch_pipeline import control_map_from_image, ink_fraction

ROOT = Path(__file__).resolve().parent.parent
MIN_INK, MAX_INK = 0.002, 0.40   # same sanity bounds as my experiment runner

EXAMPLE_PROMPTS = {
    "car": "a modern car, realistic photograph, glossy red paint, chrome details, parked on a city street, daylight, high detail",
    "house": "a two-storey family house, realistic architectural photograph, brick walls, tiled roof, front garden, daylight, high detail",
    "sword": "a medieval steel longsword, realistic product photograph, polished blade, leather-wrapped grip, plain dark background, studio lighting, high detail",
}


def flatten(value) -> Image.Image | None:
    """Turn whatever Gradio hands me into a plain RGB image on white.

    The drawing canvas gives me a dict with a "composite" layer that is often
    transparent where I have not drawn; an upload gives me a PIL image or an array.
    Transparent pixels would read as black (ink) after thresholding, so I paste
    everything onto white first.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("composite")
        if value is None:
            return None
    if isinstance(value, np.ndarray):
        value = Image.fromarray(value)
    rgba = value.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    white.alpha_composite(rgba)
    return white.convert("RGB")


class SketchApp:
    """Everything my web page does, minus the web page."""

    def __init__(self, settings: dict, pipe_factory=None, generator_factory=None):
        self.s = settings
        self._pipe_factory = pipe_factory
        self._generator_factory = generator_factory
        self._pipe = None

    def pipe(self):
        # I load the models on the first Generate click, not at start-up, so the page opens fast.
        if self._pipe is None:
            if self._pipe_factory is None:
                from .run_experiments import load_pipeline
                self._pipe = load_pipeline(self.s, allow_cpu=False)[0]
            else:
                self._pipe = self._pipe_factory()
        return self._pipe

    def generator(self, seed: int):
        if self._generator_factory is not None:
            return self._generator_factory(seed)
        import torch
        return torch.Generator(device="cpu").manual_seed(seed)

    def pick_sketch(self, drawn, uploaded, threshold: int):
        """Use my drawing if the canvas has any ink on it, otherwise the upload."""
        size = int(self.s["size"])
        for source in (drawn, uploaded):
            image = flatten(source)
            if image is None:
                continue
            control = control_map_from_image(image, size, threshold)
            if ink_fraction(control) > 0:
                return image, control
        return None, None

    def preview(self, drawn, uploaded, threshold: int):
        _, control = self.pick_sketch(drawn, uploaded, int(threshold))
        if control is None:
            return None, "Draw something or upload a sketch."
        return control, self.ink_message(ink_fraction(control))

    @staticmethod
    def ink_message(frac: float) -> str:
        if frac < MIN_INK:
            return f"Ink {frac:.3f}: almost no lines. Draw bolder or raise the threshold."
        if frac > MAX_INK:
            return f"Ink {frac:.3f}: far too much. Shadows have turned into ink; lower the threshold."
        return f"Ink {frac:.3f}: looks usable."

    def run(self, drawn, uploaded, prompt: str, strength: float, seed: int, steps: int, threshold: int):
        """Return (control map, generated image, caption). Raises ValueError for anything I can fix in the page."""
        _, control = self.pick_sketch(drawn, uploaded, int(threshold))
        if control is None:
            raise ValueError("Draw something or upload a sketch first.")
        frac = ink_fraction(control)
        if not MIN_INK <= frac <= MAX_INK:
            raise ValueError(self.ink_message(frac))
        if not prompt or not prompt.strip():
            raise ValueError("Type a prompt describing the scene.")
        if not 0.0 <= float(strength) <= 2.0:
            raise ValueError("Strength must be between 0 and 2.")
        seed, steps = int(seed), int(steps)
        image = self.pipe()(
            prompt=prompt.strip(), negative_prompt=self.s.get("negative_prompt"), image=control,
            width=int(self.s["size"]), height=int(self.s["size"]),
            num_inference_steps=steps, guidance_scale=float(self.s["guidance_scale"]),
            controlnet_conditioning_scale=float(strength),
            generator=self.generator(seed)).images[0]
        caption = f"Seed {seed} · scale {float(strength):.2f} · {steps} steps · ink {frac:.3f}"
        return control, image, caption


def build_demo(app: SketchApp):
    import gradio as gr

    s = app.s

    def on_generate(drawn, uploaded, prompt, strength, seed, steps, threshold):
        try:
            return app.run(drawn, uploaded, prompt, strength, seed, steps, threshold)
        except ValueError as exc:
            raise gr.Error(str(exc))

    with gr.Blocks(title="Bringing Sketches to Life") as demo:
        gr.Markdown(
            "# Bringing Sketches to Life\n"
            "Nosa Peter Inwe · Computer Vision and Machine Learning II, UPEC · Professor Arezoo Ghodsifard\n\n"
            "Draw on the canvas or upload a photo of a pen sketch, describe the scene, and generate. "
            "Strength 0 ignores the sketch completely; around 0.65 keeps the shape and leaves the model room for detail.")
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Draw"):
                        drawn = gr.Sketchpad(
                            type="pil", label="My sketch", canvas_size=(512, 512), layers=False,
                            brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=4))
                    with gr.Tab("Upload"):
                        uploaded = gr.Image(type="pil", label="Photo of a sketch", sources=["upload", "clipboard"])
                prompt = gr.Textbox(label="Prompt", lines=3, value=EXAMPLE_PROMPTS["car"])
                with gr.Row():
                    for name, text in EXAMPLE_PROMPTS.items():
                        gr.Button(f"{name.title()} prompt", size="sm").click(lambda t=text: t, outputs=prompt)
                strength = gr.Slider(0.0, 2.0, value=float(s.get("prompt_variant_strength", 0.65)), step=0.05,
                                     label="ControlNet strength")
                with gr.Accordion("More settings", open=False):
                    seed = gr.Number(value=int(s["seed"]), precision=0, label="Seed")
                    steps = gr.Slider(10, 50, value=int(s["num_inference_steps"]), step=1, label="Denoising steps")
                    threshold = gr.Slider(100, 250, value=int(s.get("threshold", 205)), step=5,
                                          label="Ink threshold (pixels darker than this become lines)")
                go = gr.Button("Generate", variant="primary")
            with gr.Column():
                control = gr.Image(type="pil", label="Control map ControlNet sees", interactive=False)
                ink = gr.Markdown()
                result = gr.Image(type="pil", label="Generated image", interactive=False, format="png")
                info = gr.Markdown()

        inputs = [drawn, uploaded, threshold]
        uploaded.change(app.preview, inputs, [control, ink])
        threshold.release(app.preview, inputs, [control, ink])
        go.click(app.preview, inputs, [control, ink]).then(
            on_generate, [drawn, uploaded, prompt, strength, seed, steps, threshold], [control, result, info])
    return demo


def main(argv=None):
    from .run_experiments import load_settings
    parser = argparse.ArgumentParser(description="Launch my sketch-to-image web app")
    parser.add_argument("--share", action="store_true", help="make a temporary public link (I use this on Colab)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    demo = build_demo(SketchApp(load_settings(ROOT / "experiments.json")))
    demo.queue(max_size=8).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
