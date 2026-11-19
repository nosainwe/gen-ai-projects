"""My optional local web interface for the same pretrained pipeline (Streamlit).

Bringing Sketches to Life with Generative AI
Author: Nosa Peter Inwe
Course: Computer Vision and Machine Learning II, Université Paris-Est Créteil (UPEC)
Professor: Arezoo Ghodsifard

I built this so I can try one sketch, prompt and strength at a time. My main
results still come from src/run_experiments.py with the fixed settings in
experiments.json. Running it locally keeps my drawings on my own machine.
"""
from __future__ import annotations
import io,json,tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
from src.sketch_pipeline import prepare_sketch,ink_fraction
from src.run_experiments import load_pipeline,load_settings

ROOT=Path(__file__).resolve().parent
SETTINGS=load_settings(ROOT/"experiments.json")
st.set_page_config(page_title="Sketch-to-image lab",layout="wide")
st.title("Bringing Sketches to Life")
st.caption("Nosa Peter Inwe · Computer Vision and Machine Learning II, UPEC · Professor Arezoo Ghodsifard")
st.write("Upload a drawing, describe the scene, pick how strongly the sketch should steer the image, and generate.")
st.info("This needs a CUDA GPU, and the first run downloads the models from Hugging Face.")

@st.cache_resource(show_spinner="Loading pretrained Stable Diffusion and Scribble ControlNet...")
def model():
    return load_pipeline(SETTINGS,allow_cpu=False)[0]

uploaded=st.file_uploader("Sketch (PNG or JPG)",type=["png","jpg","jpeg"])
prompt=st.text_area("Scene prompt","a modern car, realistic photograph, glossy red paint, parked on a city street, daylight")
strength=st.slider("Scribble guidance strength",0.0,2.0,0.65,0.05)
seed=st.number_input("Seed",min_value=0,max_value=2**31-1,value=int(SETTINGS["seed"]))
steps=st.slider("Denoising steps",10,50,int(SETTINGS["num_inference_steps"]))

if uploaded:
    try:
        raw=Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"sketch.png";raw.save(path)
            condition=prepare_sketch(path,size=int(SETTINGS["size"]),threshold=int(SETTINGS.get("threshold",205)))
        col1,col2=st.columns(2)
        col1.image(raw,caption="Input sketch",use_container_width=True)
        col2.image(condition,caption="Control map (white strokes on black)",use_container_width=True)
        frac=ink_fraction(condition)
        st.caption(f"Stroke fraction: {frac:.3f}. If shadows show up as white patches, retake the photo in even light.")
    except Exception as exc:
        st.error(f"Image could not be read: {exc}")
        st.stop()
    if st.button("Generate",type="primary"):
        if not prompt.strip():
            st.error("Enter a text prompt")
        elif not .002<=frac<=.4:
            st.error("The control map has too little or too much ink. Retake the photo or darken the lines.")
        else:
            import torch
            try:
                pipe=model()
                with st.spinner("Generating..."):
                    result=pipe(prompt=prompt,image=condition,negative_prompt=SETTINGS.get("negative_prompt"),
                        width=int(SETTINGS["size"]),height=int(SETTINGS["size"]),num_inference_steps=steps,
                        guidance_scale=float(SETTINGS["guidance_scale"]),controlnet_conditioning_scale=float(strength),
                        generator=torch.Generator(device="cpu").manual_seed(int(seed))).images[0]
                st.image(result,caption="Generated image",use_container_width=True)
                buf=io.BytesIO();result.save(buf,format="PNG")
                st.download_button("Download generated PNG",data=buf.getvalue(),file_name="my_sketch_result.png",mime="image/png")
                st.caption(f"Seed {int(seed)} · scale {float(strength):.2f} · {steps} steps")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
else:
    st.info("Upload a sketch to see its control map. The model only loads when you press Generate.")
