"""Starts my sketch-to-image web app (Gradio). The app itself lives in src/gradio_app.py.

Bringing Sketches to Life with Generative AI
Author: Nosa Peter Inwe
Course: Computer Vision and Machine Learning II, Université Paris-Est Créteil (UPEC)
Professor: Arezoo Ghodsifard

    python app.py            # local, needs a CUDA GPU
    python app.py --share    # also prints a temporary public link
"""
from src.gradio_app import main

if __name__ == "__main__":
    main()
