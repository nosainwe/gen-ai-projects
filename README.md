# Generative AI Projects

**Author:** Nosa Peter Inwe

This is where I keep my generative AI projects. Each one lives in its own folder with its own README, code, requirements and results, so I can run any project without touching the others.

## Projects

| Project | What it does | Stack |
|---|---|---|
| [Bringing Sketches to Life](sketch-to-image-controlnet/) | Turns my hand-drawn sketches (a car, a house and a sword) into finished images, and tests how control strength and prompt wording affect how much of each drawing survives. Built for Computer Vision and Machine Learning II at UPEC, with Professor Arezoo Ghodsifard. | Stable Diffusion 1.5, ControlNet v1.1 Scribble, Diffusers, Google Colab |

## Layout

```
gen-ai-projects/
├── README.md                      this page
├── .github/workflows/tests.yml    runs each project's tests on every push
└── sketch-to-image-controlnet/    one folder per project
```

When I add a project, I give it its own folder, add a row to the table above and add its tests to the workflow.
