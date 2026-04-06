"""
Diffusion model loading and image generation (§4.2)

Paper setup:
  - Model: Stable Diffusion v1.4 (890M params, text-conditioned)
  - Sampler: PLMS (§4.2.1: "we use the default PLMS sampling scheme")
  - Resolution: 512×512 (§4.2)
  - Generations: 500 per prompt (§4.2.1: "generate 500 candidate images")

For CIFAR-10 experiments (§5):
  - Model: OpenAI improved-diffusion (unconditional + class-conditional)
  - Resolution: 32×32
  - FID target: ≤ 3.5 (unconditional), ≤ 4.0 (class-conditional)
  - Per-model generations: 2^16 = 65,536 images
  - Number of models: 16 (each trained on a random 50% split)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline, PNDMScheduler


# ---------------------------------------------------------------------------
# Stable Diffusion (§4.2) — black-box attack target
# ---------------------------------------------------------------------------

def load_stable_diffusion(
    model_id: str = "CompVis/stable-diffusion-v1-4",  # § SD v1.4 (not v1.5)
    use_plms: bool = True,                             # § "default PLMS sampling"
    device: Optional[str] = None,
) -> StableDiffusionPipeline:
    """
    Load Stable Diffusion with the PLMS scheduler as used in the paper.

    Paper §4.2: "We generate from the model using the default PLMS sampling
    scheme at a resolution of 512×512 pixels."

    Note: 'runwayml/stable-diffusion-v1-5' is a later version and was NOT
    used in the paper. Use 'CompVis/stable-diffusion-v1-4'.
    """
    if device is None:
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if use_plms:
        # PLMS sampler = PNDMScheduler in diffusers
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    pipe.safety_checker = None       # disable for research use — no NSFW content in prompts
    return pipe


def generate_images_for_prompt(
    pipe: StableDiffusionPipeline,
    prompt: str,
    output_dir: Path,
    num_images: int = 500,           # § "generate 500 candidate images for each prompt"
    image_size: int = 512,           # § "resolution of 512×512"
    guidance_scale: float = 7.5,
    batch_size: int = 4,
) -> List[Path]:
    """
    Generate num_images for a single prompt and save them to output_dir.

    Returns list of saved paths.
    """
    safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:80]
    prompt_dir = output_dir / safe_prompt
    prompt_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    generated = 0

    while generated < num_images:
        batch = min(batch_size, num_images - generated)
        images = pipe(
            [prompt] * batch,
            height=image_size,
            width=image_size,
            guidance_scale=guidance_scale,
        ).images

        for i, img in enumerate(images):
            out_path = prompt_dir / f"{generated + i:04d}.png"
            img.save(out_path)
            saved.append(out_path)

        generated += batch
        if generated % 50 == 0:
            print(f"  Generated {generated}/{num_images} for '{prompt}'")

    return saved


def generate_all_prompts(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    output_dir: Path,
    num_images_per_prompt: int = 500,
) -> None:
    """
    Run the generation stage for all prompts.

    Paper §4.2: generate 500 images per prompt using the top 350,000
    most-duplicated captions from LAION. For research on smaller datasets,
    provide a list of prompts corresponding to known training captions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating for: {prompt}")
        generate_images_for_prompt(pipe, prompt, output_dir, num_images_per_prompt)
    print("\n✅ All images generated.")


# ---------------------------------------------------------------------------
# Prompts used in the paper (§4.2 — duplicated training captions)
# NOTE: The paper uses actual LAION captions for highly-duplicated images.
# The prompts below are from the paper's qualitative results (Figure 3, §6.1).
# Do NOT use prompts of real people not in the original training set analysis.
# ---------------------------------------------------------------------------

PAPER_EXAMPLE_PROMPTS = [
    "Ann Graham Lotz",                        # Figure 1 in paper
    "Mona Lisa painting",
    "Nike logo",
    "Apple logo",
    "Coca Cola logo",
]
