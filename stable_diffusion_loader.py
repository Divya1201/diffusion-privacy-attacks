"""Utility for loading Stable Diffusion with HuggingFace diffusers.

Requirements:
    pip install torch diffusers transformers accelerate
"""

from __future__ import annotations

import torch
from diffusers import StableDiffusionPipeline


MODEL_ID = "runwayml/stable-diffusion-v1-5"


def load_pipeline(model_id: str = MODEL_ID) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline and move it to GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    if device == "cuda":
        # Optional memory optimization (safe to skip if xformers is unavailable).
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    pipe = pipe.to(device)
    return pipe


PIPELINE = load_pipeline()


def generate(prompt: str, seed: int):
    """Generate a single image for a prompt using a deterministic seed."""
    device = PIPELINE.device
    generator = torch.Generator(device=device).manual_seed(seed)

    result = PIPELINE(prompt=prompt, generator=generator)
    return result.images[0]


if __name__ == "__main__":
    image = generate("a futuristic city at sunset, cinematic lighting", seed=42)
    image.save("generated.png")
    print("Saved generated image to generated.png")
