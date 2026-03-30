from diffusers import StableDiffusionPipeline
import torch

def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe
