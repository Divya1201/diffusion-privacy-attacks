from diffusion_privacy_attacks.diffusion import load_model
import os

pipe = load_model()

prompt = "a photo of a dog"

os.makedirs("generated", exist_ok=True)

for i in range(100):   # start with 100 (later 500)
    image = pipe(prompt).images[0]
    image.save(f"generated/img_{i}.png")
