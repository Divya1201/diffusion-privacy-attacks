from pathlib import Path
import torch
from diffusion_privacy_attacks.diffusion import load_stable_diffusion, generate_images_for_prompt

# ==============================
# CONFIG (Paper settings)
# ==============================

OUTPUT_DIR = Path("outputs/generated")

PROMPTS = [
    "Mona Lisa painting",
    "Nike logo",
    "Apple logo",
    "Coca Cola logo",
]

IMAGES_PER_PROMPT = 500   # 🔥 CRITICAL (paper requirement)

# ==============================
# MAIN
# ==============================

def main():
    print("🚀 Loading Stable Diffusion v1.4 (PLMS)...")
    pipe = load_stable_diffusion()

    print("\n📸 Generating images (paper setting: 500 per prompt)...")

    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] Prompt: {prompt}")

        generate_images_for_prompt(
            pipe=pipe,
            prompt=prompt,
            output_dir=OUTPUT_DIR,
            num_images=IMAGES_PER_PROMPT,  # 🔥 KEY FIX
            image_size=512,               # paper: 512×512
        )

    print("\n✅ Generation complete!")


if __name__ == "__main__":
    main()
