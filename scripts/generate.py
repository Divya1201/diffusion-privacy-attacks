from diffusion_privacy_attacks.diffusion import load_model
import os

def generate_images(prompts, num_images=50, output_dir="generated"):
    pipe = load_model()

    os.makedirs(output_dir, exist_ok=True)

    for prompt in prompts:
        safe_prompt = prompt.replace(" ", "_")

        print(f"\n🚀 Generating for: {prompt}")

        for i in range(num_images):
            image = pipe(prompt).images[0]

            image.save(f"{output_dir}/{safe_prompt}_{i}.png")

    print("\n✅ All images generated")


if __name__ == "__main__":
    prompts = [
        "a photo of a dog",
        "a photo of a cat",
        "a car on the road"
    ]

    generate_images(prompts, num_images=50)
