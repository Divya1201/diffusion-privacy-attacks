from pathlib import Path
import random

# ==============================
# BASE PROMPTS 
# ==============================

BASE_PROMPTS = [
    "a stock photo of a smiling woman looking at the camera",
    "a professional portrait of a smiling woman",
    "a close up photo of a smiling woman",

    "a product image of white sneakers on plain background",
    "white sneakers product photography",

    "a professional headshot of a man in a suit studio lighting",
    "a corporate portrait of a man in formal attire",

    "a close-up photo of a dog sitting on grass",
    "a dog portrait outdoors natural light",
    "a pet dog sitting in a park"
]

# ==============================
# SIMULATE DUPLICATION (CRITICAL)
# ==============================

NUM_TOTAL_PROMPTS = 100   # simulate "many captions"
HIGH_DUP_FACTOR = 20      # strong duplication

def main():
    prompts = []

    # Strongly duplicated prompts (like LAION top duplicates)
    for p in BASE_PROMPTS:
        prompts.extend([p] * HIGH_DUP_FACTOR)

    # Add some random variation (optional realism)
    random.shuffle(prompts)

    # Limit total prompts
    prompts = prompts[:NUM_TOTAL_PROMPTS]

    # Save
    output_file = Path("prompts.txt")
    with open(output_file, "w") as f:
        for p in prompts:
            f.write(p + "\n")

    print(f"Saved {len(prompts)} prompts")
    print("High duplication simulated ✔")

if __name__ == "__main__":
    main()
