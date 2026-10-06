"""
Q4: Image Generation Using a Diffusion Model

This program:
1. Loads a pre-trained diffusion model.
2. Accepts a text prompt from the user.
3. Allows the user to configure inference steps.
4. Generates an image.
5. Saves the image as output.png.
6. Displays the generation time.
"""

import time

import torch
from diffusers import StableDiffusionPipeline


# ============================================================
# Step 1: Select device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# ============================================================
# Step 2: Load pre-trained diffusion model
# ============================================================

model_id = "runwayml/stable-diffusion-v1-5"

print("Loading diffusion model...")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

print("Model loaded successfully.")


# ============================================================
# Step 3: Get user prompt
# ============================================================

prompt = input("\nEnter your image prompt: ")


# ============================================================
# Step 4: Configure inference steps
# ============================================================

steps = int(
    input("Enter number of inference steps (default 30): ") or "30"
)


# ============================================================
# Step 5: Generate image and measure time
# ============================================================

start_time = time.time()

result = pipe(
    prompt,
    num_inference_steps=steps
)

end_time = time.time()

generation_time = end_time - start_time


# ============================================================
# Step 6: Save image
# ============================================================

output_path = "output.png"

result.images[0].save(output_path)


# ============================================================
# Step 7: Display result
# ============================================================

print("\nImage generated successfully!")
print("Output file:", output_path)
print(f"Generation time: {generation_time:.2f} seconds")