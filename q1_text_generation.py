"""
Question 1: Text Generation using a Pre-trained Model

Library:
Hugging Face Transformers

Model:
GPT-2

Description:
This program accepts a user-provided prompt and uses the
pre-trained GPT-2 model to generate at least 100 new tokens.
Temperature and top_k are used to control randomness.
"""

from transformers import pipeline, GenerationConfig


# Load the pre-trained GPT-2 text-generation model
generator = pipeline(
    "text-generation",
    model="gpt2"
)


# Take a prompt from the user
prompt = input("Enter your prompt: ")


# Define generation parameters
generation_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    do_sample=True
)


# Generate text
result = generator(
    prompt,
    generation_config=generation_config
)


# Display the generated text
print("\n========== Generated Text ==========")
print(result[0]["generated_text"])
print("====================================")