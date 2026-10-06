# Generative AI Test

This project contains solutions for all four questions of the Generative AI assessment.

## Project Structure

```text
Generative_AI_Test/
│
├── q1_text_generation.py
├── q2_rag.py
├── q3_fine_tuning.py
├── q4_image_generation.py
├── output.png
├── requirements.txt
└── README.md
```

## Prerequisites

* Python 3.10 or later
* Internet connection for downloading pre-trained Hugging Face models
* Sufficient disk space for the required models
* NVIDIA GPU is recommended for faster image generation in Q4, but CPU can also be used

## 1. Create a Virtual Environment

Open PowerShell/Command Prompt in the project folder:

```powershell
python -m venv venv
```

Activate the virtual environment.

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, Command Prompt can be used instead:

```cmd
venv\Scripts\activate.bat
```

## 2. Install Dependencies

After activating the virtual environment:

```powershell
pip install -r requirements.txt
```

---

# Q1 — Text Generation

Run:

```powershell
python q1_text_generation.py
```

The program loads a pre-trained GPT-2 model and accepts a text prompt from the user.

Example:

```text
Enter your prompt: Artificial Intelligence is
```

The model then generates text based on the prompt.

---

# Q2 — Retrieval-Augmented Generation

Run:

```powershell
python q2_rag.py
```

The program retrieves relevant information from the provided knowledge source and uses it to generate an answer.

Follow the prompts displayed by the program.

---

# Q3 — Fine-Tuning a Small Model

Run:

```powershell
python q3_fine_tuning.py
```

The program:

1. Creates a custom dataset containing 20 samples.
2. Loads a DistilBERT tokenizer.
3. Tokenizes the dataset.
4. Splits the data into training and evaluation sets.
5. Fine-tunes DistilBERT for 3 epochs using Hugging Face Trainer.
6. Calculates evaluation loss and accuracy.
7. Saves the fine-tuned model locally.
8. Tests the saved model.

The fine-tuned model is saved in:

```text
q3_finetuned_model/
```

Training results/checkpoints are stored in:

```text
q3_results/
```

---

# Q4 — Image Generation

Run:

```powershell
python q4_image_generation.py
```

The program loads a pre-trained Stable Diffusion model.

It asks for:

```text
Enter your image prompt:
```

Example:

```text
A futuristic city at sunset with flying cars, highly detailed digital art
```

It then asks for the number of inference steps:

```text
Enter number of inference steps (default 30):
```

Example:

```text
30
```

The generated image is saved as:

```text
output.png
```

The program also displays the image generation time.

## Important Note

The first execution of the programs may take longer because the required pre-trained models need to be downloaded from Hugging Face.

Q4 requires significantly more disk space because the Stable Diffusion model is several GB in size.

## Submission

The project can be submitted as a ZIP file containing the source code, README, requirements file, and generated output.

The `venv` folder is intentionally excluded because it is machine-specific and can be recreated using:

```powershell
python -m venv venv
pip install -r requirements.txt
```
