"""
Q3: Fine-tuning a Small Model on Custom Data

Task:
Fine-tune DistilBERT for binary sentiment classification.

Dataset:
20 custom text samples.

Labels:
0 = Negative
1 = Positive
"""

import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

from sklearn.metrics import accuracy_score


# ============================================================
# STEP 1: Create custom dataset
# ============================================================

texts = [
    "I really enjoyed this product.",
    "The service was excellent.",
    "This is an amazing experience.",
    "I am very happy with the purchase.",
    "The product quality is outstanding.",
    "The support team was very helpful.",
    "I love using this application.",
    "The delivery was fast and smooth.",
    "The product works perfectly.",
    "I would definitely recommend this service.",

    "I am very disappointed with this product.",
    "The service was terrible.",
    "This was a very bad experience.",
    "I am unhappy with my purchase.",
    "The product quality is poor.",
    "The support team was not helpful.",
    "I hate using this application.",
    "The delivery was very late.",
    "The product does not work properly.",
    "I would not recommend this service."
]

labels = [
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
]

print("Number of samples:", len(texts))


# ============================================================
# STEP 2: Create Hugging Face Dataset
# ============================================================

data = {
    "text": texts,
    "label": labels
}

dataset = Dataset.from_dict(data)

print("\nDataset:")
print(dataset)


# ============================================================
# STEP 3: Load DistilBERT tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


# ============================================================
# STEP 4: Tokenize dataset
# ============================================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

print("\nTokenization completed!")


# ============================================================
# STEP 5: Split dataset
# ============================================================

dataset = tokenized_dataset.class_encode_column("label")

split_dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="label"
)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print("\nTraining samples:", len(train_dataset))
print("Evaluation samples:", len(eval_dataset))


# ============================================================
# STEP 6: Load DistilBERT classification model
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# ============================================================
# STEP 7: Accuracy metric
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy
    }


# ============================================================
# STEP 8: Training configuration
# ============================================================

training_args = TrainingArguments(
    output_dir="./q3_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)


# ============================================================
# STEP 9: Create Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)


# ============================================================
# STEP 10: Fine-tune model
# ============================================================

print("\nStarting fine-tuning...")

trainer.train()


# ============================================================
# STEP 11: Evaluate model
# ============================================================

print("\nEvaluating model...")

evaluation_results = trainer.evaluate()

print("\nEvaluation Results:")

for key, value in evaluation_results.items():
    print(f"{key}: {value}")


# ============================================================
# STEP 12: Save fine-tuned model
# ============================================================

output_model_dir = "./q3_finetuned_model"

trainer.save_model(output_model_dir)

tokenizer.save_pretrained(output_model_dir)

print("\nFine-tuned model saved to:")
print(output_model_dir)


# ============================================================
# STEP 13: Test saved model
# ============================================================

classifier = pipeline(
    "text-classification",
    model=output_model_dir,
    tokenizer=output_model_dir
)

test_text = "I really enjoyed using this product."

prediction = classifier(test_text)

print("\nTest Prediction:")
print(prediction)