"""
Question 2: Document Question Answering (RAG Basics)

Libraries:
- Sentence Transformers
- Scikit-learn

This program:
1. Splits the document into chunks.
2. Creates embeddings for each chunk.
3. Takes a user question.
4. Creates an embedding for the question.
5. Uses cosine similarity to find the most relevant chunk.
6. Returns the relevant chunk as the answer.
"""

import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Provided document from the machine test
document = """
Generative AI refers to models that can create new content such as text,
images, or audio. Large Language Models (LLMs) like GPT and LLaMA are
trained on massive text corpora. Retrieval Augmented Generation (RAG)
combines retrieval of relevant documents with generative models to
produce accurate answers. Python is widely used in the AI ecosystem
due to libraries like PyTorch, TensorFlow, and Hugging Face Transformers.
"""


# Step 1: Split the document into chunks
chunks = re.split(r'(?<=[.!?])\s+', document.strip())

print("Document Chunks:")
for i, chunk in enumerate(chunks):
    print(f"{i}: {chunk}")


# Step 2: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Step 3: Create embeddings for document chunks
chunk_embeddings = model.encode(chunks)


# Step 4: Take a question from the user
query = input("\nAsk a question: ")


# Step 5: Create an embedding for the user query
query_embedding = model.encode([query])


# Step 6: Calculate cosine similarity
similarities = cosine_similarity(
    query_embedding,
    chunk_embeddings
)[0]


# Step 7: Find the most relevant chunk
best_chunk_index = similarities.argmax()
best_chunk = chunks[best_chunk_index]


# Step 8: Display the answer
print("\nAnswer:")
print(best_chunk)