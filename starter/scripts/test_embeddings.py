"""Smoke test: verify real embeddings and consistent dimensions."""
import sys
from pathlib import Path

starter_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(starter_dir))

from src.embeddings import MyEmbedder

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = MyEmbedder(MODEL_NAME)
print("Model:", embedder.model_name)
print("Dimensions:", embedder.dimensions)

texts = [
    "Supervised learning trains on labeled examples.",
    "Unsupervised learning identifies patterns without labels.",
]

vectors = embedder.embed_documents(texts)
query_vector = embedder.embed_query("What is supervised learning?")

assert len(vectors) == len(texts)
assert all(len(v) == embedder.dimensions for v in vectors)
assert len(query_vector) == embedder.dimensions
assert vectors[0] != vectors[1]
assert embedder.embed_documents([]) == []
print("Document vectors:", len(vectors))
print("Query vector length:", len(query_vector))
print("Embeddings test PASSED")
