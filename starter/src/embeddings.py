"""Real local text embeddings.
"""
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """TODO (starter): Implement a real provider adapter; never use mock vectors.
    """

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the embedding dependency: "
                "python -m pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        dimensions = self._model.get_embedding_dimension()
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise RuntimeError("The embedding model did not report valid dimensions")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Each input document must be a non-empty string")

        vectors = self._model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        result = vectors.tolist()
        if len(result) != len(texts) or any(len(v) != self.dimensions for v in result):
            raise RuntimeError("Embedding output has unexpected shape")
        return result

    def embed_query(self, question: str) -> list[float]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        return self.embed_documents([question])[0]
