"""
SentenceTransformer embedding + FAISS index utilities.

Extracted from train_geoai_geocoder.py Part 6–7 and Inference.py Part 6–7.
Bug fix: uses IndexFlatIP (cosine similarity) instead of IndexFlatL2.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64


def load_model(model_name: str = DEFAULT_MODEL):
    """Load a SentenceTransformer model.

    Returns
    -------
    SentenceTransformer instance
    """
    from sentence_transformers import SentenceTransformer
    logger.info("Loading SentenceTransformer model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Model loaded — embedding dimension: %d", model.get_sentence_embedding_dimension())
    return model


def encode_texts(
    model,
    texts: list,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode a list of texts into embedding vectors.

    Parameters
    ----------
    model : SentenceTransformer
    texts : list[str]
    batch_size : int
    normalize : bool
        If True, L2-normalise embeddings (required for cosine similarity via IP).
    show_progress : bool

    Returns
    -------
    np.ndarray of shape (len(texts), dim), dtype float32
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    logger.info("Encoded %d texts → shape %s", len(texts), embeddings.shape)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS Inner Product index (cosine similarity on normalised vectors).

    Bug fix: original code used IndexFlatL2. Since all-MiniLM-L6-v2 outputs
    L2-normalised vectors, Inner Product = cosine similarity. Scores become
    directly interpretable: 1.0 = identical, 0.0 = orthogonal.

    Parameters
    ----------
    embeddings : np.ndarray of shape (N, dim), dtype float32, L2-normalised

    Returns
    -------
    faiss.IndexFlatIP
    """
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    logger.info("FAISS IndexFlatIP built: %d vectors, dim=%d", index.ntotal, dim)
    return index


def search_faiss(
    index,
    query_embeddings: np.ndarray,
    top_k: int = 20,
) -> tuple:
    """Search the FAISS index.

    Parameters
    ----------
    index : faiss.Index
    query_embeddings : np.ndarray of shape (Q, dim)
    top_k : int

    Returns
    -------
    (scores, indices) — both np.ndarray of shape (Q, top_k)
        scores: cosine similarity (higher = better, 0–1 range for normalised vectors)
        indices: row indices into the original embedding matrix
    """
    scores, indices = index.search(query_embeddings.astype(np.float32), top_k)
    return scores, indices


def save_faiss_index(index, path: str):
    """Save FAISS index to disk."""
    import faiss
    faiss.write_index(index, path)
    logger.info("FAISS index saved: %s", path)


def load_faiss_index(path: str):
    """Load FAISS index from disk."""
    import faiss
    index = faiss.read_index(path)
    logger.info("FAISS index loaded: %s (%d vectors)", path, index.ntotal)
    return index
