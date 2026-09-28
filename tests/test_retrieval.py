import numpy as np
from src.retrieval.faiss_index import FAISSIndex


def test_faiss_build_and_search():
    dim = 32
    n = 500
    embeddings = np.random.randn(n, dim).astype(np.float32)
    # Normalize
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    ids = [f"p_{i}" for i in range(n)]

    index = FAISSIndex(dimension=dim)
    index.build(embeddings, ids)

    query = embeddings[0]
    results = index.search(query, k=5)

    assert len(results) == 5
    # Top result should be the query itself
    assert results[0][0] == "p_0"
    assert results[0][1] > 0.99
