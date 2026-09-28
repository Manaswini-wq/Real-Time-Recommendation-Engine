import numpy as np
import pandas as pd
from src.embeddings.product_encoder import ProductEncoder
from src.embeddings.user_encoder import UserEncoder


def test_product_encoder():
    products = pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "title": ["wireless headphones", "running shoes", "desk lamp"],
        "category": ["electronics", "clothing", "home"],
        "price": [49.99, 89.99, 29.99],
        "rating": [4.2, 3.8, 4.5],
        "popularity_score": [0.8, 0.6, 0.4],
    })
    encoder = ProductEncoder(embedding_dim=64)
    encoder.fit(products)
    embeddings = encoder.encode(products)

    assert embeddings.shape == (3, 64)
    # Should be L2 normalized
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=0.01)


def test_user_encoder():
    product_embs = {
        "p1": np.random.randn(64).astype(np.float32),
        "p2": np.random.randn(64).astype(np.float32),
    }
    history = pd.DataFrame({
        "product_id": ["p1", "p2", "p1"],
        "event_type": ["view", "click", "purchase"],
    })
    encoder = UserEncoder(embedding_dim=64)
    emb = encoder.encode_from_history(history, product_embs)
    assert emb.shape == (64,)
