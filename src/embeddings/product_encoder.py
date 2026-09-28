import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProductEncoder:
    """Encodes products into dense vector embeddings.

    Uses a combination of text embeddings (from title) and numeric
    feature normalization for a hybrid representation.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim
        self._encoder = None
        self._category_map: dict[str, int] = {}

    def fit(self, products: pd.DataFrame) -> None:
        """Build category mappings and optionally load sentence encoder."""
        categories = products["category"].unique().tolist()
        self._category_map = {c: i for i, c in enumerate(categories)}
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformer for product encoding")
        except ImportError:
            logger.info("sentence-transformers not available, using hash encoding")

    def encode(self, products: pd.DataFrame) -> np.ndarray:
        """Encode products into fixed-dimension vectors."""
        n = len(products)
        embeddings = np.zeros((n, self.embedding_dim), dtype=np.float32)

        if self._encoder is not None:
            text_emb = self._encoder.encode(
                products["title"].tolist(), show_progress_bar=False,
                normalize_embeddings=True,
            )
            text_dim = min(text_emb.shape[1], self.embedding_dim - 16)
            embeddings[:, :text_dim] = text_emb[:, :text_dim]
        else:
            # Hash-based fallback encoding
            for i, title in enumerate(products["title"]):
                h = hash(title)
                rng = np.random.RandomState(abs(h) % (2**31))
                embeddings[i, :self.embedding_dim - 16] = rng.randn(
                    self.embedding_dim - 16
                ) * 0.1

        # Append numeric features (last 16 dims)
        offset = self.embedding_dim - 16
        if "price" in products.columns:
            prices = products["price"].values.astype(np.float32)
            prices = (prices - prices.mean()) / (prices.std() + 1e-8)
            embeddings[:, offset] = prices
        if "rating" in products.columns:
            ratings = products["rating"].values.astype(np.float32)
            embeddings[:, offset + 1] = (ratings - 3.5) / 1.5
        if "popularity_score" in products.columns:
            embeddings[:, offset + 2] = products["popularity_score"].values
        if "category" in products.columns:
            for i, cat in enumerate(products["category"]):
                cat_idx = self._category_map.get(cat, 0)
                embeddings[i, offset + 3 + (cat_idx % 12)] = 1.0

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings

    def encode_single(self, product: dict) -> np.ndarray:
        """Encode a single product dict into a vector."""
        df = pd.DataFrame([product])
        return self.encode(df)[0]
