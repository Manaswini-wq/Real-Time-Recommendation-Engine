import numpy as np
import pandas as pd


class UserEncoder:
    """Encodes users into dense vectors from interaction history.

    Computes a weighted average of product embeddings the user interacted
    with, blended with user profile features.
    """

    EVENT_WEIGHTS = {
        "view": 0.2, "click": 0.5, "add_to_cart": 0.8, "purchase": 1.0,
    }

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim

    def encode_from_history(
        self, user_interactions: pd.DataFrame,
        product_embeddings: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Generate user embedding as weighted average of interacted products."""
        if user_interactions.empty:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        weighted_sum = np.zeros(self.embedding_dim, dtype=np.float32)
        total_weight = 0.0

        for _, row in user_interactions.iterrows():
            pid = row["product_id"]
            if pid not in product_embeddings:
                continue
            w = self.EVENT_WEIGHTS.get(row.get("event_type", "view"), 0.1)
            # Recency decay: more recent interactions weigh more
            recency = row.get("recency_weight", 1.0)
            weight = w * recency
            weighted_sum += weight * product_embeddings[pid]
            total_weight += weight

        if total_weight > 0:
            weighted_sum /= total_weight

        # L2 normalize
        norm = np.linalg.norm(weighted_sum)
        if norm > 1e-8:
            weighted_sum /= norm

        return weighted_sum

    def batch_encode(
        self, interactions: pd.DataFrame,
        product_embeddings: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Encode all users from their interaction histories."""
        result: dict[str, np.ndarray] = {}
        for uid, group in interactions.groupby("user_id"):
            result[uid] = self.encode_from_history(group, product_embeddings)
        return result
