import time

import numpy as np
import pandas as pd


class RankingFeatureBuilder:
    """Builds features for the re-ranking stage.

    Combines user features, product features, and cross-features.
    """

    def build(self, user_profile: dict, candidates: pd.DataFrame,
              retrieval_scores: dict[str, float]) -> pd.DataFrame:
        """Build ranking features for each candidate.

        Args:
            user_profile: User-level features dict.
            candidates: DataFrame of candidate products.
            retrieval_scores: product_id -> retrieval similarity score.

        Returns:
            DataFrame with one row per candidate, containing ranking features.
        """
        df = candidates.copy()

        # Retrieval score from FAISS
        df["retrieval_score"] = df["product_id"].map(retrieval_scores).fillna(0.0)

        # Product features
        df["log_price"] = np.log1p(df["price"])
        df["log_reviews"] = np.log1p(df["num_reviews"])
        df["rating_normalized"] = (df["rating"] - 2.5) / 2.5

        # User-product cross features
        avg_price = user_profile.get("avg_price_viewed", df["price"].median())
        df["price_affinity"] = 1.0 - np.abs(df["price"] - avg_price) / (avg_price + 1)
        df["price_affinity"] = df["price_affinity"].clip(0, 1)

        fav_cat = user_profile.get("favorite_category", "")
        df["category_match"] = (df["category"] == fav_cat).astype(float)

        # Popularity and recency
        df["popularity_rank"] = df["popularity_score"].rank(ascending=False, method="min")
        df["popularity_rank_normalized"] = df["popularity_rank"] / len(df)

        return df

    @staticmethod
    def get_feature_columns() -> list[str]:
        return [
            "retrieval_score", "log_price", "log_reviews",
            "rating_normalized", "price_affinity", "category_match",
            "popularity_score", "popularity_rank_normalized",
        ]
