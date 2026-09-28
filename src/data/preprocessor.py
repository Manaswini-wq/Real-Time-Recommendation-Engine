import numpy as np
import pandas as pd


class InteractionPreprocessor:
    """Preprocesses user interactions into training signals."""

    EVENT_WEIGHTS = {
        "view": 0.2, "click": 0.5, "add_to_cart": 0.8, "purchase": 1.0,
    }

    def build_interaction_matrix(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Creates weighted user-product interaction scores."""
        df = interactions.copy()
        df["weight"] = df["event_type"].map(self.EVENT_WEIGHTS).fillna(0.1)
        matrix = df.groupby(["user_id", "product_id"]).agg(
            interaction_score=("weight", "sum"),
            num_interactions=("weight", "count"),
            last_interaction=("timestamp", "max"),
        ).reset_index()
        return matrix

    def build_user_profiles(self, interactions: pd.DataFrame,
                            products: pd.DataFrame) -> pd.DataFrame:
        """Aggregate user-level features from interaction history."""
        merged = interactions.merge(products, on="product_id", how="left")
        profiles = merged.groupby("user_id").agg(
            total_interactions=("event_type", "count"),
            total_purchases=("event_type", lambda x: (x == "purchase").sum()),
            avg_price_viewed=("price", "mean"),
            favorite_category=("category", lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"),
            avg_rating_preference=("rating", "mean"),
        ).reset_index()
        return profiles
