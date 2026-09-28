import time
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = [
    "electronics", "clothing", "sports", "books", "home",
    "beauty", "toys", "automotive", "grocery", "garden",
]
PRICE_RANGES = {
    "electronics": (20, 500), "clothing": (10, 150), "sports": (15, 200),
    "books": (5, 40), "home": (10, 300), "beauty": (5, 80),
    "toys": (5, 60), "automotive": (10, 200), "grocery": (2, 50),
    "garden": (10, 150),
}


class ProductCatalogGenerator:
    """Generates a synthetic product catalog."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def generate(self, num_products: int) -> pd.DataFrame:
        records = []
        for i in range(num_products):
            cat = self.rng.choice(CATEGORIES)
            lo, hi = PRICE_RANGES[cat]
            records.append({
                "product_id": f"p_{i:06d}",
                "title": f"{cat} product {i}",
                "category": cat,
                "price": round(self.rng.uniform(lo, hi), 2),
                "rating": round(self.rng.uniform(2.5, 5.0), 1),
                "num_reviews": int(self.rng.exponential(50)),
                "popularity_score": round(self.rng.uniform(0, 1), 4),
            })
        return pd.DataFrame(records)


class UserInteractionGenerator:
    """Generates synthetic user interaction history."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def generate(self, num_users: int, num_products: int,
                 avg_interactions: int = 30) -> pd.DataFrame:
        records = []
        for i in range(num_users):
            uid = f"u_{i:06d}"
            # Each user has preferred categories
            prefs = self.rng.choice(CATEGORIES, size=3, replace=False).tolist()
            n_interactions = max(5, int(self.rng.exponential(avg_interactions)))

            for _ in range(n_interactions):
                pid_idx = self.rng.randint(0, num_products)
                event = self.rng.choice(
                    ["view", "click", "add_to_cart", "purchase"],
                    p=[0.6, 0.25, 0.1, 0.05],
                )
                records.append({
                    "user_id": uid,
                    "product_id": f"p_{pid_idx:06d}",
                    "event_type": event,
                    "timestamp": time.time() - self.rng.randint(0, 86400 * 90),
                    "preferred_categories": ",".join(prefs),
                })
        return pd.DataFrame(records)


def save_data(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
