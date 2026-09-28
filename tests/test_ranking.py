import numpy as np
import pandas as pd
from src.ranking.ranker import DiversityReranker


def test_diversity_reranker():
    df = pd.DataFrame({
        "product_id": [f"p_{i}" for i in range(20)],
        "category": ["electronics"] * 10 + ["clothing"] * 5 + ["books"] * 5,
        "ranking_score": np.linspace(1.0, 0.1, 20),
    })
    result = DiversityReranker.diversify(df, k=10, max_per_category=3)

    assert len(result) == 10
    cat_counts = result["category"].value_counts()
    assert cat_counts.max() <= 3
