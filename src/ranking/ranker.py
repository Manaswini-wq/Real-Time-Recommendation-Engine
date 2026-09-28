from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ranking.features import RankingFeatureBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReRanker:
    """LightGBM-based re-ranking model.

    Takes candidate products from retrieval stage and re-orders
    them based on predicted relevance.
    """

    def __init__(self, learning_rate: float = 0.05, num_trees: int = 300,
                 max_depth: int = 6) -> None:
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": learning_rate,
            "num_leaves": 2 ** max_depth - 1,
            "max_depth": max_depth,
            "verbose": -1,
        }
        self.num_trees = num_trees
        self.model = None
        self.feature_builder = RankingFeatureBuilder()

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> dict:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, log_loss

        train_set = lgb.Dataset(X, label=y)
        valid_sets = [train_set]
        callbacks = [lgb.log_evaluation(50)]

        if X_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(X_val, label=y_val))
            callbacks.append(lgb.early_stopping(30))

        self.model = lgb.train(
            self.params, train_set, num_boost_round=self.num_trees,
            valid_sets=valid_sets, callbacks=callbacks,
        )

        val_proba = self.predict_scores(X_val if X_val is not None else X)
        val_y = y_val if y_val is not None else y
        return {
            "auc": roc_auc_score(val_y, val_proba),
            "logloss": log_loss(val_y, val_proba),
        }

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def rank(self, candidates_df: pd.DataFrame,
             feature_columns: list[str]) -> pd.DataFrame:
        """Score and sort candidates by predicted relevance."""
        X = candidates_df[feature_columns].fillna(0.0)
        candidates_df = candidates_df.copy()
        candidates_df["ranking_score"] = self.predict_scores(X)
        return candidates_df.sort_values("ranking_score", ascending=False)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Ranker saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ReRanker":
        return joblib.load(path)


class DiversityReranker:
    """Post-processing step to enforce category diversity.

    Prevents all top-K results from being the same category.
    """

    @staticmethod
    def diversify(ranked_df: pd.DataFrame, k: int = 10,
                  max_per_category: int = 3) -> pd.DataFrame:
        """Re-order to enforce category diversity in top-K.

        Uses a greedy algorithm: iterate through ranked list, skip
        items if their category already has max_per_category in results.
        """
        category_counts: dict[str, int] = {}
        selected_indices = []

        for idx, row in ranked_df.iterrows():
            cat = row.get("category", "unknown")
            if category_counts.get(cat, 0) < max_per_category:
                selected_indices.append(idx)
                category_counts[cat] = category_counts.get(cat, 0) + 1
            if len(selected_indices) >= k:
                break

        # If not enough diverse items, fill from remaining
        if len(selected_indices) < k:
            remaining = ranked_df.index.difference(selected_indices)
            needed = k - len(selected_indices)
            selected_indices.extend(remaining[:needed])

        return ranked_df.loc[selected_indices].reset_index(drop=True)
