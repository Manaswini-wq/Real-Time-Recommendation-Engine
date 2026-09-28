import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import joblib

from src.data.preprocessor import InteractionPreprocessor
from src.ranking.ranker import ReRanker
from src.ranking.features import RankingFeatureBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    products = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    interactions = pd.read_csv(os.path.join(args.data_dir, "interactions.csv"))

    print("Building interaction matrix...")
    preprocessor = InteractionPreprocessor()
    matrix = preprocessor.build_interaction_matrix(interactions)

    # Create training labels: purchase/add_to_cart = positive, view only = negative
    labels = interactions.copy()
    labels["label"] = labels["event_type"].isin(["purchase", "add_to_cart"]).astype(int)
    merged = labels.merge(products, left_on="product_id", right_on="product_id", how="left")

    # Build simple ranking features for training
    merged["retrieval_score"] = np.random.uniform(0.3, 1.0, len(merged))
    merged["log_price"] = np.log1p(merged["price"])
    merged["log_reviews"] = np.log1p(merged["num_reviews"])
    merged["rating_normalized"] = (merged["rating"] - 2.5) / 2.5
    merged["price_affinity"] = np.random.uniform(0.2, 1.0, len(merged))
    merged["category_match"] = np.random.choice([0.0, 1.0], len(merged), p=[0.7, 0.3])
    merged["popularity_rank_normalized"] = np.random.uniform(0, 1, len(merged))

    feature_cols = RankingFeatureBuilder.get_feature_columns()
    X = merged[feature_cols].fillna(0.0)
    y = merged["label"]

    # Temporal split
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    print(f"Positive rate: {y_train.mean():.4f}")

    print("\nTraining re-ranker...")
    ranker = ReRanker(learning_rate=0.05, num_trees=300, max_depth=6)
    metrics = ranker.fit(X_train, y_train, X_val, y_val)
    print(f"Val AUC: {metrics['auc']:.4f}, Val LogLoss: {metrics['logloss']:.4f}")

    ranker.save(os.path.join(args.output_dir, "ranker.joblib"))
    print("Done.")


if __name__ == "__main__":
    main()
