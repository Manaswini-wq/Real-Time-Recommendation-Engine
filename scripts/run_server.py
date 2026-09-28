import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import joblib
    import pandas as pd
    from src.retrieval.faiss_index import FAISSIndex
    from src.ranking.ranker import ReRanker
    from src.serving.recommender import RecommendationEngine
    from src.data.preprocessor import InteractionPreprocessor
    import src.serving.api as api_module

    print("Loading artifacts...")
    products = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    interactions = pd.read_csv(os.path.join(args.data_dir, "interactions.csv"))
    product_embeddings = joblib.load(os.path.join(args.artifacts_dir, "product_embeddings.joblib"))
    ranker = ReRanker.load(os.path.join(args.artifacts_dir, "ranker.joblib"))

    index = FAISSIndex(dimension=128)
    index.load(os.path.join(args.artifacts_dir, "products.index"))

    engine = RecommendationEngine(
        faiss_index=index, ranker=ranker,
        product_catalog=products, product_embeddings=product_embeddings,
    )

    # Build user profiles and histories
    preprocessor = InteractionPreprocessor()
    profiles_df = preprocessor.build_user_profiles(interactions, products)
    user_profiles = {
        row["user_id"]: row.to_dict()
        for _, row in profiles_df.iterrows()
    }
    user_histories = {
        uid: group for uid, group in interactions.groupby("user_id")
    }

    api_module.engine = engine
    api_module.user_profiles = user_profiles
    api_module.user_histories = user_histories

    print(f"Starting server on {args.host}:{args.port}")
    import uvicorn
    uvicorn.run("src.serving.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
