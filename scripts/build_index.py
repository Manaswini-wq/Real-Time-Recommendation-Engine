import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import joblib

from src.embeddings.product_encoder import ProductEncoder
from src.retrieval.faiss_index import FAISSIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--embedding_dim", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading products...")
    products = pd.read_csv(os.path.join(args.data_dir, "products.csv"))

    print("Encoding products...")
    encoder = ProductEncoder(embedding_dim=args.embedding_dim)
    encoder.fit(products)
    embeddings = encoder.encode(products)

    print(f"Embeddings shape: {embeddings.shape}")

    # Save product embeddings as dict
    emb_dict = {
        pid: embeddings[i]
        for i, pid in enumerate(products["product_id"])
    }
    joblib.dump(emb_dict, os.path.join(args.output_dir, "product_embeddings.joblib"))

    print("Building FAISS index...")
    index = FAISSIndex(dimension=args.embedding_dim)
    index.build(embeddings, products["product_id"].tolist())
    index.save(os.path.join(args.output_dir, "products.index"))

    # Save encoder
    joblib.dump(encoder, os.path.join(args.output_dir, "product_encoder.joblib"))

    print("Done.")


if __name__ == "__main__":
    main()
