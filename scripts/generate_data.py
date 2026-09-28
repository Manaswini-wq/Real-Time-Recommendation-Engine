import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.generator import ProductCatalogGenerator, UserInteractionGenerator, save_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_products", type=int, default=50000)
    parser.add_argument("--num_users", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_products} products...")
    products = ProductCatalogGenerator(seed=args.seed).generate(args.num_products)
    save_data(products, os.path.join(args.output_dir, "products.csv"))

    print(f"Generating interactions for {args.num_users} users...")
    interactions = UserInteractionGenerator(seed=args.seed).generate(
        args.num_users, args.num_products
    )
    save_data(interactions, os.path.join(args.output_dir, "interactions.csv"))

    print(f"Products: {len(products)}, Interactions: {len(interactions)}")
    print("Done.")


if __name__ == "__main__":
    main()
