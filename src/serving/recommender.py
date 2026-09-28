import time

import numpy as np
import pandas as pd

from src.retrieval.faiss_index import FAISSIndex
from src.ranking.ranker import ReRanker, DiversityReranker
from src.ranking.features import RankingFeatureBuilder
from src.embeddings.user_encoder import UserEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationEngine:
    """Orchestrates the full two-stage recommendation pipeline."""

    def __init__(self, faiss_index: FAISSIndex, ranker: ReRanker,
                 product_catalog: pd.DataFrame,
                 product_embeddings: dict[str, np.ndarray]) -> None:
        self.index = faiss_index
        self.ranker = ranker
        self.catalog = product_catalog.set_index("product_id")
        self.product_embeddings = product_embeddings
        self.user_encoder = UserEncoder()
        self.feature_builder = RankingFeatureBuilder()

    def recommend(self, user_id: str, user_history: pd.DataFrame,
                  user_profile: dict, k: int = 10,
                  num_candidates: int = 50) -> list[dict]:
        """Generate top-K recommendations for a user.

        Args:
            user_id: User identifier.
            user_history: User's interaction history DataFrame.
            user_profile: Aggregated user features dict.
            k: Number of final recommendations.
            num_candidates: Candidates from retrieval stage.

        Returns:
            List of recommendation dicts with product_id, score, rank.
        """
        start = time.perf_counter()

        # Stage 1: Retrieval via FAISS
        user_embedding = self.user_encoder.encode_from_history(
            user_history, self.product_embeddings
        )
        candidates = self.index.search(user_embedding, k=num_candidates)
        retrieval_ms = (time.perf_counter() - start) * 1000

        if not candidates:
            return []

        # Build candidate DataFrame
        candidate_ids = [pid for pid, _ in candidates]
        retrieval_scores = {pid: s for pid, s in candidates}
        candidates_df = self.catalog.loc[
            self.catalog.index.isin(candidate_ids)
        ].reset_index()

        # Stage 2: Re-ranking
        ranked_features = self.feature_builder.build(
            user_profile, candidates_df, retrieval_scores
        )
        feature_cols = RankingFeatureBuilder.get_feature_columns()
        ranked = self.ranker.rank(ranked_features, feature_cols)

        # Stage 3: Diversity post-processing
        final = DiversityReranker.diversify(ranked, k=k)

        total_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Recommend({user_id}): retrieval={retrieval_ms:.1f}ms, "
                     f"total={total_ms:.1f}ms, candidates={len(candidates)}")

        results = []
        for rank, (_, row) in enumerate(final.iterrows(), 1):
            results.append({
                "product_id": row["product_id"],
                "title": row.get("title", ""),
                "category": row.get("category", ""),
                "price": row.get("price", 0),
                "ranking_score": round(row.get("ranking_score", 0), 4),
                "rank": rank,
            })
        return results

    def find_similar(self, product_id: str, k: int = 10) -> list[dict]:
        """Find products similar to a given product."""
        if product_id not in self.product_embeddings:
            return []
        query = self.product_embeddings[product_id]
        results = self.index.search(query, k=k + 1)
        # Exclude the query product itself
        return [
            {"product_id": pid, "similarity": round(score, 4)}
            for pid, score in results if pid != product_id
        ][:k]
