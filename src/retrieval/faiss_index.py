import time
from pathlib import Path

import faiss
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSIndex:
    """FAISS-based approximate nearest neighbor index for candidate retrieval.

    Supports IVF (Inverted File) index for sub-linear search over
    millions of product embeddings.
    """

    def __init__(self, dimension: int = 128, index_type: str = "IVF256,Flat",
                 nprobe: int = 16) -> None:
        self.dimension = dimension
        self.index_type = index_type
        self.nprobe = nprobe
        self.index: faiss.Index | None = None
        self._id_map: list[str] = []

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Build FAISS index from product embeddings.

        Args:
            embeddings: (N, D) float32 array of product vectors.
            ids: Product ID strings corresponding to each row.
        """
        n, d = embeddings.shape
        assert d == self.dimension, f"Expected dim {self.dimension}, got {d}"
        self._id_map = list(ids)

        logger.info(f"Building FAISS index ({self.index_type}) for {n} vectors")
        start = time.time()

        if n < 1000:
            # Use flat index for small datasets
            self.index = faiss.IndexFlatIP(d)
        else:
            self.index = faiss.index_factory(d, self.index_type, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)

        self.index.add(embeddings)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        logger.info(f"Index built in {time.time() - start:.2f}s, total={self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query_vector: (D,) float32 query embedding.
            k: Number of candidates to retrieve.

        Returns:
            List of (product_id, similarity_score) tuples, sorted by score desc.
        """
        if self.index is None:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)
        # Save ID map alongside
        import json
        id_path = path.replace(".index", "_ids.json")
        Path(id_path).write_text(json.dumps(self._id_map))
        logger.info(f"Index saved to {path}")

    def load(self, path: str) -> None:
        import json
        self.index = faiss.read_index(path)
        id_path = path.replace(".index", "_ids.json")
        self._id_map = json.loads(Path(id_path).read_text())
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        logger.info(f"Index loaded: {self.index.ntotal} vectors")
