# Real-Time Recommendation Engine

A production-grade, two-stage recommendation system that serves personalized product recommendations in real time. Implements the industry-standard candidate retrieval + re-ranking architecture used at scale by Amazon, Netflix, YouTube, and Spotify.

---

## Architecture

### End-to-End Recommendation Flow

```
  +-----------+       +----------------+       +----------------+       +-----------------+
  |  Client   | ----> |  FastAPI       | ----> |  User Encoder  | ----> | FAISS ANN Index |
  |  Request  |       |  /recommend    |       |  (weighted avg |       | (IVF256,Flat)   |
  |           |       |                |       |   of product   |       |                 |
  | user_id   |       |                |       |   embeddings)  |       | ~50 candidates  |
  | context   |       |                |       |                |       | in <5ms         |
  +-----------+       +----------------+       +----------------+       +--------+--------+
                                                                                 |
                                                                        +--------v--------+
                                                                        | Re-Ranking      |
                                                                        | (LightGBM)      |
                                                                        |                 |
                                                                        | Cross-features: |
                                                                        | - price affinity|
                                                                        | - category match|
                                                                        | - popularity    |
                                                                        | - retrieval     |
                                                                        |   score         |
                                                                        +--------+--------+
                                                                                 |
                                                                        +--------v--------+
                                                                        | Diversity       |
                                                                        | Post-Processing |
                                                                        |                 |
                                                                        | Greedy category |
                                                                        | dedup, max 3    |
                                                                        | per category    |
                                                                        +--------+--------+
                                                                                 |
                                                                        +--------v--------+
                                                                        | Top-K Results   |
                                                                        | (default K=10)  |
                                                                        +-----------------+
```

### Product Embedding Pipeline (Offline)

```
  +-------------------+       +-------------------+       +-------------------+
  |  Product Catalog  | ----> | Product Encoder   | ----> | FAISS Index       |
  |  (50K+ products)  |       |                   |       | (IVF256,Flat)     |
  |                   |       | Sentence-          |       |                   |
  | - title           |       |   Transformer     |       | Stores all        |
  | - category        |       |   text embedding  |       | product vectors   |
  | - price           |       | + normalized      |       | for ANN search    |
  | - rating          |       |   numeric features|       |                   |
  | - popularity      |       | + category one-hot|       |                   |
  +-------------------+       +-------------------+       +-------------------+
```

### Real-Time User Signal Processing

```
  User Events                 Event Processor              User Profile
  (clicks, views,             (sliding window              (Redis / Memory)
   purchases)                  aggregation)

  +--click--------+       +-------------------+       +-------------------+
  |  user: u_001  | ----> | Window size: 50   | ----> | recent_click_rate |
  |  product: p_5 |       | Compute:          |       | top_category      |
  |  time: now    |       | - click rate      |       | session_length    |
  +---------------+       | - purchase rate   |       | purchase_rate     |
  +--view---------+       | - top category    |       +-------------------+
  |  user: u_001  | ----> | - session length  |
  |  product: p_8 |       +-------------------+
  +---------------+
```

---

## Why Two Stages?

| Approach | Latency | Accuracy | Scales to |
|----------|---------|----------|-----------|
| Brute-force rank all items | ~500ms | Highest | ~1K items |
| **Retrieval + Ranking** | **<20ms** | **High** | **Millions** |
| Retrieval only (no re-rank) | <5ms | Medium | Millions |

The two-stage approach is the industry standard because:
1. **Retrieval** (FAISS) narrows millions of products to ~50 candidates in <5ms using approximate nearest neighbors
2. **Ranking** (LightGBM) applies expensive cross-features only on those 50 candidates, keeping total latency under 20ms

---

## Key Features

| Feature | Description |
|---------|-------------|
| **FAISS ANN Retrieval** | Sub-linear similarity search over 50K+ product embeddings using IVF index (<5ms) |
| **Hybrid Embeddings** | Sentence-transformer text encoding + normalized numeric features + category one-hot |
| **LightGBM Re-Ranker** | Cross-feature scoring: price affinity, category match, popularity, retrieval score |
| **Diversity Enforcement** | Greedy post-processing caps max items per category in top-K results |
| **User Encoding** | Weighted average of interacted product embeddings (purchase > click > view) |
| **Streaming Signals** | Sliding-window event processor for real-time user profile updates |
| **Similar Products** | Content-based similarity via shared FAISS index |
| **Low Latency** | End-to-end <20ms p99 for full retrieval + ranking pipeline |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | PyTorch, Sentence-Transformers (all-MiniLM-L6-v2) |
| Retrieval | FAISS (IVF256,Flat with inner product) |
| Ranking | LightGBM |
| Serving | FastAPI, Uvicorn |
| User Profiles | Redis (with in-memory fallback) |
| Streaming | Custom sliding-window event processor |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## Project Structure

```
realtime-recommendation-engine/
|
+-- README.md
+-- requirements.txt
+-- setup.py
+-- Dockerfile
+-- docker-compose.yml
+-- .gitignore
|
+-- .github/workflows/
|   +-- ci.yml
|
+-- configs/
|   +-- default.yaml
|
+-- scripts/
|   +-- generate_data.py          # Generate 50K products + user interactions
|   +-- build_index.py            # Encode products + build FAISS index
|   +-- train_ranker.py           # Train LightGBM re-ranking model
|   +-- run_server.py             # Launch recommendation API
|
+-- src/
|   +-- data/
|   |   +-- generator.py          # Synthetic product catalog + interactions
|   |   +-- preprocessor.py       # Interaction matrix + user profiles
|   |
|   +-- embeddings/
|   |   +-- product_encoder.py    # Hybrid text + numeric product encoder
|   |   +-- user_encoder.py       # Weighted-average user encoder
|   |
|   +-- retrieval/
|   |   +-- faiss_index.py        # FAISS IVF index: build, search, save, load
|   |
|   +-- ranking/
|   |   +-- features.py           # Cross-feature builder for re-ranking
|   |   +-- ranker.py             # LightGBM re-ranker + diversity filter
|   |
|   +-- serving/
|   |   +-- recommender.py        # Full pipeline orchestrator
|   |   +-- api.py                # FastAPI endpoints
|   |
|   +-- streaming/
|   |   +-- event_processor.py    # Sliding-window user signal aggregation
|   |
|   +-- monitoring/
|   |   +-- metrics.py            # Latency p50/p95/p99 + request counts
|   |
|   +-- utils/
|       +-- logger.py             # Structured JSON logging
|       +-- config.py             # Pydantic-based configuration
|
+-- tests/
    +-- test_embeddings.py
    +-- test_retrieval.py
    +-- test_ranking.py
    +-- test_api.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python scripts/generate_data.py --num_products 50000 --num_users 10000
```

Generates:
- **Product catalog** (50K): titles, categories, prices, ratings, popularity scores across 10 categories
- **User interactions** (300K+): views, clicks, add-to-cart, purchases with realistic event distributions (60% view, 25% click, 10% cart, 5% purchase)

### 3. Build product embeddings and FAISS index

```bash
python scripts/build_index.py --data_dir data/ --output_dir artifacts/
```

This step:
- Encodes all products into 128-dim vectors (text + numeric hybrid)
- Builds a FAISS IVF index for approximate nearest neighbor search
- Saves embeddings and index to artifacts/

### 4. Train re-ranking model

```bash
python scripts/train_ranker.py --data_dir data/ --output_dir artifacts/
```

Trains LightGBM on user interaction data with cross-features (price affinity, category match, popularity).

### 5. Start the recommendation server

```bash
python scripts/run_server.py --port 8000
```

### 6. Get recommendations

**Personalized recommendations:**
```bash
curl "http://localhost:8000/recommend?user_id=u_000001&k=10"
```

Response:
```json
{
  "user_id": "u_000001",
  "recommendations": [
    {
      "product_id": "p_004231",
      "title": "electronics product 4231",
      "category": "electronics",
      "price": 149.99,
      "ranking_score": 0.9312,
      "rank": 1
    },
    {
      "product_id": "p_012847",
      "title": "clothing product 12847",
      "category": "clothing",
      "price": 59.99,
      "ranking_score": 0.8745,
      "rank": 2
    }
  ],
  "latency_ms": 12.34
}
```

**Similar products:**
```bash
curl "http://localhost:8000/similar/p_000042?k=5"
```

**Record user feedback:**
```bash
curl -X POST "http://localhost:8000/feedback?user_id=u_000001&product_id=p_000042&event_type=click"
```

**Health check:**
```bash
curl http://localhost:8000/health
```

**Metrics:**
```bash
curl http://localhost:8000/metrics
```

---

## Docker

```bash
# Build and run with Redis
docker-compose up --build

# Or standalone
docker build -t recommendation-engine .
docker run -p 8000:8000 recommendation-engine
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/recommend` | Get top-K personalized recommendations for a user |
| `GET` | `/similar/{product_id}` | Find similar products by embedding similarity |
| `POST` | `/feedback` | Record user interaction (click, view, purchase) |
| `GET` | `/health` | Health check with uptime |
| `GET` | `/metrics` | Latency percentiles (p50/p95/p99) and request counts |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Two-stage (retrieval + ranking) | Scales to millions of items; retrieval narrows candidates cheaply |
| FAISS IVF index | Sub-linear search; handles 1M+ products with <5ms latency |
| Inner product metric | Works with L2-normalized embeddings; equivalent to cosine similarity |
| LightGBM re-ranker | Fast inference, handles tabular cross-features well |
| Weighted-average user embedding | Simple, effective, and fast to compute from interaction history |
| Sliding window user profiles | Captures recent behavior without unbounded memory growth |
| Category diversity penalty | Prevents recommendation bubbles; improves user engagement |
| Sentence-transformer with fallback | Uses hash encoding if transformers unavailable for lightweight dev |

---

## License

MIT
