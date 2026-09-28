import time
import uuid

from fastapi import FastAPI, HTTPException, Query, Request

from src.utils.logger import get_logger
from src.monitoring.metrics import MetricsCollector

logger = get_logger(__name__)
app = FastAPI(title="Real-Time Recommendation Engine", version="1.0.0")

# Globals initialized by run_server.py
engine = None
user_profiles: dict = {}
user_histories: dict = {}
metrics = MetricsCollector()
start_time = time.time()


@app.middleware("http")
async def add_metadata(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Latency-Ms"] = f"{ms:.2f}"
    return response


@app.get("/recommend")
async def recommend(user_id: str, k: int = Query(default=10, le=100)):
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    import pandas as pd
    history = user_histories.get(user_id, pd.DataFrame())
    profile = user_profiles.get(user_id, {})

    start = time.perf_counter()
    results = engine.recommend(user_id, history, profile, k=k)
    latency = (time.perf_counter() - start) * 1000

    metrics.record_latency(latency)
    metrics.record_request("recommend", "success")

    return {"user_id": user_id, "recommendations": results, "latency_ms": round(latency, 2)}


@app.get("/similar/{product_id}")
async def similar(product_id: str, k: int = Query(default=10, le=50)):
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    results = engine.find_similar(product_id, k=k)
    return {"product_id": product_id, "similar_products": results}


@app.post("/feedback")
async def feedback(user_id: str, product_id: str, event_type: str = "click"):
    metrics.record_request("feedback", "success")
    return {"status": "recorded", "user_id": user_id, "product_id": product_id}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if engine else "not_initialized",
        "uptime_seconds": round(time.time() - start_time, 1),
    }


@app.get("/metrics")
async def get_metrics():
    return metrics.get_summary()
