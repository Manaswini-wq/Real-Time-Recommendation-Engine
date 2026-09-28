from src.monitoring.metrics import MetricsCollector


def test_metrics_collector():
    m = MetricsCollector()
    for i in range(100):
        m.record_latency(float(i))
    m.record_request("recommend", "success")
    m.record_request("recommend", "success")
    m.record_request("recommend", "error")

    summary = m.get_summary()
    assert summary["total_requests"] == 3
    assert summary["latency"]["p50"] > 0
    assert "recommend" in summary["endpoints"]
