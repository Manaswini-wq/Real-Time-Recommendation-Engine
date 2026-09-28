import json
import logging
import time
from contextlib import contextmanager
from typing import Generator


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        })


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def latency_tracker(operation: str, logger: logging.Logger | None = None) -> Generator[dict, None, None]:
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        log = logger or get_logger("latency")
        log.info(f"{operation}: {result['latency_ms']:.2f}ms")
