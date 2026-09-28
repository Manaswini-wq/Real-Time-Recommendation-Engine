import time
from collections import deque

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserEventProcessor:
    """Processes streaming user events and maintains sliding window profiles.

    Aggregates recent user behavior (clicks, views, purchases) into
    real-time feature updates for the recommendation engine.
    """

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self._user_events: dict[str, deque] = {}
        self._user_category_counts: dict[str, dict[str, int]] = {}

    def process_event(self, user_id: str, product_id: str,
                      event_type: str, category: str) -> dict:
        """Process a single user event and return updated profile signals."""
        if user_id not in self._user_events:
            self._user_events[user_id] = deque(maxlen=self.window_size)
            self._user_category_counts[user_id] = {}

        event = {
            "product_id": product_id,
            "event_type": event_type,
            "category": category,
            "timestamp": time.time(),
        }
        self._user_events[user_id].append(event)

        counts = self._user_category_counts[user_id]
        counts[category] = counts.get(category, 0) + 1

        return self._compute_signals(user_id)

    def _compute_signals(self, user_id: str) -> dict:
        """Compute real-time user signals from sliding window."""
        events = list(self._user_events.get(user_id, []))
        if not events:
            return {}

        recent = events[-10:]
        click_count = sum(1 for e in events if e["event_type"] == "click")
        purchase_count = sum(1 for e in events if e["event_type"] == "purchase")

        counts = self._user_category_counts.get(user_id, {})
        top_cat = max(counts, key=counts.get) if counts else "unknown"

        return {
            "recent_event_count": len(events),
            "recent_click_rate": click_count / max(len(events), 1),
            "recent_purchase_rate": purchase_count / max(len(events), 1),
            "top_category": top_cat,
            "session_length": len(recent),
        }

    def get_user_signals(self, user_id: str) -> dict:
        return self._compute_signals(user_id)
