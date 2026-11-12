from dataclasses import dataclass
import numpy as np


@dataclass
class ConsistencyConfig:
    mode: str = "confidence"
    lambda_decay: float = 0.25
    predict_position: bool = False
    max_prediction_age: float = 8.0
    filter_stale: bool | None = None
    logical_clocks: bool = True
    failure_detection: bool = True
    heartbeat_interval: float = 1.0
    heartbeat_timeout: float = 4.0
    unavailable_timeout: float = 8.0

    def __post_init__(self):
        if self.mode not in ("naive", "timestamp", "confidence"):
            raise ValueError("mode must be naive, timestamp, or confidence")
        if self.lambda_decay < 0 or self.heartbeat_interval <= 0 or self.max_prediction_age < 0:
            raise ValueError("invalid consistency timing")

    @property
    def reject_stale(self):
        return self.mode != "naive" if self.filter_stale is None else self.filter_stale


def estimate(entry, now, status, config):
    age = max(0.0, now - entry.last_update_time)
    confidence = float(np.exp(-config.lambda_decay * age)) if config.mode == "confidence" else 1.0
    if status == "UNAVAILABLE":
        confidence = 0.0
    elif status == "SUSPECTED":
        confidence *= .25
    position = entry.position.copy()
    if config.predict_position:
        position += entry.velocity * min(age, config.max_prediction_age)
    return position, age, confidence
