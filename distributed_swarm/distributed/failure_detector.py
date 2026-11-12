"""Timeouts express suspicion, never proof of a crash."""
from dataclasses import dataclass

STATUS_CODE = {"UNKNOWN": 0., "ACTIVE": .25, "SUSPECTED": .5, "UNAVAILABLE": .75, "RECOVERED": 1.}


@dataclass
class NeighborHealth:
    last_heartbeat: float
    status: str = "ACTIVE"
    suspected_since: float | None = None
    heartbeat_sequence: int = -1


class FailureDetector:
    def __init__(self, timeout=4.0, unavailable_timeout=8.0, enabled=True):
        if not 0 < timeout < unavailable_timeout:
            raise ValueError("require 0 < timeout < unavailable_timeout")
        self.timeout, self.unavailable_timeout = timeout, unavailable_timeout
        self.enabled = enabled
        self.neighbors = {}
        self.events = []

    def heartbeat(self, robot, sent_time, sequence, now):
        old = self.neighbors.get(robot)
        if old and sequence <= old.heartbeat_sequence:
            return
        status = "ACTIVE"
        if old and old.status in ("SUSPECTED", "UNAVAILABLE"):
            status = old.status
            if now - sent_time <= self.timeout:
                status = "RECOVERED"
                self.events.append(dict(robot=robot, time=now, status=status))
        self.neighbors[robot] = NeighborHealth(sent_time, status, None, sequence)

    def tick(self, now):
        if not self.enabled:
            return
        for robot, h in self.neighbors.items():
            age = now - h.last_heartbeat
            status = ("UNAVAILABLE" if age > self.unavailable_timeout else
                      "SUSPECTED" if age > self.timeout else "ACTIVE")
            if h.status != status:
                if status in ("SUSPECTED", "UNAVAILABLE"):
                    self.events.append(dict(robot=robot, time=now, status=status))
                h.status = status
                if status == "SUSPECTED":
                    h.suspected_since = now

    def status(self, robot):
        if not self.enabled:
            return "ACTIVE"
        h = self.neighbors.get(robot)
        return h.status if h else "UNKNOWN"
