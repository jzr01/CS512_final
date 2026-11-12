"""Seeded, simulation-time transport. A queued packet is immutable in transit."""
from dataclasses import dataclass, field
import heapq
import numpy as np


@dataclass(frozen=True)
class RobotMessage:
    sender_id: int
    receiver_id: int
    sequence_number: int
    logical_timestamp: int
    simulation_timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    message_type: str = "state"


@dataclass
class NetworkConfig:
    packet_loss: float = 0.1
    min_delay: float = 0.0
    max_delay: float = 2.0
    communication_radius: float = 8.0
    # Reordering emerges from independent random delays; disable for FIFO links.
    reorder: bool = True
    # Half-open intervals [start, end); links fail in both directions.
    disconnections: list = field(default_factory=list)
    partitions: list = field(default_factory=list)
    crashes: list = field(default_factory=list)  # {robot_id, time}

    def __post_init__(self):
        if not 0 <= self.packet_loss <= 1:
            raise ValueError("packet_loss must be in [0,1]")
        if self.min_delay < 0 or self.max_delay < self.min_delay:
            raise ValueError("require 0 <= min_delay <= max_delay")
        if self.communication_radius <= 0:
            raise ValueError("communication_radius must be positive")


class CommunicationSimulator:
    def __init__(self, config=None, seed=None):
        self.config = config or NetworkConfig()
        self.rng = np.random.default_rng(seed)
        self.queue = []
        self.counter = 0
        self.last_arrival = {}
        self.stats = dict(sent=0, delivered=0, dropped=0, bytes_sent=0)

    def crashed(self, robot, now):
        return any(e["robot_id"] == robot and now >= e["time"] for e in self.config.crashes)

    def blocked(self, a, b, now):
        for e in self.config.disconnections:
            if e["start"] <= now < e["end"] and (a in e["robots"] or b in e["robots"]):
                return True
        for e in self.config.partitions:
            if e["start"] <= now < e["end"]:
                groups = e["groups"]
                if not any(a in g and b in g for g in groups):
                    return True
        return False

    def send(self, message, distance):
        now = message.simulation_timestamp
        a, b = message.sender_id, message.receiver_id
        self.stats["sent"] += 1
        self.stats["bytes_sent"] += 48 + message.position.nbytes + message.velocity.nbytes
        if (self.crashed(a, now) or self.crashed(b, now) or self.blocked(a, b, now)
                or distance > self.config.communication_radius or self.rng.random() < self.config.packet_loss):
            self.stats["dropped"] += 1
            return
        arrival = now + self.rng.uniform(self.config.min_delay, self.config.max_delay)
        if not self.config.reorder:
            arrival = max(arrival, self.last_arrival.get((a, b), 0.0))
        self.last_arrival[a, b] = arrival
        self.counter += 1
        # Copy payloads so later simulator updates cannot change queued state.
        packet = RobotMessage(a, b, message.sequence_number, message.logical_timestamp,
                              now, message.position.copy(), message.velocity.copy(), message.message_type)
        heapq.heappush(self.queue, (arrival, self.counter, packet))

    def deliver(self, now):
        ready = []
        while self.queue and self.queue[0][0] <= now:
            _, _, packet = heapq.heappop(self.queue)
            if self.crashed(packet.receiver_id, now) or self.blocked(packet.sender_id, packet.receiver_id, now):
                self.stats["dropped"] += 1
            else:
                ready.append(packet)
                self.stats["delivered"] += 1
        return ready
