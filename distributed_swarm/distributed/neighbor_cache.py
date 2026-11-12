from dataclasses import dataclass
import numpy as np


@dataclass
class NeighborState:
    position: np.ndarray
    velocity: np.ndarray
    sequence_number: int
    logical_timestamp: int
    last_update_time: float  # sender simulation timestamp, never receipt time


class NeighborCache:
    def __init__(self, filter_stale=True):
        self.entries = {}
        self.filter_stale = filter_stale
        self.high_water = {}
        self.stale_received = 0
        self.stale_rejected = 0

    def update(self, message):
        sender = message.sender_id
        stale = message.sequence_number <= self.high_water.get(sender, -1)
        self.stale_received += int(stale)
        if stale and self.filter_stale:
            self.stale_rejected += 1
            return False
        self.high_water[sender] = max(message.sequence_number, self.high_water.get(sender, -1))
        self.entries[sender] = NeighborState(message.position.copy(), message.velocity.copy(),
            message.sequence_number, message.logical_timestamp, message.simulation_timestamp)
        return True
