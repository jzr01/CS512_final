from dataclasses import dataclass


@dataclass
class LamportClock:
    value: int = 0
    sequence_number: int = 0

    def send(self):
        self.value += 1
        self.sequence_number += 1
        return self.value, self.sequence_number

    def receive(self, timestamp):
        self.value = max(self.value, timestamp) + 1
