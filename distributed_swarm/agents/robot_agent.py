from distributed_swarm.distributed.logical_clock import LamportClock
from distributed_swarm.distributed.neighbor_cache import NeighborCache
from distributed_swarm.distributed.failure_detector import FailureDetector
from distributed_swarm.distributed.communication import RobotMessage


class RobotAgent:
    def __init__(self, robot_id, config):
        self.robot_id, self.config = robot_id, config
        self.clock = LamportClock()
        self.cache = NeighborCache(config.reject_stale)
        self.detector = FailureDetector(config.heartbeat_timeout, config.unavailable_timeout, config.failure_detection)
        self.last_heartbeat_sent = float("-inf")

    def broadcast(self, now, own_position, own_velocity, num_agents):
        heartbeat = now - self.last_heartbeat_sent >= self.config.heartbeat_interval - 1e-9
        types = ["state", "heartbeat"] if heartbeat else ["state"]
        if heartbeat:
            self.last_heartbeat_sent = now
        for kind in types:
            for receiver in range(num_agents):
                if receiver == self.robot_id:
                    continue
                logical, sequence = self.clock.send()
                yield RobotMessage(self.robot_id, receiver, sequence,
                    logical if self.config.logical_clocks else 0, now,
                    own_position.copy(), own_velocity.copy(), kind)

    def receive(self, message, now):
        if self.config.logical_clocks:
            self.clock.receive(message.logical_timestamp)
        if message.message_type == "heartbeat":
            self.detector.heartbeat(message.sender_id, message.simulation_timestamp,
                                    message.sequence_number, now)
        else:
            self.cache.update(message)
