"""Gymnasium uses scalar team rewards; info preserves per-agent training rewards."""
import gymnasium as gym
import numpy as np
from distributed_swarm.distributed.communication import CommunicationSimulator, NetworkConfig
from distributed_swarm.distributed.consistency import ConsistencyConfig
from distributed_swarm.agents.robot_agent import RobotAgent


class DistributedMixin:
    def init_distributed(self, num_agents, network, consistency):
        if num_agents < 1:
            raise ValueError("num_agents must be positive")
        self.num_agents = num_agents
        self.network_config = NetworkConfig(**(network or {}))
        self.consistency_config = ConsistencyConfig(**(consistency or {}))

    def reset_distributed(self):
        seed = int(self.np_random.integers(0, 2**32))
        self.network = CommunicationSimulator(self.network_config, seed)
        self.agents = [RobotAgent(i, self.consistency_config) for i in range(self.num_agents)]
        self.crashed = np.zeros(self.num_agents, dtype=bool)
        self.steps = 0
        self.now = 0.
        self._communicate()

    def _communicate(self):
        self.crashed = np.array([self.network.crashed(i, self.now) for i in range(self.num_agents)])
        for i, agent in enumerate(self.agents):
            if self.crashed[i] or not self.physically_active[i]:
                continue
            for message in agent.broadcast(self.now, self.positions[i], self.velocities[i], self.num_agents):
                distance = np.linalg.norm(self.positions[i] - self.positions[message.receiver_id])
                self.network.send(message, distance)
        for message in self.network.deliver(self.now):
            if self.physically_active[message.receiver_id]:
                self.agents[message.receiver_id].receive(message, self.now)
        for agent in self.agents:
            agent.detector.tick(self.now)

    @property
    def active(self):
        return ~(self.crashed) & self.physically_active

    def connectivity(self):
        ids = np.flatnonzero(self.active)
        if len(ids) <= 1:
            return 1.
        positions = self.positions[ids]
        adjacency = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1) <= self.network_config.communication_radius
        seen, pending = {0}, [0]
        while pending:
            for j in np.flatnonzero(adjacency[pending.pop()]):
                if int(j) not in seen:
                    seen.add(int(j)); pending.append(int(j))
        return len(seen) / len(ids)

    def common_info(self, rewards, collisions):
        distance = np.linalg.norm(self.positions - self.target, axis=-1)
        reached = distance <= self.goal_radius
        return dict(agent_rewards=np.asarray(rewards, np.float32), active_mask=self.active.copy(),
                    success=bool(self.active.any() and np.all(reached[self.active])),
                    goal_reaching_rate=float(np.mean(reached[self.active])) if self.active.any() else 0.,
                    connectivity=self.connectivity(), collisions=int(collisions),
                    crashed=self.crashed.copy(), fallen=(~self.physically_active).copy(),
                    network=dict(self.network.stats), simulation_time=self.now)


def make_env(kind="2d", **kwargs):
    if kind == "2d":
        from .swarm_2d import Swarm2D
        return Swarm2D(**kwargs)
    if kind == "3d":
        from .swarm_ant_3d import SwarmAnt3D
        return SwarmAnt3D(**kwargs)
    raise ValueError("environment must be 2d or 3d")
