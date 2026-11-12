import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .wrappers import DistributedMixin
from distributed_swarm.agents.observation_builder import build_observation, NEIGHBOR_WIDTH


class Swarm2D(DistributedMixin, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, num_agents=5, map_size=20., max_episode_steps=200, robot_radius=.35,
                 collision_radius=.7, goal_radius=2., communication_radius=8.,
                 obstacle_sensor_radius=5., obstacles=None, network=None, consistency=None,
                 reward_weights=None, render_mode=None, speed=.4):
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("unsupported render_mode")
        self.init_distributed(num_agents, {"communication_radius": communication_radius, **(network or {})}, consistency)
        self.map_size, self.max_episode_steps = map_size, max_episode_steps
        self.robot_radius, self.collision_radius, self.goal_radius = robot_radius, collision_radius, goal_radius
        if map_size < 8 or speed <= 0 or max_episode_steps <= 0:
            raise ValueError("require map_size >= 8, speed > 0, max_episode_steps > 0")
        self.obstacle_sensor_radius, self.speed = obstacle_sensor_radius, speed
        self.obstacles = obstacles if obstacles is not None else [([map_size * .5, map_size * .5], 1.2)]
        self.target = np.array([map_size * .85, map_size * .85])
        self.weights = dict(progress=1., success=5., collision=1., disconnect=.2, step=.01)
        self.weights.update(reward_weights or {})
        self.render_mode = render_mode
        self.action_space = spaces.MultiDiscrete([5] * num_agents)
        width = 6 + 4 * len(self.obstacles) + NEIGHBOR_WIDTH * (num_agents - 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, (num_agents, width), np.float32)
        self._renderer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        positions = []
        for _ in range(self.num_agents):
            for attempt in range(10000):
                candidate = self.np_random.uniform(self.robot_radius + .1, self.map_size * .4, 2)
                if (all(np.linalg.norm(candidate - p) > self.collision_radius + .1 for p in positions)
                    and all(np.linalg.norm(candidate - c) > r + self.robot_radius for c, r in self.obstacles)):
                    positions.append(candidate)
                    break
            else:
                raise ValueError("could not place nonoverlapping robots; reduce num_agents")
        self.positions = np.asarray(positions)
        self.velocities = np.zeros_like(self.positions)
        self.physically_active = np.ones(self.num_agents, bool)
        self.trajectories = [[p.copy()] for p in self.positions]
        self.reset_distributed()
        return self._observations(), self.common_info(np.zeros(self.num_agents), 0)

    def _observations(self):
        return np.stack([build_observation(agent, self.positions[i], self.velocities[i], self.target,
            self.obstacles, self.now, self.num_agents, self.map_size, self.obstacle_sensor_radius)
            for i, agent in enumerate(self.agents)])

    def step(self, actions):
        actions = np.asarray(actions)
        if not self.action_space.contains(actions):
            raise ValueError("expected one discrete action in [0,4] per robot")
        # Faults take effect at this control interval's starting time.
        self.crashed = np.array([self.network.crashed(i, self.now) for i in range(self.num_agents)])
        old = self.positions.copy()
        before = np.linalg.norm(old - self.target, axis=-1)
        directions = np.array([[0, 1], [0, -1], [-1, 0], [1, 0], [0, 0]])
        proposed = old + directions[actions] * self.speed * self.active[:, None]
        proposed = np.clip(proposed, self.robot_radius, self.map_size - self.robot_radius)
        collided = np.zeros(self.num_agents, bool)
        for i, pos in enumerate(proposed):
            if any(np.linalg.norm(pos - c) < r + self.robot_radius for c, r in self.obstacles):
                collided[i] = True
        # Reject both participants; iterate because rejecting one can obstruct another.
        for _ in range(self.num_agents + 1):
            effective = np.where(collided[:, None], old, proposed)
            d = np.linalg.norm(effective[:, None] - effective[None, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            new = collided | (d < self.collision_radius - 1e-8).any(axis=1)
            if np.array_equal(new, collided):
                break
            collided = new
        self.positions = np.where(collided[:, None], old, proposed)
        self.velocities = self.positions - old
        self.steps += 1
        self.now = float(self.steps)
        self._communicate()
        after = np.linalg.norm(self.positions - self.target, axis=-1)
        reached_now = (before > self.goal_radius) & (after <= self.goal_radius) & self.active
        w = self.weights
        team = (w["success"] * np.mean(reached_now) - w["collision"] * np.mean(collided)
                - w["disconnect"] * (1 - self.connectivity()) - w["step"])
        rewards = w["progress"] * np.clip((before - after) / self.speed, -1, 1) + team
        rewards[~self.active] = 0.
        info = self.common_info(rewards, collided.sum())
        for i in range(self.num_agents):
            self.trajectories[i].append(self.positions[i].copy())
        terminated = info["success"] or not self.active.any()
        truncated = self.steps >= self.max_episode_steps and not terminated
        if self.render_mode == "human":
            self.render()
        return self._observations(), float(np.mean(rewards)), bool(terminated), bool(truncated), info

    def render(self):
        from distributed_swarm.visualization.render_2d import Renderer2D
        if self._renderer is None:
            self._renderer = Renderer2D(self.render_mode)
        return self._renderer.render(self)

    def close(self):
        if self._renderer:
            self._renderer.close()
