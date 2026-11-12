from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from .wrappers import DistributedMixin
from .ant_model import build_ant_world
from distributed_swarm.agents.observation_builder import build_observation, NEIGHBOR_WIDTH


class SwarmAnt3D(DistributedMixin, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, num_agents=5, max_episode_steps=500, communication_radius=10.,
                 network=None, consistency=None, reward_weights=None, render_mode=None,
                 frame_skip=5, goal_radius=2., fall_as_failure=True,
                 control_mode="torque", low_level_checkpoint=None):
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("unsupported render_mode")
        if frame_skip < 1 or max_episode_steps < 1:
            raise ValueError("frame_skip and max_episode_steps must be positive")
        self.init_distributed(num_agents, {"communication_radius": communication_radius, **(network or {})}, consistency)
        xml = (Path(__file__).with_name("ant_world.xml").read_text() if num_agents == 5
               else build_ant_world(num_agents))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps, self.frame_skip = max_episode_steps, frame_skip
        self.render_mode, self.goal_radius, self.fall_as_failure = render_mode, goal_radius, fall_as_failure
        self.map_size = 20.
        self.target = np.array([16., 16.])
        self.obstacles = []
        self.weights = dict(progress=2., success=5., healthy=.05, collision=.2, disconnect=.1, control=.01)
        self.weights.update(reward_weights or {})
        if control_mode not in ("torque", "direction", "discrete_direction"):
            raise ValueError("control_mode must be torque, direction, or discrete_direction")
        self.control_mode = control_mode
        self.direction_controller = None
        if control_mode in ("direction", "discrete_direction"):
            from distributed_swarm.agents.ant_controller import AntDirectionController
            self.direction_controller = AntDirectionController(low_level_checkpoint)
        self.action_space = (spaces.MultiDiscrete([5] * num_agents)
            if control_mode == "discrete_direction" else
            spaces.Box(-1., 1., (num_agents, 2 if control_mode == "direction" else 8), np.float32))
        self.observation_space = spaces.Box(-np.inf, np.inf,
            (num_agents, 6 + 27 + NEIGHBOR_WIDTH * (num_agents - 1)), np.float32)
        self._renderer, self._viewer = None, None
        self.geom_owners = np.full(self.model.ngeom, -1, dtype=int)
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("ant"):
                self.geom_owners[g] = int(name.split("_")[0][3:])

    def _read_state(self):
        self.positions = np.stack([self.data.qpos[15*i:15*i+2].copy() for i in range(self.num_agents)])
        self.velocities = np.stack([self.data.qvel[14*i:14*i+2].copy() * (self.frame_skip * self.model.opt.timestep) for i in range(self.num_agents)])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.num_agents):
            q = self.data.qpos[15*i:15*(i+1)]
            q[:2] += self.np_random.uniform(-.2, .2, 2)
            q[7:] = np.array([0., 1., 0., -1., 0., -1., 0., 1.]) + self.np_random.uniform(-.05, .05, 8)
        self.data.qvel[:] = self.np_random.normal(0, .02, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self._read_state()
        self.physically_active = np.ones(self.num_agents, bool)
        self.reset_distributed()
        return self._observations(), self.common_info(np.zeros(self.num_agents), 0)

    def _observations(self):
        observations = []
        for i, agent in enumerate(self.agents):
            # 13 own qpos values (height, quaternion, joints) + 14 own qvel values.
            proprio = np.r_[self.data.qpos[15*i+2:15*(i+1)], self.data.qvel[14*i:14*(i+1)]]
            observations.append(build_observation(agent, self.positions[i], self.velocities[i],
                self.target, [], self.now, self.num_agents, self.map_size, 0., proprio))
        return np.stack(observations)

    def step(self, actions):
        raw_actions = np.asarray(actions)
        if raw_actions.shape != self.action_space.shape or not np.isfinite(raw_actions).all():
            raise ValueError(f"expected finite actions shaped {self.action_space.shape}")
        if self.control_mode == "discrete_direction":
            if (not np.equal(raw_actions, np.floor(raw_actions)).all()
                    or not self.action_space.contains(raw_actions.astype(np.int64))):
                raise ValueError("expected one discrete direction action in [0,4] per Ant")
            # Match Swarm2D: UP, DOWN, LEFT, RIGHT, STAY.
            direction_table = np.array([[0,1],[0,-1],[-1,0],[1,0],[0,0]], dtype=np.float32)
            commands = direction_table[raw_actions.astype(np.int64)]
        else:
            actions = np.clip(raw_actions.astype(np.float32), -1, 1)
            commands = actions.copy() if self.control_mode == "direction" else None
        if self.control_mode in ("direction", "discrete_direction"):
            actions = np.stack([self.direction_controller.torques(
                self.data.qpos[15*i:15*(i+1)], self.data.qvel[14*i:14*(i+1)], commands[i])
                for i in range(self.num_agents)])
        self.crashed = np.array([self.network.crashed(i, self.now) for i in range(self.num_agents)])
        before = np.linalg.norm(self.positions - self.target, axis=-1)
        pinned = {i: self.data.qpos[15*i:15*(i+1)].copy() for i in np.flatnonzero(self.crashed)}
        # Motor-disabled fallen robots remain physical; controller crashes are pinned obstacles.
        self.data.ctrl[:] = (actions * self.active[:, None]).ravel()
        collisions = set()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            for i, qpos in pinned.items():
                self.data.qpos[15*i:15*(i+1)] = qpos
                self.data.qvel[14*i:14*(i+1)] = 0.
            if pinned:
                mujoco.mj_forward(self.model, self.data)
            for contact in self.data.contact:
                a, b = self.geom_owners[contact.geom1], self.geom_owners[contact.geom2]
                if a >= 0 and b >= 0 and a != b:
                    collisions.add(tuple(sorted((int(a), int(b)))))
        self._read_state()
        healthy = np.array([.2 <= self.data.qpos[15*i+2] <= 1. for i in range(self.num_agents)])
        if self.fall_as_failure:
            self.physically_active &= healthy
        self.steps += 1
        # Communication time is in control ticks in both worlds; physical dt is separate.
        self.now = float(self.steps)
        self._communicate()
        after = np.linalg.norm(self.positions - self.target, axis=-1)
        reached = (before > self.goal_radius) & (after <= self.goal_radius) & self.active
        w = self.weights
        team = (w["success"] * np.mean(reached) - w["collision"] * len(collisions) / max(1, self.num_agents)
                - w["disconnect"] * (1 - self.connectivity()))
        dt = self.frame_skip * self.model.opt.timestep
        rewards = (w["progress"] * np.clip((before - after) / dt, -2., 2.)
                   + w["healthy"] * healthy - w["control"] * np.mean(actions**2, axis=-1) + team)
        rewards[~self.active] = 0.
        info = self.common_info(rewards, len(collisions))
        info["physical_time"] = float(self.data.time)
        info["healthy"] = healthy.copy()
        if commands is not None:
            info["direction_commands"] = commands
            info["motor_torques"] = (actions * self.active[:, None]).copy()
        terminated = info["success"] or not self.active.any()
        truncated = self.steps >= self.max_episode_steps and not terminated
        if self.render_mode == "human":
            self.render()
        return self._observations(), float(np.mean(rewards)), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "human":
            from mujoco import viewer
            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=600, width=800)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [9, 9, 0]
            camera.distance, camera.azimuth, camera.elevation = 30, 90, -65
            self._renderer.update_scene(self.data, camera=camera)
            return self._renderer.render()

    def close(self):
        if self._renderer:
            self._renderer.close()
        if self._viewer:
            self._viewer.close()
