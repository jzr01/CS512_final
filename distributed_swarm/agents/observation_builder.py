"""Only this robot's sensors and delivered cache entries enter policy input."""
import numpy as np
from distributed_swarm.distributed.consistency import estimate
from distributed_swarm.distributed.failure_detector import STATUS_CODE

NEIGHBOR_WIDTH = 9  # relative xy, velocity xy, age, confidence, status, presence, usable


def neighbor_features(agent, own_position, now, num_agents, scale):
    features = []
    for robot in range(num_agents):
        if robot == agent.robot_id:
            continue
        entry = agent.cache.entries.get(robot)
        if entry is None:
            features.extend([0.] * NEIGHBOR_WIDTH)
            continue
        status = agent.detector.status(robot)
        position, age, confidence = estimate(entry, now, status, agent.config)
        usable = status != "UNAVAILABLE"
        relative = (position - own_position) / scale if usable else np.zeros(2)
        velocity = entry.velocity / scale if usable else np.zeros(2)
        features.extend([*relative, *velocity,
                         0. if agent.config.mode == "naive" else min(age / 20., 10.),
                         confidence, STATUS_CODE[status], 1., float(usable)])
    return np.asarray(features, dtype=np.float32)


def build_observation(agent, own_position, own_velocity, target, obstacles, now,
                      num_agents, map_size, obstacle_sensor_radius, proprioception=None):
    local = [*(own_position / map_size), *(own_velocity / map_size), *((target - own_position) / map_size)]
    # Obstacles are static local sensors, with masks; no robot ground truth accepted here.
    for center, radius in obstacles:
        delta = np.asarray(center) - own_position
        visible = np.linalg.norm(delta) - radius <= obstacle_sensor_radius
        local.extend([*((delta / map_size) if visible else (0., 0.)), radius / map_size if visible else 0., float(visible)])
    if proprioception is not None:
        local.extend(proprioception)
    return np.concatenate([np.asarray(local, np.float32),
                           neighbor_features(agent, own_position, now, num_agents, map_size)])
