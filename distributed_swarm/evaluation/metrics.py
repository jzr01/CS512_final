"""Evaluation-only ground truth; never passed into the policy observation builder."""
import numpy as np
from distributed_swarm.distributed.consistency import estimate


def mean_or_none(values):
    return float(np.mean(values)) if values else None


class EpisodeMetrics:
    def __init__(self):
        self.ages, self.errors, self.connectivity = [], [], []
        self.collisions = 0
        self.steps = 0
        self.suspicion_ticks = 0
        self.false_suspicion_ticks = 0
        self.fall_times = {}
        self.known_before_crash = set()
        self.episode_return = 0.

    def observe(self, env, info, reward):
        self.steps += 1
        self.collisions += info["collisions"]
        self.connectivity.append(info["connectivity"])
        self.episode_return += reward
        for i in np.flatnonzero(~env.physically_active):
            self.fall_times.setdefault(int(i), env.now)
        for agent in env.agents:
            if not env.active[agent.robot_id]:
                continue
            for event in env.network_config.crashes:
                target = event["robot_id"]
                if env.now < event["time"] and target in agent.detector.neighbors:
                    self.known_before_crash.add((agent.robot_id, target))
            for robot, entry in agent.cache.entries.items():
                status = agent.detector.status(robot)
                prediction, age, _ = estimate(entry, env.now, status, agent.config)
                self.ages.append(age)
                self.errors.append(float(np.linalg.norm(prediction - env.positions[robot])))
                if status in ("SUSPECTED", "UNAVAILABLE"):
                    self.suspicion_ticks += 1
                    # Timeouts of live controllers (including partitioned ones) are false suspicions.
                    self.false_suspicion_ticks += int(not env.crashed[robot] and env.physically_active[robot])

    def finish(self, env, info):
        detection, recovery = [], []
        detected_pairs = 0
        eligible_pairs = 0
        for fault in env.network_config.crashes:
            target, when = fault["robot_id"], fault["time"]
            if when > env.now:
                continue
            for agent in env.agents:
                if (agent.robot_id, target) not in self.known_before_crash:
                    continue
                # Exclude observers that also crashed before they could detect.
                observer_end = min([e["time"] for e in env.network_config.crashes
                    if e["robot_id"] == agent.robot_id] + [env.now])
                if observer_end <= when:
                    continue
                eligible_pairs += 1
                events = [e["time"] for e in agent.detector.events if e["robot"] == target
                          and e["status"] in ("SUSPECTED", "UNAVAILABLE") and when <= e["time"] <= observer_end]
                if events:
                    detection.append(min(events) - when)
                    detected_pairs += 1
        for fault in [*env.network_config.disconnections, *env.network_config.partitions]:
            end = fault["end"]
            if end > env.now:
                continue
            for agent in env.agents:
                for event in agent.detector.events:
                    if event["status"] != "RECOVERED" or event["time"] < end:
                        continue
                    target = event["robot"]
                    affected = (agent.robot_id in fault.get("robots", []) or target in fault.get("robots", []))
                    if "groups" in fault:
                        affected = not any(agent.robot_id in g and target in g for g in fault["groups"])
                    if affected:
                        # First observed reintegration after this outage for this observer.
                        recovery.append(event["time"] - end)
                        break
        return dict(episode_return=self.episode_return, success=int(info["success"]),
            goal_reaching_rate=info["goal_reaching_rate"], episode_steps=self.steps,
            completion_time=env.now if info["success"] else None,
            connectivity=float(np.mean(self.connectivity)), collisions=self.collisions,
            collision_rate=self.collisions / max(1, self.steps * env.num_agents),
            average_message_age=mean_or_none(self.ages), estimation_error=mean_or_none(self.errors),
            stale_received=sum(a.cache.stale_received for a in env.agents),
            stale_rejected=sum(a.cache.stale_rejected for a in env.agents),
            messages_sent=env.network.stats["sent"], messages_delivered=env.network.stats["delivered"],
            messages_dropped=env.network.stats["dropped"], bytes_sent=env.network.stats["bytes_sent"],
            detection_latency=mean_or_none(detection), detection_eligible_pairs=eligible_pairs,
            detection_rate=detected_pairs / eligible_pairs if eligible_pairs else None,
            false_suspicion_rate=self.false_suspicion_ticks / self.suspicion_ticks if self.suspicion_ticks else 0.,
            recovery_latency=mean_or_none(recovery), recovery_observations=len(recovery),
            controller_crashes=int(env.crashed.sum()), physical_falls=len(self.fall_times))
