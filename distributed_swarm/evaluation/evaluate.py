import argparse
import csv
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from distributed_swarm.config import load_config
from distributed_swarm.envs.wrappers import make_env
from distributed_swarm.training.ppo import build_policy
from .metrics import EpisodeMetrics


def evaluate(kind="2d", env_config=None, checkpoint=None, episodes=10, seed=1000, render_mode=None, video=None, stochastic=False):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = make_env(kind, **{**(env_config or {}), "render_mode": render_mode})
    policy = None
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["kind"] != kind:
            raise ValueError("2D and 3D policies are separate; checkpoint kind mismatch")
        if saved["obs_dim"] != env.observation_space.shape[-1]:
            raise ValueError("checkpoint observation size differs (check agent/obstacle counts)")
        continuous = saved.get("continuous", isinstance(env.action_space, gym.spaces.Box))
        policy = build_policy(saved["obs_dim"], saved["action_dim"], continuous,
                              env.num_agents, saved.get("training_mode", "shared"))
        policy.load_state_dict(saved["policy"]); policy.eval()
    rows = []
    video_writer = None
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video, fps=env.metadata["render_fps"], macro_block_size=1)
    try:
        for episode in range(episodes):
            torch.manual_seed(seed + episode)
            obs, info = env.reset(seed=seed + episode)
            env.action_space.seed(seed + episode)
            metrics = EpisodeMetrics()
            for _ in range(env.max_episode_steps):
                if policy is None:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy.act(torch.as_tensor(obs), deterministic=not stochastic)[0].numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                metrics.observe(env, info, reward)
                if video_writer is not None and episode == 0:
                    video_writer.append_data(env.render())
                if terminated or truncated:
                    break
            row = metrics.finish(env, info)
            row.update(seed=seed+episode, policy="checkpoint" if checkpoint else "random")
            row["action_selection"] = ("stochastic" if stochastic else "deterministic") if checkpoint else "random"
            rows.append(row)
    finally:
        env.close()
        if video_writer is not None:
            video_writer.close()
    return rows


def write_results(rows, output):
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for key in rows[0]:
        values = [r[key] for r in rows if isinstance(r[key], (int,float)) and r[key] is not None]
        if values:
            summary[key] = dict(mean=float(np.mean(values)), std=float(np.std(values)), n=len(values))
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None, default_kind="2d"):
    parser = argparse.ArgumentParser(description="Evaluate shared policies or an explicitly labelled random baseline")
    parser.add_argument("--kind", choices=["2d", "3d"], default=default_kind)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", default="results/evaluation.csv")
    parser.add_argument("--render", choices=["human", "rgb_array"])
    parser.add_argument("--video")
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions instead of choosing deterministic actions")
    args = parser.parse_args(argv)
    if args.config:
        env_config = load_config(args.config, args.kind)["env"]
    elif args.checkpoint:
        env_config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["env_config"]
    else:
        env_config = load_config(kind=args.kind)["env"]
    rows = evaluate(args.kind, env_config, args.checkpoint, args.episodes, args.seed,
                    "rgb_array" if args.video else args.render, args.video, stochastic=args.stochastic)
    summary = write_results(rows, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
