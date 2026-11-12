"""Parameter-shared PPO adapted from CleanRL's single-file PPO implementations.
See THIRD_PARTY_NOTICES.md for upstream attribution and MIT license.
"""
import argparse
import csv
from functools import partial
import json
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from distributed_swarm.config import load_config
from distributed_swarm.envs.wrappers import make_env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, continuous=False):
        super().__init__()
        self.continuous = continuous
        def net(output, gain):
            return nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, output), gain))
        self.actor, self.critic = net(action_dim, .01), net(1, 1.)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @staticmethod
    def log_jacobian(latent):
        # Stable log(1 - tanh(z)^2), including saturated torques.
        return 2 * (np.log(2) - latent - nn.functional.softplus(-2 * latent))

    def act(self, obs, latent=None, deterministic=False):
        logits = self.actor(obs)
        if self.continuous:
            distribution = Normal(logits, self.log_std.squeeze(0).clamp(-5, 2).exp().expand_as(logits))
            if latent is None:
                latent = logits if deterministic else distribution.sample()
            action = torch.tanh(latent)
            logprob = (distribution.log_prob(latent) - self.log_jacobian(latent)).sum(-1)
            # Monte Carlo entropy of the squashed distribution (not Gaussian entropy).
            sample = distribution.rsample()
            entropy = distribution.entropy().sum(-1) + self.log_jacobian(sample).sum(-1)
        else:
            distribution = Categorical(logits=logits)
            if latent is None:
                latent = logits.argmax(-1) if deterministic else distribution.sample()
            action, logprob, entropy = latent, distribution.log_prob(latent), distribution.entropy()
        return action, logprob, entropy, self.value(obs), latent


class IndependentPolicies(nn.Module):
    """One actor/critic per robot; no shared parameters or observation mixing."""
    def __init__(self, obs_dim, action_dim, continuous, num_agents):
        super().__init__()
        self.continuous = continuous
        self.policies = nn.ModuleList([Policy(obs_dim, action_dim, continuous) for _ in range(num_agents)])

    def actor(self, obs):
        return torch.stack([policy.actor(obs[..., i, :]) for i, policy in enumerate(self.policies)], dim=-2)

    def value(self, obs):
        return torch.stack([policy.value(obs[..., i, :]) for i, policy in enumerate(self.policies)], dim=-1)

    def act(self, obs, latent=None, deterministic=False):
        results = [policy.act(obs[..., i, :],
            None if latent is None else (latent[..., i, :] if self.continuous else latent[..., i]),
            deterministic) for i, policy in enumerate(self.policies)]
        return tuple(torch.stack([result[k] for result in results],
            dim=-2 if self.continuous and k in (0, 4) else -1) for k in range(5))


def build_policy(obs_dim, action_dim, continuous, num_agents=5, training_mode="shared"):
    if training_mode == "independent":
        return IndependentPolicies(obs_dim, action_dim, continuous, num_agents)
    if training_mode != "shared":
        raise ValueError("training_mode must be shared or independent")
    return Policy(obs_dim, action_dim, continuous)


def update_ppo(policy, optimizer, obs, latent, old_logprob, old_value, advantages, returns, cfg):
    """Update exactly one learner, using only the supplied learner-local batch."""
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = len(advantages)
    minibatch_size = max(1, batch_size // cfg["num_minibatches"])
    clipfracs = []
    for _ in range(cfg["update_epochs"]):
        indices = torch.randperm(batch_size, device=obs.device)
        for offset in range(0, batch_size, minibatch_size):
            mb = indices[offset:offset+minibatch_size]
            _, new_logprob, entropy, new_value, _ = policy.act(obs[mb], latent[mb])
            logratio = new_logprob - old_logprob[mb]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1).abs() > cfg["clip_coef"]).float().mean().item())
            pg_loss = torch.maximum(-advantages[mb] * ratio,
                -advantages[mb] * ratio.clamp(1 - cfg["clip_coef"], 1 + cfg["clip_coef"])).mean()
            clipped_value = old_value[mb] + (new_value - old_value[mb]).clamp(-cfg["clip_coef"], cfg["clip_coef"])
            value_loss = .5 * torch.maximum((new_value - returns[mb])**2,
                                           (clipped_value - returns[mb])**2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO loss")
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            optimizer.step()
        if cfg["target_kl"] is not None and approx_kl > cfg["target_kl"]:
            break
    variance = returns.var(unbiased=False)
    ev = (1 - (returns - old_value).var(unbiased=False) / variance).item() if variance > 0 else 0.
    return dict(policy_loss=pg_loss.item(), value_loss=value_loss.item(), entropy=entropy_loss.item(),
        approx_kl=approx_kl.item(), clipfrac=float(np.mean(clipfracs)), explained_variance=ev,
        loss=loss.item(), samples=batch_size)


def compute_gae(rewards, values, next_values, bootstrap_mask, trace_mask, gamma=.99, gae_lambda=.95):
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * bootstrap_mask[t] - values[t]
        last = delta + gamma * gae_lambda * trace_mask[t] * last
        advantages[t] = last
    return advantages, advantages + values


def train(kind="2d", argv=None):
    parser = argparse.ArgumentParser(description="CleanRL-derived parameter-shared PPO")
    parser.add_argument("--config")
    parser.add_argument("--training-mode", choices=["shared", "independent"], default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--async-envs", action="store_true", help="Separate simulator worker processes")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", help="Resume policy and optimizer; seed restarts simulator episodes")
    parser.add_argument("--track", action="store_true", help="Log metrics and config to Weights & Biases")
    parser.add_argument("--wandb-project-name", default="distributed-swarm")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--capture-video", action="store_true", help="Record periodic headless evaluation clips")
    parser.add_argument("--video-interval", type=int, default=100000, help="Robot transitions between clips")
    parser.add_argument("--video-length", type=int, default=200, help="Maximum control steps per clip")
    parser.add_argument("--video-seed", type=int, default=1000000, help="Fixed evaluation seed for comparable clips")
    args = parser.parse_args(argv)
    if args.video_interval < 1 or args.video_length < 1:
        parser.error("--video-interval and --video-length must be positive")
    config = load_config(args.config, kind)
    training_mode = args.training_mode or config.get("training", {}).get("training_mode", "shared")
    cfg = dict(learning_rate=3e-4, total_timesteps=500000, num_envs=4, num_steps=128,
        num_minibatches=4, update_epochs=4, gamma=.99, gae_lambda=.95, clip_coef=.2,
        ent_coef=.01 if kind == "2d" else 0., vf_coef=.5, max_grad_norm=.5, target_kl=.03)
    cfg.update(config.get("training", {}))
    for key in ("total_timesteps", "num_envs", "num_steps"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if min(cfg[k] for k in ("total_timesteps", "num_envs", "num_steps", "num_minibatches", "update_epochs")) < 1:
        raise ValueError("training counts must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    # Small MLP batches otherwise suffer excessive thread overhead on large CPU servers.
    torch.set_num_threads(1)
    run = Path(args.run_dir or f"runs/{kind}__seed{args.seed}__{time.time_ns()}")
    run.mkdir(parents=True, exist_ok=True)
    resolved = {"training_mode": training_mode, "kind": kind, "seed": args.seed, "env": config["env"], "training": cfg,
                "device": str(device), "torch_version": torch.__version__, "numpy_version": np.__version__}
    resolved["tracking"] = dict(enabled=args.track, project=args.wandb_project_name,
        entity=args.wandb_entity, name=args.wandb_run_name or run.name, mode=args.wandb_mode)
    resolved["video"] = dict(enabled=args.capture_video, interval=args.video_interval,
        length=args.video_length, seed=args.video_seed)
    (run / "config.json").write_text(json.dumps(resolved, indent=2))
    writer = SummaryWriter(str(run))
    vector_class = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vector_class([partial(make_env, kind, **config["env"]) for _ in range(cfg["num_envs"])],
                        autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    n = config["env"].get("num_agents", 5)
    obs_dim = envs.single_observation_space.shape[-1]
    continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    if continuous:
        action_dim = envs.single_action_space.shape[-1]
    else:
        if not isinstance(envs.single_action_space, gym.spaces.MultiDiscrete):
            raise TypeError("multi-agent discrete training requires MultiDiscrete actions")
        if not np.all(envs.single_action_space.nvec == envs.single_action_space.nvec[0]):
            raise ValueError("all robots must have the same discrete action count")
        action_dim = int(envs.single_action_space.nvec[0])
    policy = build_policy(obs_dim, action_dim, continuous, n, training_mode).to(device)
    learners = list(policy.policies) if training_mode == "independent" else [policy]
    optimizers = [torch.optim.Adam(learner.parameters(), lr=cfg["learning_rate"], eps=1e-5)
                  for learner in learners]
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["kind"] != kind or checkpoint["env_config"] != config["env"]:
            raise ValueError("resume requires matching environment kind and configuration")
        if checkpoint.get("training_mode", "shared") != training_mode:
            raise ValueError("resume requires matching training_mode; shared and independent policies differ")
        policy.load_state_dict(checkpoint["policy"])
        saved_optimizers = checkpoint["optimizer"] if training_mode == "independent" else [checkpoint["optimizer"]]
        if len(saved_optimizers) != len(optimizers):
            raise ValueError("checkpoint optimizer count differs from robot count")
        for optimizer, state in zip(optimizers, saved_optimizers):
            optimizer.load_state_dict(state)
        global_step = checkpoint["global_step"]
    obs, info = envs.reset(seed=args.seed)
    active = np.asarray(info["active_mask"])
    episode_return = np.zeros(cfg["num_envs"])
    tracking_run = None
    if args.track:
        try:
            import wandb
            tracking_run = wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                name=args.wandb_run_name or run.name, config=resolved, dir=str(run.resolve()),
                mode=args.wandb_mode)
            tracking_run.define_metric("global_step")
            tracking_run.define_metric("*", step_metric="global_step")
        except Exception:
            envs.close(); writer.close()
            raise
    next_video_step = global_step  # First clip after the first completed PPO update.
    started = time.monotonic()
    with (run / "metrics.csv").open("w", newline="") as stream:
        columns = ["global_step", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "explained_variance", "sps"]
        log = csv.DictWriter(stream, fieldnames=columns); log.writeheader()
        try:
            while global_step < cfg["total_timesteps"]:
                completed_returns, completed_successes = [], []
                buffers = {k: [] for k in ("obs", "latent", "logprob", "value", "reward", "next_value", "bootstrap", "trace", "valid")}
                for _ in range(cfg["num_steps"]):
                    observation = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        action, logprob, _, value, latent = policy.act(observation)
                    next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                    next_active = np.asarray(info["active_mask"])
                    ended = terminated | truncated
                    with torch.no_grad():
                        next_value = policy.value(torch.as_tensor(next_obs, dtype=torch.float32, device=device))
                    entries = dict(obs=observation, latent=latent, logprob=logprob, value=value,
                        reward=torch.as_tensor(info["agent_rewards"], device=device), next_value=next_value,
                        bootstrap=torch.as_tensor((~terminated[:, None]) & next_active, device=device),
                        trace=torch.as_tensor((~ended[:, None]) & next_active, device=device),
                        valid=torch.as_tensor(active.copy(), device=device))
                    for key, entry in entries.items():
                        buffers[key].append(entry)
                    global_step += cfg["num_envs"] * n
                    episode_return += reward
                    for idx in np.flatnonzero(ended):
                        writer.add_scalar("charts/episodic_return", episode_return[idx], global_step)
                        writer.add_scalar("charts/success", info["success"][idx], global_step)
                        completed_returns.append(float(episode_return[idx]))
                        completed_successes.append(float(info["success"][idx]))
                        episode_return[idx] = 0
                    if ended.any():
                        next_obs, reset_info = envs.reset(options={"reset_mask": ended.astype(bool)})
                        next_active = next_active.copy()
                        next_active[ended] = np.asarray(reset_info["active_mask"])[ended]
                    obs, active = next_obs, next_active
                    if global_step >= cfg["total_timesteps"]:
                        break
                b = {k: torch.stack(v) for k, v in buffers.items()}
                advantages, returns = compute_gae(b["reward"], b["value"], b["next_value"], b["bootstrap"], b["trace"], cfg["gamma"], cfg["gae_lambda"])
                mask = b["valid"].reshape(-1)
                flat_obs = b["obs"].reshape(-1, obs_dim)[mask]
                flat_latent = b["latent"].reshape((-1, action_dim) if continuous else (-1,))[mask]
                flat_logprob, flat_value = b["logprob"].reshape(-1)[mask], b["value"].reshape(-1)[mask]
                flat_adv, flat_return = advantages.reshape(-1)[mask], returns.reshape(-1)[mask]
                if len(flat_adv) < 1:
                    continue
                robot_ids = torch.arange(n, device=device).expand(b["valid"].shape).reshape(-1)[mask]
                learner_metrics = []
                robot_metrics = {}
                for robot, (learner, optimizer) in enumerate(zip(learners, optimizers)):
                    selected = robot_ids == robot if training_mode == "independent" else torch.ones_like(robot_ids, dtype=torch.bool)
                    if not selected.any():
                        continue  # Crashed/inactive robots never receive someone else's experience.
                    optimizer.param_groups[0]["lr"] = cfg["learning_rate"] * max(0.01, 1 - global_step / cfg["total_timesteps"])
                    result = update_ppo(learner, optimizer, flat_obs[selected], flat_latent[selected],
                        flat_logprob[selected], flat_value[selected], flat_adv[selected], flat_return[selected], cfg)
                    learner_metrics.append(result)
                    if training_mode == "independent":
                        for key, value in result.items():
                            robot_metrics[f"robots/{robot}/{key}"] = value
                row = {key: float(np.mean([result[key] for result in learner_metrics]))
                       for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "explained_variance")}
                row.update(global_step=global_step, sps=int(global_step / max(.001, time.monotonic() - started)))
                for key, value in robot_metrics.items():
                    writer.add_scalar(key, value, global_step)
                log.writerow(row); stream.flush()
                for key, value in row.items():
                    if key != "global_step":
                        writer.add_scalar("losses/" + key if key != "sps" else "charts/SPS", value, global_step)
                if tracking_run is not None:
                    metrics = {("charts/SPS" if key == "sps" else "losses/" + key): value
                               for key, value in row.items() if key != "global_step"}
                    metrics["global_step"] = global_step
                    metrics["charts/learning_rate"] = optimizers[0].param_groups[0]["lr"]
                    metrics.update(robot_metrics)
                    if completed_returns:
                        metrics["charts/episodic_return"] = float(np.mean(completed_returns))
                        metrics["charts/success"] = float(np.mean(completed_successes))
                    tracking_run.log(metrics)
                checkpoint = dict(policy=policy.state_dict(),
                    optimizer=[opt.state_dict() for opt in optimizers] if training_mode == "independent" else optimizers[0].state_dict(),
                    training_mode=training_mode, global_step=global_step,
                    kind=kind, obs_dim=obs_dim, action_dim=action_dim, continuous=continuous,
                    env_config=config["env"], config=resolved)
                torch.save(checkpoint, run / "checkpoint.pt")
                if args.capture_video and (global_step >= next_video_step or global_step >= cfg["total_timesteps"]):
                    from .video import record_policy_video
                    video_path = record_policy_video(policy, kind, config["env"],
                        run / "videos" / f"step_{global_step}.mp4", args.video_length, args.video_seed)
                    if tracking_run is not None:
                        tracking_run.log({"global_step": global_step,
                            "evaluation/video": wandb.Video(str(video_path), format="mp4",
                                caption=f"{kind} deterministic evaluation at {global_step} robot transitions")})
                    next_video_step = (global_step // args.video_interval + 1) * args.video_interval
                print(f"step={global_step} loss={np.mean([result['loss'] for result in learner_metrics]):.4f} sps={row['sps']}", flush=True)
        finally:
            envs.close(); writer.close()
            if tracking_run is not None:
                tracking_run.finish()
    return run / "checkpoint.pt"


if __name__ == "__main__":
    train()
