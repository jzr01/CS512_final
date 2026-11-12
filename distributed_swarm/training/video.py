"""Record a bounded evaluation clip without touching training envs or RNG state."""
from pathlib import Path
import imageio.v2 as imageio
import torch
from distributed_swarm.envs.wrappers import make_env


def record_policy_video(policy, kind, env_config, output, length=200, seed=1000000):
    if length < 1:
        raise ValueError("video length must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = make_env(kind, **{**env_config, "render_mode": "rgb_array"})
    was_training = policy.training
    device = next(policy.parameters()).device
    try:
        obs, _ = env.reset(seed=seed)
        policy.eval()
        with imageio.get_writer(output, fps=env.metadata["render_fps"], macro_block_size=1) as writer:
            for _ in range(min(length, env.max_episode_steps)):
                with torch.inference_mode():
                    # Direct deterministic inference avoids drawing policy entropy samples.
                    logits = policy.actor(torch.as_tensor(obs, dtype=torch.float32, device=device))
                    action = torch.tanh(logits) if policy.continuous else logits.argmax(-1)
                obs, _, terminated, truncated, _ = env.step(action.cpu().numpy())
                writer.append_data(env.render())
                if terminated or truncated:
                    break
    finally:
        policy.train(was_training)
        env.close()
    return output
