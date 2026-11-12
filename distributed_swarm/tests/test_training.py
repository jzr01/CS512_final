import csv
import numpy as np
import pytest
import torch
import yaml
from distributed_swarm.training.ppo import Policy, compute_gae, train
from distributed_swarm.evaluation.evaluate import evaluate


def test_squashed_gaussian_logprob_including_saturation():
    torch.set_num_threads(1)
    policy = Policy(3,8,True)
    obs = torch.zeros((4,3))
    latent = torch.tensor([[.1,1.,-2.,3.,-.1,10.,-10.,0.]]).repeat(4,1)
    action, logprob, entropy, _, _ = policy.act(obs,latent)
    dist = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(policy.actor(obs),policy.log_std.exp()),
        [torch.distributions.TanhTransform(cache_size=1)])
    # Independent reference away from saturation.
    safe = latent.clamp(-3,3)
    a, lp, _, _, _ = policy.act(obs,safe)
    torch.testing.assert_close(lp, dist.log_prob(a).sum(-1), atol=1e-4, rtol=1e-4)
    assert (action.abs() <= 1).all()
    assert torch.isfinite(logprob).all() and torch.isfinite(entropy).all()
    assert torch.equal(logprob,policy.act(obs,latent)[1])


def test_gae_bootstraps_truncation_but_stops_trace():
    rewards=torch.tensor([[1.],[100.]])
    values=torch.zeros_like(rewards)
    next_values=torch.tensor([[10.],[0.]])
    bootstrap=torch.tensor([[True],[False]])
    trace=torch.tensor([[False],[False]])
    adv,returns=compute_gae(rewards,values,next_values,bootstrap,trace,gamma=.9)
    torch.testing.assert_close(adv,torch.tensor([[10.],[100.]]))
    # Terminal transitions must not use the successor value.
    adv,_=compute_gae(rewards,values,next_values,torch.zeros_like(bootstrap),trace,gamma=.9)
    assert adv[0].item() == 1


@pytest.mark.parametrize("kind",["2d","3d"])
def test_training_reset_checkpoint_evaluation(tmp_path,kind):
    config={"env":{"num_agents":2,"max_episode_steps":2},"training":{"num_envs":2,"num_steps":4,
        "total_timesteps":32,"num_minibatches":2,"update_epochs":1}}
    path=tmp_path/'test.yaml'; path.write_text(yaml.safe_dump(config))
    checkpoint=train(kind,["--config",str(path),"--run-dir",str(tmp_path/'run'),"--device","cpu"])
    saved=torch.load(checkpoint,weights_only=False)
    assert saved["global_step"] == 32
    rows=evaluate(kind,config["env"],checkpoint,episodes=2,seed=100)
    assert len(rows) == 2
    assert all(np.isfinite(row["episode_return"]) for row in rows)
    with (tmp_path/'run/metrics.csv').open() as stream:
        metrics=list(csv.DictReader(stream))
    assert all(np.isfinite(float(row["policy_loss"])) for row in metrics)


@pytest.mark.parametrize("kind", ["2d", "3d"])
def test_wandb_tracking_logs_and_finishes(tmp_path, monkeypatch, kind):
    import sys
    from types import SimpleNamespace
    calls = {"metrics": [], "finished": False}

    class Run:
        def define_metric(self, *args, **kwargs):
            pass

        def log(self, metrics):
            calls["metrics"].append(metrics)

        def finish(self):
            calls["finished"] = True

    def init(**kwargs):
        calls["init"] = kwargs
        return Run()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"env": {"num_agents": 2, "max_episode_steps": 2},
        "training": {"num_envs": 1, "num_steps": 2, "total_timesteps": 4,
                     "num_minibatches": 1, "update_epochs": 1}}))
    checkpoint = train(kind, ["--config", str(path), "--run-dir", str(tmp_path / "run"),
        "--device", "cpu", "--track", "--wandb-project-name", "test-project",
        "--wandb-entity", "test-team", "--wandb-mode", "offline"])
    assert checkpoint.exists()
    assert calls["init"]["project"] == "test-project"
    assert calls["init"]["entity"] == "test-team"
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["config"]["kind"] == kind
    assert calls["metrics"][0]["global_step"] == 4
    assert "losses/policy_loss" in calls["metrics"][0]
    assert "charts/episodic_return" in calls["metrics"][0]
    assert calls["finished"]


def test_headless_video_preserves_rng_and_policy_mode(tmp_path, monkeypatch):
    from distributed_swarm.training.video import record_policy_video
    import imageio.v2 as imageio
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    torch.manual_seed(10)
    policy = Policy(15, 5)  # Two robots, no obstacles: 6 + 9 features.
    policy.train()
    rng = torch.get_rng_state().clone()
    output = record_policy_video(policy, "2d", {"num_agents": 2, "obstacles": [],
        "max_episode_steps": 4}, tmp_path / "clip.mp4", length=2)
    assert policy.training
    assert torch.equal(rng, torch.get_rng_state())
    with imageio.get_reader(output) as reader:
        assert reader.count_frames() == 2
        assert reader.get_data(0).shape == (720, 720, 3)


def test_training_schedules_and_logs_videos(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from distributed_swarm.training import video
    recorded, logged = [], []

    def record(policy, kind, config, output, length, seed):
        recorded.append((output.name, length, seed))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        return output

    class Run:
        def define_metric(self, *args, **kwargs):
            pass
        def log(self, metrics):
            logged.append(metrics)
        def finish(self):
            pass

    monkeypatch.setattr(video, "record_policy_video", record)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kw: Run(),
        Video=lambda path, **kw: {"path": path, **kw}))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"env": {"num_agents": 2, "max_episode_steps": 2},
        "training": {"num_envs": 1, "num_steps": 2, "total_timesteps": 12,
                     "num_minibatches": 1, "update_epochs": 1}}))
    train("2d", ["--config", str(path), "--run-dir", str(tmp_path / "run"),
        "--device", "cpu", "--track", "--capture-video", "--video-interval", "10",
        "--video-length", "3", "--video-seed", "123"])
    assert recorded == [("step_4.mp4", 3, 123), ("step_12.mp4", 3, 123)]
    video_logs = [row for row in logged if "evaluation/video" in row]
    assert [row["global_step"] for row in video_logs] == [4, 12]


@pytest.mark.parametrize("continuous", [False, True])
def test_independent_learner_update_is_isolated(continuous):
    from distributed_swarm.training.ppo import build_policy, update_ppo
    torch.set_num_threads(1)
    group = build_policy(3, 2 if continuous else 5, continuous, 2, "independent")
    first, second = group.policies
    first_ids = {id(p) for p in first.parameters()}
    assert first_ids.isdisjoint({id(p) for p in second.parameters()})
    before_first = {k:v.clone() for k,v in first.state_dict().items()}
    before_second = {k:v.clone() for k,v in second.state_dict().items()}
    obs = torch.randn(8,3)
    with torch.no_grad():
        _,logprob,_,value,latent = first.act(obs)
    optimizer = torch.optim.Adam(first.parameters(), lr=.01)
    cfg = dict(num_minibatches=2,update_epochs=1,clip_coef=.2,vf_coef=.5,
               ent_coef=.01,max_grad_norm=.5,target_kl=None)
    update_ppo(first,optimizer,obs,latent,logprob,value,torch.randn(8),torch.randn(8),cfg)
    assert any(not torch.equal(v,first.state_dict()[k]) for k,v in before_first.items())
    assert all(torch.equal(v,second.state_dict()[k]) for k,v in before_second.items())
    batch = torch.randn(4,2,3)
    perturbed = batch.clone(); perturbed[:,1] += 100
    assert torch.equal(group.actor(batch)[:,0],group.actor(perturbed)[:,0])


@pytest.mark.parametrize("kind", ["2d", "3d"])
def test_independent_train_resume_evaluate(tmp_path,kind):
    config = {"env":{"num_agents":2,"max_episode_steps":2},"training":{"num_envs":2,
        "num_steps":4,"total_timesteps":32,"num_minibatches":2,"update_epochs":1}}
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    checkpoint=train(kind,["--config",str(path),"--run-dir",str(tmp_path/'run'),
        "--training-mode","independent","--device","cpu"])
    saved=torch.load(checkpoint,weights_only=False)
    assert saved["training_mode"] == "independent"
    assert len(saved["optimizer"]) == 2
    assert all(any(k.startswith(f'policies.{i}.') for k in saved['policy']) for i in range(2))
    rows=evaluate(kind,config["env"],checkpoint,episodes=1)
    assert np.isfinite(rows[0]['episode_return'])
    resumed=train(kind,["--config",str(path),"--run-dir",str(tmp_path/'resumed'),
        "--training-mode","independent","--device","cpu","--resume",str(checkpoint),
        "--total-timesteps","48"])
    assert torch.load(resumed,weights_only=False)['global_step'] == 48


def test_3d_discrete_direction_uses_categorical_independent_ppo(tmp_path):
    config = {"env": {"num_agents": 2, "max_episode_steps": 2,
                      "control_mode": "discrete_direction",
                      "fall_as_failure": False},
              "training": {"training_mode": "independent", "num_envs": 1,
                           "num_steps": 2, "total_timesteps": 8,
                           "num_minibatches": 1, "update_epochs": 1}}
    path = tmp_path / "direction.yaml"
    path.write_text(yaml.safe_dump(config))
    checkpoint = train("3d", ["--config", str(path), "--device", "cpu",
                              "--run-dir", str(tmp_path / "run")])
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["continuous"] is False
    assert saved["action_dim"] == 5
    assert saved["training_mode"] == "independent"
    rows = evaluate("3d", config["env"], checkpoint, episodes=1)
    assert np.isfinite(rows[0]["episode_return"])
