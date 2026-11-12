# Fault-Tolerant Distributed Robot Swarm Control

This project studies robot navigation and coordination when communication is delayed, lost, reordered, or temporarily unavailable, and when robot controllers crash. It combines local reinforcement-learning policies with per-robot neighbor caches, sequence filtering, Lamport clocks, confidence features, and heartbeat failure detection.

There are two simulation environments: a discrete 2D navigation world and a shared-world MuJoCo Ant environment. The saved results documented here are **2D stochastic-policy evaluations for 5, 10, and 20 robots**. The 3D setup and commands are included, but those 2D numbers are not evidence of 3D convergence.

## Contents

- [Environment setup and task](#environment-setup-and-task)
- [Distributed communication method](#distributed-communication-method)
- [Reinforcement-learning method](#reinforcement-learning-method)
- [Installation and training](#installation-and-training)
- [Evaluation and experimental protocol](#evaluation-and-experimental-protocol)
- [Metrics](#metrics)
- [Measured 2D results](#measured-2d-results)
- [Videos and trajectory demonstrations](#videos-and-trajectory-demonstrations)
- [Limitations and next experiments](#limitations-and-next-experiments)
- [Project layout and validation](#project-layout-and-validation)

## Environment setup and task

### Task and success

Every robot navigates toward a common goal disk. Each policy acts from its own state and delivered neighbor information. Robots must coordinate movement around each other; the 2D task also includes a static obstacle.

At the end of each control step:

- An active robot is a robot that has neither crashed nor become physically inactive.
- **Team success** means at least one robot is active and every active robot is inside the goal disk simultaneously.
- **Goal-reaching fraction** is the fraction of currently active robots inside that disk.
- Success terminates the episode. No active robots also terminates it, as a failure.
- Otherwise, the episode ends by timeout at its configured horizon.

Reaching the goal once is not enough if a robot subsequently leaves before its teammates arrive. Crashed robots are excluded from the success denominator but remain physical obstructions. Success therefore means completion by the surviving team, not preservation of the entire initial team.

### 2D navigation world

| Setting | 5 robots | 10 robots | 20 robots |
| --- | ---: | ---: | ---: |
| Map extent | 20 × 20 | 30 × 30 | 45 × 45 |
| Episode limit, control steps | 200 | 300 | 450 |
| Goal center | (17, 17) | (25.5, 25.5) | (38.25, 38.25) |
| Goal radius | 2 | 3 | 4 |
| Communication radius | 8 | 12 | 16 |
| Obstacle sensor radius | 5 | 7.5 | 10 |
| Robot radius | 0.35 | 0.35 | 0.35 |
| Robot collision distance | 0.7 | 0.7 | 0.7 |
| Movement per action | 0.4 | 0.4 | 0.4 |
| Observation width per robot | 46 | 91 | 181 |

These values describe the environments used by the saved evaluations. Resolved run configs and checkpoint environment settings take precedence over preset defaults.

Robots start at seeded, random, nonoverlapping positions in the lower-left portion of the map. The goal is fixed at 85% of each map axis. By default, there is one circular obstacle at the map center with radius 1.2. The map layout and goal are not randomized in these evaluations; the reset seeds vary starts and communication randomness.

Each robot selects one categorical action:

| Action | Meaning | Displacement direction |
| --- | --- | --- |
| 0 | UP | (0, +1) |
| 1 | DOWN | (0, −1) |
| 2 | LEFT | (−1, 0) |
| 3 | RIGHT | (+1, 0) |
| 4 | STAY | (0, 0) |

The joint action space is `MultiDiscrete([5] * num_agents)`: one choice among five actions per robot, not five continuous controls. Obstacle and robot collisions reject movement; both participants can be blocked. Crashed robots stay in place and still participate in physical collision checks.

The 2D per-robot reward is:

```text
progress_i = clip((previous_distance_i − current_distance_i) / speed, −1, 1)

team_reward =
    5 × mean(goal_entry_events)
  − 1 × mean(collided_robot_flags)
  − 0.2 × (1 − connectivity)
  − 0.01

reward_i = progress_i + team_reward
```

A goal-entry event is an outside-to-inside crossing during the current step; the implementation does not permanently latch a robot's first arrival. Inactive robots receive zero reward. The Gymnasium environment returns the mean reward, while PPO learns from the individual rewards in `info["agent_rewards"]`.

### 3D shared-world Ant environment

All Ants occupy one MuJoCo world and physically interact. There are 15 generalized position values, 14 generalized velocity values, and 8 actuators per Ant. The checked-in `ant_world.xml` holds five Ants; other team sizes are generated from `ant_template.xml`.

| Setting | Current implementation |
| --- | --- |
| Team sizes supported by direction presets | 5, 10, 20 |
| Goal center and radius | (16, 16), radius 2 |
| Episode limit | 500 control steps |
| Communication radius | 10 |
| Physics substeps per control step | 5 |
| Default physical time per control step | 0.05 seconds |
| Observation scale | Fixed map scale of 20; not an enclosing wall |
| Static walls, pillars, or obstacle field | None |
| Background and floor appearance | White |
| Fall criterion | Torso height outside [0.2, 1.0] |
| Direction-policy observation widths | 69, 114, 204 for 5, 10, 20 Ants |

Ant resets perturb the initial positions, joints, and velocities. Other team sizes use a generated grid of starting positions; increasing the team size does not automatically enlarge the goal, communication radius, or observation scale.

Three control modes are implemented:

| Mode | Per-robot action |
| --- | --- |
| `torque` | 8 continuous controls in [−1, 1] |
| `direction` | 2 continuous world-frame direction components |
| `discrete_direction` | The five categorical directions listed above |

The direction presets use `discrete_direction`. A frozen pretrained Ant gait converts a direction into eight joint torques, with local heading steering. STAY uses posture control; it is not a simulator freeze or a guarantee of zero displacement. No external force or teleportation moves the robot toward its command.

The pretrained gait is a forward-locomotion policy from [sdpkjc/Ant-v4-ppo_fix_continuous_action-seed2](https://huggingface.co/sdpkjc/Ant-v4-ppo_fix_continuous_action-seed2), pinned by the downloader to revision `b88f77df64955077718a67c826f2dc290dfbe4e9`. It is not a pretrained navigation or arbitrary-direction policy. Its normalization statistics and actor weights remain frozen while high-level PPO learns. Download metadata records the source and SHA-256 digest.

The 3D reward uses clipped distance-progress velocity (weight 2), goal crossings (5), healthy-height reward (0.05), mean squared torque cost (0.01), inter-Ant contact-pair cost (0.2), and disconnection cost (0.1). Floor contact is not counted as an inter-robot collision.

With `fall_as_failure: true`, a robot becomes permanently inactive for that episode after failing the height check; its motors are disabled and its body remains dynamic. A controller crash instead pins the current pose and restores zero velocity each physics substep. This pinned-body model is an approximation, not a welded constraint.

## Distributed communication method

### Per-robot state and simulated transport

Each robot owns a message generator, logical clock, sequence numbers, neighbor cache, and heartbeat detector. Communication is an in-process, seeded discrete-event simulation rather than a real network of robot processes.

Every packet is an immutable snapshot containing sender and receiver IDs, sequence number, logical timestamp, simulation timestamp, xy position, xy velocity, and message type. A priority queue delivers packets in scheduled arrival order.

- A transmission outside the communication radius is dropped.
- Packet loss is sampled independently during transmission; delay is sampled uniformly between configured bounds.
- Independent delays can reorder packets. Disabling reordering preserves FIFO order on each directed link.
- Outages and partitions block both directions of affected links.
- Links are checked for outage/partition blocking at send and delivery time; range is checked at send time.
- Messages sent before a sender crashes can still arrive afterward.
- Communication is direct robot-to-robot; the implementation does not relay state through intermediate robots.

The baseline settings use 10% packet loss, delays from 0 to 2 control ticks, and reordering enabled. Thus, “baseline” does not mean a perfect network.

Communication time uses control ticks in both worlds. MuJoCo physical seconds are separate. The 2D xy velocity and communicated 3D xy velocity are expressed as displacement per control tick, matching the age used for position prediction.

### Ordering, consistency, and failure detection

| Mechanism | Implementation and purpose |
| --- | --- |
| Sequence filtering | Reject old state packets by per-sender sequence number |
| Heartbeat ordering | Reject non-increasing heartbeat sequences |
| Lamport clocks | Increment at send; receive updates to max(local, remote) + 1 |
| Timestamp age | Current simulation time minus sender simulation timestamp |
| Confidence | exp(−0.25 × age) in confidence mode |
| Suspected neighbor | Confidence multiplied by 0.25 |
| Unavailable neighbor | Zero usable position/velocity features and zero confidence |
| Optional predictor | Cached position + cached velocity × min(age, 8) |

Age uses simulation timestamps, not Lamport timestamps or receipt time. Logical clocks are instrumentation: changing only the clock switch does not change filtering or policy features.

The three consistency modes are:

- **Naive:** stale-state filtering disabled by default, age feature hidden, unit base confidence.
- **Timestamp:** stale-state filtering enabled and age exposed, unit base confidence.
- **Confidence:** filtering and age exposed, with exponentially decaying confidence.

An explicit `filter_stale` setting overrides the mode default. Confidence is a policy input; it does not automatically blend neighbor positions into a consensus estimate.

Heartbeats are scheduled every tick. For a known neighbor, heartbeat age greater than 4 ticks triggers SUSPECTED, and greater than 8 triggers UNAVAILABLE. A sufficiently fresh heartbeat can record recovery. A robot initially knows no neighbors, so no heartbeat history is not the same as a previously known neighbor timing out. Detectors use received data, never the ground-truth crash schedule.

In 2D videos, blue means no receiver currently suspects the robot, yellow means at least one receiver considers it suspected or unavailable, and red means an actual controller crash. Range loss alone can therefore produce yellow robots even while they move normally.

### Local observations and decentralization

| Feature group | Width |
| --- | ---: |
| Own xy, own xy velocity, relative target xy | 6 |
| Each static obstacle: relative xy, radius, visibility mask | 4 |
| Ant only: own height, quaternion, joints, generalized velocities | 27 |
| Each other robot ID: relative cached xy, cached xy velocity, age, confidence, status, presence and usability | 9 |

Own planar values and obstacle features are scaled by map size. Unseen neighbors occupy zero-padded, stable ID slots. Only locally sensed obstacles and delivered neighbor cache entries enter the observation. The actor cannot access another robot's live ground-truth position.

Global state is used by simulation, routing, collision handling, reward computation, and evaluation. In particular, the reward contains a team connectivity term. Decentralized policy inputs do not imply entirely local reward generation.

## Reinforcement-learning method

PPO is adapted from CleanRL; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- **Shared PPO:** one actor, critic, and optimizer; robot-local observations and trajectories share parameters.
- **Independent PPO:** each robot ID has its own actor, critic, optimizer, and experience subset.
- Both actor and critic use two 64-unit tanh hidden layers.
- Discrete actions use categorical distributions; continuous actions use tanh-squashed Gaussians.
- Independent learners never optimize on another robot's trajectory.
- Inactive samples are excluded. Time-limit truncations bootstrap the final state; terminal and inactive trajectories stop bootstrapping.
- PPO uses clipped policy/value objectives, generalized advantage estimation, entropy regularization, gradient clipping, learning-rate decay, and KL-based early stopping.

Default common parameters are learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clipping 0.2, value coefficient 0.5, gradient norm limit 0.5, and target KL 0.03. Saved 2D runs used entropy coefficient 0.01, 4 epochs, and 8 minibatches. Hierarchical 3D presets use entropy coefficient 0, 10 epochs, and 16 minibatches.

### Step accounting and PPO updates

`--total-timesteps` counts **robot transitions across all environments**, including inactive slots:

```text
counter increment per vector step = num_envs × num_agents
transitions per full rollout = num_envs × num_steps × num_agents
samples per independent policy ≈ num_envs × num_steps
```

The final rollout can be shorter to meet the budget; the counter can overshoot by one vector step. Inactivity reduces valid learning samples.

The current recommended 3D commands explicitly select 4 environments and 256 steps per environment:

| Robots | Total rollout transitions | Samples per independent policy |
| --- | ---: | ---: |
| 5 | 5,120 | 1,024 |
| 10 | 10,240 | 1,024 |
| 20 | 20,480 | 1,024 |

Each full independent batch is normally divided into 16 minibatches of 64 and reused for up to 10 epochs. KL stopping can reduce actual optimizer steps.

The direction YAML files currently store 64 environments and 8 steps, following an earlier experiment. The explicit CLI flags below override them. Those settings would instead yield 512 samples per independent policy per update.

Simulator instances step sequentially by default. `--async-envs` creates separate simulator worker processes, not one controller process per robot. MuJoCo physics and low-level gait execution run on CPU; CUDA accelerates high-level policy learning.

### Training success versus evaluation actions

During training, categorical actions are sampled. TensorBoard records binary team success per completed episode. W&B `charts/success` is the mean over episodes completed in that rollout update; it is not a cumulative evaluation success rate or the fraction of robots at the goal.

The evaluation CLI defaults to deterministic actions. **Use `--stochastic` for the reported experiments.** The five-robot checkpoint achieved 100% success with sampling but only 2% under deterministic evaluation in the separate 100-episode tests. Deterministic action selection repeatedly produced blocked movement in those tests. Sampling uses fixed policy weights and does not perform learning.

## Installation and training

Run commands from this project directory, currently `/home/zj90/CS512_final`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,tracking]'
wandb login
pytest -q
```

Use a CUDA-compatible PyTorch installation for GPU training. Without `--track`, W&B is not contacted. `--track --wandb-mode offline` stores tracking data locally.

### Reproduce the evaluated 2D training setups

These commands start fresh runs. Choose new run directories if the named runs already exist: reusing a directory can overwrite its checkpoint and metrics.

```bash
CUDA_VISIBLE_DEVICES=0 python -m distributed_swarm.training.train_2d \
  --config distributed_swarm/configs/2d.yaml --training-mode shared \
  --num-envs 8 --num-steps 128 --total-timesteps 10000000 \
  --device cuda:0 --track --wandb-project-name CS512-swarm \
  --seed 1 --run-dir runs/reproduce_2d_5robots_10m_seed1

CUDA_VISIBLE_DEVICES=0 python -m distributed_swarm.training.train_2d \
  --config distributed_swarm/configs/2d_10robots.yaml --training-mode independent \
  --num-envs 16 --num-steps 128 --total-timesteps 20000000 --async-envs \
  --device cuda:0 --track --wandb-project-name CS512-swarm \
  --seed 1 --run-dir runs/reproduce_2d_10robots_20m_seed1

CUDA_VISIBLE_DEVICES=0 python -m distributed_swarm.training.train_2d \
  --config distributed_swarm/configs/2d_20robots.yaml --training-mode independent \
  --num-envs 16 --num-steps 128 --total-timesteps 50000000 --async-envs \
  --device cuda:0 --track --wandb-project-name CS512-swarm \
  --seed 1 --run-dir runs/reproduce_2d_20robots_50m_seed1
```

These reproduce the main hyperparameters, not a bitwise replay of a previous process. Saved versions, resolved configs, and any interrupted/resumed history matter for exact replication.

### Train hierarchical 3D without rendering

Download the low-level weights once:

```bash
python -m distributed_swarm.scripts.download_ant_policy
```

For 5 robots / 20M transitions and 10 robots / 50M transitions on GPU 0, and 20 robots / 100M transitions on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=0 python -m distributed_swarm.training.train_3d \
  --config distributed_swarm/configs/3d_direction.yaml \
  --training-mode independent --num-envs 4 --num-steps 256 \
  --total-timesteps 20000000 --device cuda:0 \
  --track --wandb-project-name CS512-swarm \
  --wandb-run-name ant-5robots-20m-4envs-seed1 \
  --seed 1 --run-dir runs/ant_5robots_20m_4envs_seed1

CUDA_VISIBLE_DEVICES=0 python -m distributed_swarm.training.train_3d \
  --config distributed_swarm/configs/3d_direction_10robots.yaml \
  --training-mode independent --num-envs 4 --num-steps 256 \
  --total-timesteps 50000000 --device cuda:0 \
  --track --wandb-project-name CS512-swarm \
  --wandb-run-name ant-10robots-50m-4envs-seed1 \
  --seed 1 --run-dir runs/ant_10robots_50m_4envs_seed1

CUDA_VISIBLE_DEVICES=1 python -m distributed_swarm.training.train_3d \
  --config distributed_swarm/configs/3d_direction_20robots.yaml \
  --training-mode independent --num-envs 4 --num-steps 256 \
  --total-timesteps 100000000 --device cuda:0 \
  --track --wandb-project-name CS512-swarm \
  --wandb-run-name ant-20robots-100m-4envs-seed1 \
  --seed 1 --run-dir runs/ant_20robots_100m_4envs_seed1
```

Physical GPU 1 appears as `cuda:0` inside a process with `CUDA_VISIBLE_DEVICES=1`. Run commands in separate terminals for concurrency; pasting them together executes them sequentially.

These commands omit video capture and asynchronous simulator workers. Earlier 3D runs aborted during the first-video stage; the native cause was not established, and isolated rendering/video tests succeeded. The commands avoid that path rather than claiming to fix its underlying cause.

### Saved files and resume

Each training directory contains `checkpoint.pt`, `config.json`, `metrics.csv`, TensorBoard event files, and optionally W&B/video artifacts. The checkpoint contains policy weights, optimizer state, environment config, learning mode, dimensions, and completed transition count.

Add `--resume PATH/checkpoint.pt` to continue compatible policies and optimizers. The environment configuration and training mode must match. The requested timestep budget is the desired total, including already completed transitions. Simulator episodes restart from seeded resets; resume is not an exact mid-episode replay.

## Evaluation and experimental protocol

### Baseline evaluation

Evaluate all three saved checkpoints with stochastic actions and 100 episodes:

```bash
for robots in 5 10 20; do
  case "$robots" in
    5) run="runs/2d_video" ;;
    10) run="runs/swarm2d_10robots_20m_seed1" ;;
    20) run="runs/swarm2d_20robots_50m_seed1" ;;
  esac
  python -m distributed_swarm.evaluation.evaluate \
    --kind 2d --checkpoint "$run/checkpoint.pt" \
    --stochastic --episodes 100 --seed 10000 \
    --output "results/2d_${robots}robots_stochastic/evaluation.csv"
done
```

These commands overwrite the stochastic results at those paths if rerun. They do not modify policy weights or older deterministic result folders. Without `--config`, single-policy evaluation loads the environment config from the checkpoint.

Evaluation seeds are 10000–10099. The policy RNG is seeded per episode, as are simulator resets. Outputs include episode CSV and a descriptive `.summary.json`.

### Fixed-policy fault-tolerance suite

The reported robustness experiments keep the trained weights fixed and change environment communication/failure settings. Run:

```bash
for robots in 5 10 20; do
  case "$robots" in
    5) run="runs/2d_video" ;;
    10) run="runs/swarm2d_10robots_20m_seed1" ;;
    20) run="runs/swarm2d_20robots_50m_seed1" ;;
  esac
  python -m distributed_swarm.evaluation.experiments \
    --kind 2d --config "$run/config.json" \
    --checkpoint "$run/checkpoint.pt" --stochastic \
    --output-dir "results/2d_${robots}robots_stochastic/robustness"
done
```

For this entry point, pass the resolved run config explicitly: otherwise it uses the default environment preset, which may not match larger checkpoints.

The suite in [experiments.yaml](distributed_swarm/configs/experiments.yaml) creates 42 cases:

| Family | Conditions | Number |
| --- | --- | ---: |
| Baseline | Original environment settings | 1 |
| Loss/delay/mode grid | Loss {0, 0.1, 0.3} × max delay {0, 2, 5} × {naive, timestamp, confidence} | 27 |
| Ordering controls | Unfiltered; sequence only; sequence + logical clocks | 3 |
| Controller crashes | 0, 1, 2 crashes | 3 |
| Component ablations | Full, no filter, no clocks, no detector, no confidence, predictor | 6 |
| Temporary network faults | Robot outage; two-group partition | 2 |

Each case has 3 evaluation seed groups × 10 episodes = 30 episodes. The same checkpoint is reused across seed groups; these are **not three independently trained policies**. Seeds are 11000–11009, 12000–12009, and 13000–13009. Each team size therefore contributes 1,260 robustness episodes.

Fault schedule:

- Crashes affect IDs 0 through count−1 permanently from control tick 30.
- The outage blocks robot 0's incident communication links during [30, 60).
- The partition separates IDs 0 through floor(N/2)−1 from the remaining IDs during [30, 60).
- Temporary communication faults do not disable robot actions.
- Outage/partition recovery is only testable in episodes that reach the relevant time.

Pairing seeds controls some randomness, but divergent trajectories and packet-routing branches can consume RNG differently. It is not a guarantee of identical random events across conditions.

The runner writes resolved case YAMLs, `suite.json`, `episodes.csv`, and a summary incrementally. The pooled summary is not a substitute for comparing individual cases.

### Ablation meanings

| Case | Change from the same fixed policy's environment |
| --- | --- |
| Full | Baseline confidence mode, filtering, clocks, detector; prediction off |
| No filter | Accept stale state updates; retain other confidence-mode features |
| No clocks | Disable Lamport instrumentation; retain sequence filtering |
| No detector | Disable failure classification; known neighbor features remain available |
| No confidence | Switch to timestamp mode; retain age and filtering |
| Predictor | Enable bounded cached-velocity position prediction |

These are deployment-time mechanism changes, not separately retrained policies. Changes to features can create a distribution shift relative to training. The no-clocks control is expected to match full behavior because logical timestamps do not influence action features or filtering.

The runner also supports `--train` for separate training per condition and seed. That is a different protocol with substantially more compute and is not the source of the results below. Without either a checkpoint or `--train`, it runs an explicitly labeled random-action baseline.

## Metrics

| Metric | Definition and interpretation |
| --- | --- |
| Team success | Binary final success; report its mean as successful episodes / total episodes |
| Goal-reaching fraction | Fraction of active robots in the goal at the final step |
| Episode return | Sum of scalar mean-team rewards; distinct from task success |
| Episode steps | Steps until success, all-inactive termination, or timeout |
| Completion time | Control ticks, recorded only for successful episodes |
| Connectivity | Fraction of active robots reachable from the first active robot in the geometric communication graph |
| Collision rate | Total reported collision count / (episode steps × initial robot count) |
| Average message age | Mean simulation-time age of cached neighbor state |
| Estimation error | Mean distance between cache-based position estimate and true neighbor position |
| Stale messages received/rejected | Per-episode cache counters |
| Messages / bytes | Simulator packet counters and modeled header/payload bytes, not actual network traffic |
| Detection rate | Detected eligible observer–crash pairs / all eligible pairs |
| Detection latency | Mean first observed suspicion/unavailability delay after injected crash |
| False suspicion rate | Fraction of suspicious observer-neighbor ticks referring to physically active, uncrashed robots |
| Recovery latency | Observed recovery-event delay after an outage/partition ends |
| Crash / fall counts | Controller crashes and physical inactivity reported separately |

In 2D, collisions count blocked robot flags per step, not necessarily distinct contact events. In 3D, they count distinct inter-Ant contact pairs per control step. Collision rates are therefore not directly comparable across worlds.

The collision denominator includes initial robot slots even after crashes. A lower collision rate after failures does not by itself mean safer navigation. Connectivity is geometric and can remain high during an injected network partition; it is not packet-delivery connectivity, nor evidence of message relay.

Estimation error includes cached entries even if their features are marked unusable. A lower estimation error need not imply higher team success. Detection/recovery averages are conditional on events being observed; absent values are blank, not zero-latency events. False suspicion measures the composition of suspicious ticks, not the fraction of all ticks spent falsely suspecting.

## Measured 2D results

The following are the saved results inspected on November 11, 2026. The checkpoint counters were checked directly; directory names alone were not used to infer completed training.

### Evaluated checkpoints and training budgets

| Robots | Saved checkpoint | Verified transitions | Training mode | Environments × rollout steps |
| --- | --- | ---: | --- | --- |
| 5 | `runs/2d_video/checkpoint.pt` | 10,000,000 | Shared PPO | 8 × 128 |
| 10 | `runs/swarm2d_10robots_20m_seed1/checkpoint.pt` | 20,000,000 | Independent PPO | 16 × 128 |
| 20 | `runs/swarm2d_20robots_50m_seed1/checkpoint.pt` | 50,000,000 | Independent PPO | 16 × 128 |

For independent policies, the budgets correspond to approximately 2M transitions per learner for 10 robots and 2.5M for 20 robots before inactive-sample filtering. The shared five-robot learner consumes pooled experience.

### Baseline: 100 episodes per team size

| Metric | 5 robots | 10 robots | 20 robots |
| --- | ---: | ---: | ---: |
| Team success | 100% | 99% | 77% |
| Mean active robots at goal | 100% | 99.2% | 96.35% |
| Collision rate | 0.0780 | 0.2075 | 0.3896 |
| Mean episode length | 71.83 | 166.21 | 363.85 |
| Position estimation error | 0.5330 | 0.5068 | 0.6327 |

The 20-robot policy often gets almost every robot to the goal, but one or more robots fail to join in time. This explains why a 96.35% individual goal-reaching fraction can coexist with 77% all-team success. Larger-team episodes also have more collision burden. Episode lengths include timeouts and should not be described as successful-only completion times.

![Baseline comparison](results/2d_5_vs_10_vs_20_stochastic/plots/baseline_comparison.png)

### Fault robustness: 30 episodes per condition

| Team success | 5 robots | 10 robots | 20 robots |
| --- | ---: | ---: | ---: |
| Baseline | 100% | 100% | 80% |
| One controller crash | 100% | 0% | 16.7% |
| Two controller crashes | 76.7% | 0% | 0% |
| Temporary robot outage | 100% | 100% | 80% |
| Temporary partition | 100% | 96.7% | 76.7% |

The 30-episode baseline uses different seeds from the 100-episode evaluation, explaining 100% versus 99% for 10 robots and 80% versus 77% for 20 robots.

Communication interruptions are much less damaging than permanent crashes in these tests. Five-robot crash robustness is stronger than the larger teams' observed robustness. The results do not identify the cause of the larger teams' crash failures: physical obstruction, failure-related observation changes, unseen deployment conditions, and policy differences require targeted tests.

![Robustness and ablation comparison](results/2d_5_vs_10_vs_20_stochastic/plots/robustness_comparison.png)

### Component ablations: team success

| Mechanism configuration | 5 robots | 10 robots | 20 robots |
| --- | ---: | ---: | ---: |
| Full | 100% | 100% | 80% |
| No stale-state filter | 100% | 100% | 83.3% |
| No logical clocks | 100% | 100% | 80% |
| No failure detector | 100% | 90% | 56.7% |
| No confidence decay / timestamp mode | 96.7% | 96.7% | 90% |
| Position predictor enabled | 100% | 96.7% | 76.7% |

These ablations use ordinary baseline network settings, not an additional crash or partition. Detector removal has a larger observed penalty for the larger teams, but the full system does not outperform every ablation. In particular, timestamp mode scores above full confidence mode for 20 robots under baseline conditions. These descriptive numbers do not establish statistical significance or universal component benefits.

Logical clocks alone produce no difference, as expected from their instrumentation-only role.

### High communication stress: 30% loss and maximum delay 5

| Metric | 5 robots | 10 robots | 20 robots |
| --- | ---: | ---: | ---: |
| Naive team success | 96.7% | 70% | 13.3% |
| Confidence-mode team success | 100% | 100% | 63.3% |
| Naive estimation error | 1.1246 | 0.9669 | 1.0778 |
| Confidence-mode estimation error | 0.8722 | 0.7773 | 0.8299 |
| Relative estimation-error reduction | 22.4% | 19.6% | 23.0% |

Under this specific stressed condition, confidence mode improves estimation accuracy and observed task success compared with naive mode. This comparison changes filtering and age features as well as confidence decay; it does not isolate confidence weighting alone.

### Deterministic evaluation discrepancy

The five-robot checkpoint's earlier 100-episode deterministic evaluation had 2% success, 40% mean goal-reaching fraction, and collision rate 0.6544. Its stochastic evaluation had 100% success and collision rate 0.0780.

A separate paired 30-seed diagnostic also found a large gap (3.3% deterministic versus 100% stochastic success). The earlier deterministic plots are retained as action-selection diagnostics, not the headline stochastic-policy performance. The numerical sample sets differ between the 100-episode runs and the paired diagnostic.

### Artifacts and uncertainty

- [Baseline figure, PDF](results/2d_5_vs_10_vs_20_stochastic/plots/baseline_comparison.pdf)
- [Robustness figure, PDF](results/2d_5_vs_10_vs_20_stochastic/plots/robustness_comparison.pdf)
- [Per-condition summary CSV](results/2d_5_vs_10_vs_20_stochastic/plots/case_summary.csv)
- [5-robot raw robustness CSV](results/2d_5robots_stochastic/robustness/episodes.csv)
- [10-robot raw robustness CSV](results/2d_10robots_stochastic/robustness/episodes.csv)
- [20-robot raw robustness CSV](results/2d_20robots_stochastic/robustness/episodes.csv)

Combined comparison plots show mean ± one standard error across evaluation episodes. The generic `visualization.plot_results` command instead shows episode standard deviation. Neither measures variability between independent training runs. At a measured success rate of 0% or 100%, the empirical standard-error bar is zero; this is not proof that the true success probability is exactly zero or one.

Saved local results, pretrained weights, and videos may not be included in a source-only checkout. Relative artifact links require those local files.

## Videos and trajectory demonstrations

Record one stochastic 2D evaluation video headlessly:

```bash
SDL_VIDEODRIVER=dummy python -m distributed_swarm.evaluation.evaluate \
  --kind 2d --checkpoint runs/2d_video/checkpoint.pt \
  --stochastic --episodes 1 --seed 10000 \
  --video videos/2d_5robots_stochastic.mp4 \
  --output results/video_example.csv
```

The evaluator only records the first episode when `--episodes` exceeds one; use separate invocations/seeds for separate videos. Rendering runs at 20 FPS and shows goal, static obstacle, trajectories, neighbor links, IDs, and robot health colors. Link endpoints use simulator truth for display only.

Fifteen verified success-selected baseline demonstrations have been saved:

| Robots | Recorded seeds | Files |
| --- | --- | --- |
| 5 | 10000, 10001, 10002, 10003, 10004 | [5-robot demonstrations](videos/2d_successful_trajectories/5robots/) |
| 10 | 10000, 10001, 10002, 10003, 10004 | [10-robot demonstrations](videos/2d_successful_trajectories/10robots/) |
| 20 | 10001, 10002, 10003, 10005, 10006 | [20-robot demonstrations](videos/2d_successful_trajectories/20robots/) |

Each episode has MP4 video, PNG trajectory plot, and compressed NPZ positions/actions. [manifest.json](videos/2d_successful_trajectories/manifest.json) records seeds, checkpoints, saved step counts, and verified metrics. These deliberately selected successes illustrate behavior; they must not be used to estimate unbiased success rates.

Interactive viewing uses `--render human`. Standalone 3D recording can use `MUJOCO_GL=egl` with `--kind 3d`; headless CPU rendering can use OSMesa when its system libraries are installed. EGL GPU selection is separate from CUDA device visibility.

Training supports `--capture-video`, `--video-interval`, `--video-length`, and `--video-seed`. Those training-time videos currently use deterministic actions in a separate environment, even when the policy trains stochastically. They do not measure stochastic evaluation performance. On this machine, prefer rendering separately from 3D training because of the observed native aborts.

## Limitations and next experiments

1. **One checkpoint per team size.** Evaluation seed groups do not replace independent training seeds.
2. **Confounded scaling comparison.** Robot count changes alongside map size, goal size, communication radius, horizon, training budget, and shared versus independent learning. The plots are comparisons of complete setups, not a controlled causal estimate of team-size scaling.
3. **Crash failures remain unresolved.** Larger teams have severe failure rates after injected crashes. Record failed trajectories and compare physical obstruction, detector settings, and failure-aware training before attributing the cause.
4. **Ablation distribution shift.** Runtime feature changes test deployment robustness of fixed policies. Separately retrained ablations answer a different question.
5. **Simple known task layout.** The goal and default obstacle are fixed. Randomized maps, unseen layouts, and different initial regions are needed for broader generalization claims.
6. **Limited fault coverage.** The standard suite tests specific early crash IDs, two-way group partitions, and scheduled outages; it does not cover all crash locations, timings, counts, or adversarial messages.
7. **Simulation, not hardware.** The message queue does not measure real socket latency, CPU contention, clock synchronization error, or physical network throughput.
8. **Success excludes losses.** Report crash counts and survivor counts alongside team success; high surviving-team success does not imply all original robots survived.
9. **3D differs materially.** There are no fixed obstacles, locomotion has its own limitations, and falls alter the active team. Evaluate 3D independently before claiming transfer of 2D findings.
10. **No universal mechanism ranking.** Confidence mode helps under tested high loss/delay, but timestamp mode can do better under baseline conditions. Clocks alone are a negative control.

Recommended next experiments are multiple independent training seeds; matched shared/independent policies; randomized fault times and affected IDs; crash-aware training; failed-trajectory inspection; and confidence intervals across training runs. Report both success and communication/state-estimation metrics.

## Project layout and validation

```text
distributed_swarm/
  agents/          Local observations, robot message generation, frozen Ant controller
  configs/         Environment/training presets and experiment suite
  distributed/     Transport, clocks, state cache, consistency, failure detector
  envs/            2D physics, MuJoCo Ant world, common metrics/state interfaces
  training/        PPO, 2D/3D entry points, training-video capture
  evaluation/      Evaluation, fixed-policy suites, ablation definitions, metrics
  visualization/   2D/3D rendering and generic result plots
  scripts/         Launchers and pinned gait downloader
  tests/           Communication, environment, and training checks
pretrained/        Frozen Ant gait and source metadata
runs/              Learned high-level checkpoints, configs, losses, tracking
results/           Baseline/suite episode CSVs, summaries, figures
videos/            Rendered evaluation clips and selected successful trajectories
```

Run `pytest -q` for the test suite. Tests cover transport reproducibility, payload isolation, reordering/FIFO, stale rejection, detector transitions, partitions/crashes, observation isolation, Gymnasium interfaces, Ant dimensions, crash-body behavior, squashed-Gaussian likelihoods, truncation-aware GAE, and short train/save/evaluate loops.

Smoke tests validate implementation paths, not convergence. The performance claims in this README come from the saved evaluation files identified above.
# CS512_final
