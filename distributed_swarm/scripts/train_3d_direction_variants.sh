#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m distributed_swarm.scripts.download_ant_policy

for robots in 5 10; do
  if [[ "$robots" == "5" ]]; then
    config="distributed_swarm/configs/3d_direction.yaml"
  else
    config="distributed_swarm/configs/3d_direction_10robots.yaml"
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  MUJOCO_GL="${MUJOCO_GL:-egl}" \
  MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}" \
  python -m distributed_swarm.training.train_3d \
    --config "$config" \
    --training-mode independent \
    --device cuda:0 \
    --async-envs \
    --track \
    --wandb-project-name "${WANDB_PROJECT:-CS512-swarm}" \
    --wandb-run-name "ant-${robots}robots-10m-seed${SEED:-1}" \
    --capture-video \
    --video-interval 500000 \
    --video-length 200 \
    --seed "${SEED:-1}" \
    --run-dir "runs/ant_${robots}robots_10m_seed${SEED:-1}"
done
