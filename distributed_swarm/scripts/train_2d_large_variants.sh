#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

for robots in 10 20; do
  config="distributed_swarm/configs/2d_${robots}robots.yaml"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m distributed_swarm.training.train_2d \
    --config "$config" \
    --training-mode independent \
    --device cuda:0 \
    --async-envs \
    --track \
    --wandb-project-name "${WANDB_PROJECT:-CS512-swarm}" \
    --wandb-run-name "swarm2d-${robots}robots-10m-seed${SEED:-1}" \
    --capture-video \
    --video-interval 500000 \
    --video-length 300 \
    --seed "${SEED:-1}" \
    --run-dir "runs/swarm2d_${robots}robots_10m_seed${SEED:-1}"
done
