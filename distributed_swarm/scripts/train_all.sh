#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for kind in 2d 3d; do
  python -m distributed_swarm.training.train_${kind} --run-dir "runs/${kind}" "$@"
done
