#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for kind in 2d 3d; do
  python -m distributed_swarm.evaluation.evaluate --kind "$kind" --checkpoint "runs/${kind}/checkpoint.pt" --output "results/${kind}.csv" "$@"
done
