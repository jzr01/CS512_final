"""Render a 3D policy or random baseline through the common evaluation runner."""
from distributed_swarm.evaluation.evaluate import main

if __name__ == "__main__":
    main(default_kind="3d")
