"""Independent mechanism ablations; clocks alone do not change state filtering."""
from copy import deepcopy


def variants(env_config):
    specs = {
        "full": {},
        "no_filter": {"filter_stale": False},
        "no_clocks": {"logical_clocks": False},
        "no_detector": {"failure_detection": False},
        "no_confidence": {"mode": "timestamp"},
        "predictor": {"predict_position": True},
    }
    for name, change in specs.items():
        config = deepcopy(env_config)
        config.setdefault("consistency", {}).update(change)
        yield name, config
