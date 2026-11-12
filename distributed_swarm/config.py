from pathlib import Path
import yaml


def load_config(path=None, kind="2d"):
    path = Path(path) if path else Path(__file__).parent / "configs" / f"{kind}.yaml"
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "env" not in config:
        raise ValueError("config requires an env mapping")
    return config
