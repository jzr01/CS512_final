"""Download pinned tensor weights; never execute the upstream training script."""
from pathlib import Path
import hashlib
import json
import urllib.request

REPO = "sdpkjc/Ant-v4-ppo_fix_continuous_action-seed2"
REVISION = "b88f77df64955077718a67c826f2dc290dfbe4e9"
FILENAME = "ppo_fix_continuous_action.cleanrl_model"


def main():
    target = Path(__file__).resolve().parents[2] / "pretrained" / FILENAME
    target.parent.mkdir(parents=True,exist_ok=True)
    url=f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"
    urllib.request.urlretrieve(url,target)
    metadata=dict(repo=REPO,revision=REVISION,filename=FILENAME,url=url,
                  sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  task="forward locomotion; not a pretrained direction-command policy")
    target.with_suffix(".json").write_text(json.dumps(metadata,indent=2))
    print(target)


if __name__ == "__main__":
    main()
