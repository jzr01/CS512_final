import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--metric", default="success")
    parser.add_argument("--output", default="results/plot.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    with open(args.csv) as stream:
        for row in csv.DictReader(stream):
            if row.get(args.metric) not in (None, ""):
                groups[row.get("case", "evaluation")].append(float(row[args.metric]))
    if not groups:
        raise ValueError("no observations for requested metric")
    names = list(groups)
    means = [np.mean(groups[name]) for name in names]
    stds = [np.std(groups[name]) for name in names]
    fig, ax = plt.subplots(figsize=(max(7, len(names)*.35), 5))
    ax.bar(names, means, yerr=stds, capsize=3)
    ax.set_ylabel(args.metric + " (mean ± episode SD)")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
