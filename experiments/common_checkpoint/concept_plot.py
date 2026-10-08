"""Show the measured shared-loss scale and setting response on one paired window."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


METHODS = {
    "sgd_low": ("SGD · smaller rate", "#0072B2", "o"),
    "momentum_low": ("HB · smaller rate", "#E69F00", "^"),
    "adam_low": ("Adam · calibrated", "#009E73", "D"),
    "sgd_high": ("SGD · larger rate", "#56B4E9", "s"),
    "momentum_high": ("HB · larger rate", "#D55E00", "v"),
    "head_low": ("Head only", "#CC79A7", "X"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root = args.root
    verification = json.loads((root / "verification.json").read_text())
    if not verification["verification_passed"] or verification["completed_runs"] != 18:
        raise ValueError("The concept plot requires all 18 checked intermediate runs.")
    sources = {}
    arrays = {}
    checked_sources = {entry["path"]: entry["sha256"] for entry in
                       json.loads((root / "figures/plot-manifest.json").read_text())["inputs"]}
    for method in METHODS:
        path = root / f"r0-{method}/trace.json"
        relative = str(path.relative_to(root))
        sources[relative] = sha(path)
        if checked_sources.get(relative) != sources[relative]:
            raise ValueError("The source trace changed after its checked figure export.")
        rows = json.loads(path.read_text())
        selected = [row for row in rows if 300 <= row["step"] <= 310]
        if [row["step"] for row in selected] != list(range(300, 311)):
            raise ValueError("The paired window must contain all 11 observations.")
        arrays[method] = {key: np.array([row[key] for row in selected])
                          for key in ["step", "loss", "difficulty"]}
    baseline = arrays["sgd_low"]["difficulty"]
    if not all(np.array_equal(value["difficulty"], baseline) for value in arrays.values()):
        raise ValueError("The six settings must share exactly the same frozen difficulty.")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10,
        "axes.labelsize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
        "axes.formatter.useoffset": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for method, (label, color, marker) in METHODS.items():
        data = arrays[method]
        for ax, y in zip(axes, [data["loss"], data["loss"] - baseline]):
            ax.plot(data["step"], y, color=color, marker=marker, markersize=4,
                    markerfacecolor="white", markeredgewidth=.9, linewidth=1.15, label=label)
    axes[0].plot(arrays["sgd_low"]["step"], baseline, color="#252525",
                 linewidth=1.35, marker="x", markersize=4, label="Shared frozen D")
    axes[0].set_title(r"(a) Shared fluctuation ($10^{-1}$)", loc="left", pad=10)
    axes[1].set_title(r"(b) Setting response ($10^{-3}$–$10^{-2}$)", loc="left", pad=10)
    axes[0].set_ylabel("Loss / difficulty (nats)")
    axes[1].set_ylabel(r"$L-D$ (nats)")
    axes[1].axhline(0, color="#777777", linewidth=.7, zorder=0)
    axes[1].set_ylim(-.035, .002)
    for ax in axes:
        ax.set_xlabel("Continuation step")
        ax.set_xlim(299.7, 310.3)
        ax.set_xticks(range(300, 311, 2))
        ax.set_xticks(range(300, 311), minor=True)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", alpha=.18, linewidth=.6)
        ax.tick_params(labelsize=9)
    handles = [Line2D([], [], color=color, marker=marker, markerfacecolor="white",
                      linewidth=1.15, markersize=4, label=label)
               for label, color, marker in METHODS.values()]
    handles.append(Line2D([], [], color="#252525", marker="x", linewidth=1.35,
                          markersize=4, label="Shared frozen D"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.02),
               ncol=3, frameon=False, columnspacing=1.6, handlelength=1.7)
    fig.subplots_adjust(left=.095, right=.98, bottom=.19, top=.70, wspace=.35)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ["pdf", "svg", "png"]:
        destination = args.output / ("common-loss-structure." + suffix)
        fig.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=.06)
        outputs.append({"path": destination.name, "sha256": sha(destination)})
    plt.close(fig)
    manifest = {
        "protocol": "intermediate adaptive protocol", "replica": 0,
        "window": [300, 310], "observations": 11, "aggregation": None,
        "sources": sources, "plan_sha256": sha(root / "plan.json"),
        "generator_sha256": sha(Path(__file__)),
        "verification_sha256": sha(root / "verification.json"),
        "difficulty_std": float(np.std(baseline)),
        "mean_responses": {method: float(np.mean(data["loss"] - baseline))
                           for method, data in arrays.items()},
        "figure_type": "measured data; not an idealized constant-offset schematic",
        "scope": "Paired batch losses and checkpoint-relative responses; no constant token-gap claim.",
        "outputs": outputs,
    }
    (args.output / "common-loss-structure-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"difficulty_std": manifest["difficulty_std"],
                      "mean_responses": manifest["mean_responses"], "figures": outputs}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("analysis/rgi_common_refinement_20261008"))
    parser.add_argument("--output", type=Path, default=Path("paper/rgi-followup/figures"))
    main(parser.parse_args())
