#!/usr/bin/env python3
import argparse
import csv
import math
import os
from typing import List, Tuple


def load_rows(path: str) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                baseline_runtime = float(r["baseline_runtime"])
                jet_runtime = float(r["jet_runtime"])
                speedup = float(r["baseline_over_jet_speedup"])
            except Exception:
                continue
            rows.append(
                {
                    "graph_name": r.get("graph_name_baseline", ""),
                    "baseline_runtime": baseline_runtime,
                    "jet_runtime": jet_runtime,
                    "speedup": speedup,
                }
            )
    return rows


def make_speedup_ecdf(rows: List[dict], outpath: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    speedups = sorted(r["speedup"] for r in rows if r["speedup"] > 0)
    if not speedups:
        raise RuntimeError("No positive speedup rows available for ECDF plot.")
    ys = [(i + 1) / len(speedups) for i in range(len(speedups))]
    median_speedup = speedups[len(speedups) // 2]
    geomean_speedup = math.exp(sum(math.log(v) for v in speedups) / len(speedups))

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.step(speedups, ys, where="post", color="#1f77b4", linewidth=2.0)
    ax.axvline(1.0, linestyle="--", linewidth=1.2, color="#444444")
    ax.axvline(median_speedup, linestyle=":", linewidth=1.2, color="#2ca02c")
    ax.axvline(geomean_speedup, linestyle=":", linewidth=1.2, color="#9467bd")
    ax.annotate(
        f"median={median_speedup:.2f}x",
        xy=(median_speedup, 0.04),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#2ca02c",
        fontsize=9,
        rotation=90,
    )
    ax.annotate(
        f"gmean={geomean_speedup:.2f}x",
        xy=(geomean_speedup, 0.22),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#9467bd",
        fontsize=9,
        rotation=90,
    )
    ax.set_xlabel("Speedup (baseline / JET)")
    ax.set_ylabel("Fraction of matched instances")
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    plt.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_speedup_plot(rows: List[dict], outpath: str) -> None:
    import matplotlib.pyplot as plt

    pairs: List[Tuple[str, float]] = [
        (r["graph_name"], r["speedup"]) for r in rows if r["speedup"] > 0
    ]
    if not pairs:
        raise RuntimeError("No positive speedup rows available for speedup plot.")

    pairs.sort(key=lambda x: x[1], reverse=True)
    xs = list(range(1, len(pairs) + 1))
    ys = [p[1] for p in pairs]
    median_speedup = ys[len(ys) // 2]
    geomean_speedup = math.exp(sum(math.log(v) for v in ys) / len(ys))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(xs, ys, color="#d62728", linewidth=2.0)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="#444444")
    ax.axhline(median_speedup, linestyle=":", linewidth=1.2, color="#2ca02c")
    ax.axhline(geomean_speedup, linestyle=":", linewidth=1.2, color="#9467bd")
    ax.text(len(xs) * 0.98, median_speedup, f"median={median_speedup:.2f}x",
            ha="right", va="bottom", color="#2ca02c", fontsize=9)
    ax.text(len(xs) * 0.98, geomean_speedup, f"gmean={geomean_speedup:.2f}x",
            ha="right", va="bottom", color="#9467bd", fontsize=9)
    ax.set_xlabel("Matched instances, sorted by baseline/JET speedup")
    ax.set_ylabel("Speedup (baseline / JET)")
    ax.set_title("JET Speedup over Baseline")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    plt.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot JET vs baseline runtime comparisons.")
    ap.add_argument(
        "--input",
        default="results/jet_vs_all_reductions_overlap.csv",
        help="Matched comparison CSV.",
    )
    ap.add_argument(
        "--outdir",
        default="results/jet_plots",
        help="Directory for output plots.",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.input)
    ecdf_path = os.path.join(args.outdir, "jet_speedup_ecdf.png")
    speedup_path = os.path.join(args.outdir, "jet_speedup_sorted.png")
    make_speedup_ecdf(rows, ecdf_path)
    make_speedup_plot(rows, speedup_path)
    print(ecdf_path)
    print(speedup_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
