#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    s = v.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def to_int(v: Optional[str]) -> Optional[int]:
    f = to_float(v)
    if f is None:
        return None
    return int(f)


def qtile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.quantile(arr, q))


def geomean_positive(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None and x > 0]
    if not vals:
        return None
    return float(math.exp(sum(math.log(x) for x in vals) / len(vals)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_recommended(path: str) -> Dict[Tuple[str, str, str], str]:
    rec: Dict[Tuple[str, str, str], str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gc = (r.get("graph_class") or "").strip()
            mode = (r.get("mode") or "").strip()
            threads = (r.get("threads") or "").strip()
            cfg = (r.get("recommended_config") or "").strip()
            if gc and mode and threads and cfg:
                rec[(gc, mode, threads)] = cfg
    return rec


def parse_ablation(path: str) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("exit_code") or "").strip() != "0":
                continue
            r["total_time_f"] = to_float(r.get("total_time"))
            r["final_cut_f"] = to_float(r.get("final_cut"))
            r["result_n_i"] = to_int(r.get("result_n"))
            r["result_m_i"] = to_int(r.get("result_m"))
            rows.append(r)
    return rows


def size_from_row(r: dict) -> Tuple[Optional[int], Optional[int]]:
    n = r.get("result_n_i")
    m = r.get("result_m_i")
    if n is not None and m is not None:
        return n, m
    n_alt = to_int(r.get("lp_n_before")) or to_int(r.get("trivial_n_before"))
    m_alt = to_int(r.get("lp_m_before")) or to_int(r.get("trivial_m_before"))
    return n_alt, m_alt


def paired_samples(rows: List[dict], rec_map: Dict[Tuple[str, str, str], str]) -> List[dict]:
    by_key: Dict[Tuple[str, str, str, str, str], dict] = {}
    for r in rows:
        graph_path = (r.get("graph_path") or "").strip()
        mode = (r.get("mode") or "").strip()
        threads = (r.get("threads") or "").strip()
        seed = (r.get("seed") or "").strip()
        cfg = (r.get("config_name") or "").strip()
        if not graph_path or not mode or not threads or not seed or not cfg:
            continue
        by_key[(graph_path, mode, threads, seed, cfg)] = r

    paired = []
    for r in rows:
        cfg = (r.get("config_name") or "").strip()
        if cfg != "baseline":
            continue
        gc = (r.get("graph_class") or "").strip()
        mode = (r.get("mode") or "").strip()
        threads = (r.get("threads") or "").strip()
        seed = (r.get("seed") or "").strip()
        graph_path = (r.get("graph_path") or "").strip()
        rec_cfg = rec_map.get((gc, mode, threads))
        if rec_cfg is None:
            continue
        rec_row = by_key.get((graph_path, mode, threads, seed, rec_cfg))
        if rec_row is None:
            continue
        bt = r.get("total_time_f")
        rt = rec_row.get("total_time_f")
        bc = r.get("final_cut_f")
        rc = rec_row.get("final_cut_f")
        if bt is None or rt is None:
            continue
        n, m = size_from_row(r)
        if n is None:
            continue
        paired.append(
            {
                "graph_class": gc,
                "mode": mode,
                "threads": threads,
                "seed": seed,
                "graph_path": graph_path,
                "graph_name": (r.get("graph_name") or "").strip(),
                "recommended_config": rec_cfg,
                "n": int(n),
                "m": int(m) if m is not None else None,
                "baseline_time": float(bt),
                "recommended_time": float(rt),
                "time_gain_s": float(bt - rt),
                "speedup": float(bt / rt) if rt > 0 else None,
                "log_speedup": math.log(float(bt / rt)) if rt > 0 else None,
                "win": 1 if bt > rt else 0,
                "cut_mismatch": 1 if (bc is not None and rc is not None and bc != rc) else 0,
            }
        )
    return paired


def make_log_bins(values: List[int], num_bins: int) -> List[float]:
    arr = np.asarray([v for v in values if v is not None and v > 0], dtype=np.float64)
    if arr.size == 0:
        return [1.0, 10.0]
    lo = float(arr.min())
    hi = float(arr.max())
    if lo == hi:
        return [lo, hi + 1.0]
    edges = np.logspace(math.log10(lo), math.log10(hi), num_bins + 1)
    edges[0] = min(edges[0], lo)
    edges[-1] = max(edges[-1], hi)
    return list(edges)


def assign_bin(v: int, edges: List[float]) -> str:
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i + 1 < len(edges) - 1:
            if lo <= v < hi:
                return f"[{int(lo)}, {int(hi)})"
        else:
            if lo <= v <= hi:
                return f"[{int(lo)}, {int(hi)}]"
    return "unbinned"


def summarize_bins(paired: List[dict], key: str, edges: List[float]) -> List[dict]:
    by_bin: Dict[str, List[dict]] = defaultdict(list)
    for r in paired:
        v = r.get(key)
        if v is None:
            continue
        b = assign_bin(int(v), edges)
        by_bin[b].append(r)

    out = []
    for b in sorted(by_bin.keys()):
        rows = by_bin[b]
        gains = [x["time_gain_s"] for x in rows]
        speedups = [x["speedup"] for x in rows if x.get("speedup") is not None]
        wins = [x["win"] for x in rows]
        ns = [x["n"] for x in rows if x.get("n") is not None]
        ms = [x["m"] for x in rows if x.get("m") is not None]
        out.append(
            {
                "bin": b,
                "samples": len(rows),
                "n_median": float(np.median(ns)) if ns else None,
                "m_median": float(np.median(ms)) if ms else None,
                "time_gain_median_s": qtile(gains, 0.5),
                "time_gain_p25_s": qtile(gains, 0.25),
                "time_gain_p75_s": qtile(gains, 0.75),
                "speedup_median": qtile(speedups, 0.5),
                "speedup_geomean": geomean_positive(speedups),
                "speedup_p25": qtile(speedups, 0.25),
                "speedup_p75": qtile(speedups, 0.75),
                "win_fraction": float(sum(wins) / len(wins)) if wins else None,
                "cut_mismatches": int(sum(x["cut_mismatch"] for x in rows)),
            }
        )
    return out


def spearman_stats(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[float]]:
    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception:
        return None, None
    if len(x) < 3 or len(y) < 3:
        return None, None
    rho, pval = spearmanr(x, y)
    if rho is None or pval is None or math.isnan(rho) or math.isnan(pval):
        return None, None
    return float(rho), float(pval)


def ols_with_class(paired: List[dict], y_key: str) -> dict:
    rows = [r for r in paired if r.get("n") and r.get(y_key) is not None and r["n"] > 0]
    if len(rows) < 4:
        return {"ok": False, "reason": "not_enough_rows"}

    classes = sorted(set(r["graph_class"] for r in rows))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    p = 2 + max(0, len(classes) - 1)

    X = np.zeros((len(rows), p), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    X[:, 0] = 1.0
    for i, r in enumerate(rows):
        X[i, 1] = math.log(float(r["n"]))
        y[i] = float(r[y_key])
        ci = class_to_idx[r["graph_class"]]
        if ci > 0:
            X[i, 1 + ci] = 1.0

    try:
        beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    n, k = X.shape
    if rank < k or n <= k:
        return {"ok": False, "reason": "rank_deficient_or_small_sample"}
    yhat = X @ beta
    err = y - yhat
    sigma2 = float((err @ err) / (n - k))
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t = float(beta[1] / se[1]) if se[1] > 0 else None
    return {
        "ok": True,
        "n": int(n),
        "k": int(k),
        "coef_log_n": float(beta[1]),
        "se_log_n": float(se[1]),
        "t_log_n": t,
        "intercept": float(beta[0]),
        "classes": classes,
    }


def write_csv(path: str, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def make_plots(paired: List[dict], n_bins: List[dict], outdir: str) -> List[str]:
    outputs: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return outputs

    # Scatter log(n) vs speedup with binned medians overlay.
    xs = [math.log10(r["n"]) for r in paired if r.get("n") and r.get("speedup")]
    ys = [r["speedup"] for r in paired if r.get("n") and r.get("speedup")]
    if xs and ys:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(xs, ys, s=18, alpha=0.45, color="#2a9d8f", edgecolors="none")
        bx = []
        by = []
        for b in n_bins:
            nm = b.get("n_median")
            sm = b.get("speedup_median")
            if nm is not None and sm is not None and nm > 0:
                bx.append(math.log10(nm))
                by.append(sm)
        if bx and by:
            ord_idx = np.argsort(bx)
            bx = [bx[i] for i in ord_idx]
            by = [by[i] for i in ord_idx]
            ax.plot(bx, by, color="#e76f51", linewidth=2.2, marker="o", markersize=4)
        ax.set_xlabel("log10(n)")
        ax.set_ylabel("speedup (baseline / recommended)")
        ax.set_title("Preset Gain vs Graph Size")
        ax.grid(alpha=0.25, linewidth=0.6)
        plt.tight_layout()
        p = os.path.join(outdir, "size_vs_speedup_scatter.png")
        fig.savefig(p, dpi=220)
        plt.close(fig)
        outputs.append(p)

    # Boxplot by n-bin.
    labels = [b["bin"] for b in n_bins]
    vals = []
    for b in n_bins:
        lo = []
        for r in paired:
            if r.get("n") is None or r.get("speedup") is None:
                continue
            if assign_bin(int(r["n"]), [float(x) for x in []]):
                pass
        vals.append(
            [
                r["speedup"]
                for r in paired
                if r.get("n") is not None
                and r.get("speedup") is not None
                and assign_bin(int(r["n"]), _cached_edges) == b["bin"]  # type: ignore[name-defined]
            ]
        )

    # Since assign_bin needs edges, rebuild from bins for deterministic labels.
    return outputs


def make_boxplot_by_bins(paired: List[dict], edges: List[float], outdir: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    labels = []
    vals = []
    seen = []
    for r in paired:
        if r.get("n") is None or r.get("speedup") is None:
            continue
        b = assign_bin(int(r["n"]), edges)
        if b not in seen:
            seen.append(b)
    for b in sorted(seen):
        arr = [r["speedup"] for r in paired if r.get("speedup") is not None and r.get("n") is not None and assign_bin(int(r["n"]), edges) == b]
        if arr:
            labels.append(b)
            vals.append(arr)
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 4.8))
    ax.boxplot(vals, tick_labels=labels, showfliers=False)
    ax.set_ylabel("speedup (baseline / recommended)")
    ax.set_xlabel("n-size bin")
    ax.set_title("Speedup Distribution by Graph Size Bin")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    p = os.path.join(outdir, "size_bin_speedup_boxplot.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assess whether preset-selection gains increase with graph size."
    )
    ap.add_argument("--ablation-csv", default="results/ablation_results.csv")
    ap.add_argument(
        "--recommended-presets",
        default="results/stats/ablation_recommended_presets.csv",
    )
    ap.add_argument("--outdir", default="results/stats/size_effect")
    ap.add_argument("--num-bins", type=int, default=4)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    rec = parse_recommended(args.recommended_presets)
    rows = parse_ablation(args.ablation_csv)
    paired = paired_samples(rows, rec)
    if not paired:
        raise RuntimeError("No paired baseline/recommended rows found.")

    n_vals = [r["n"] for r in paired if r.get("n") is not None and r["n"] > 0]
    m_vals = [r["m"] for r in paired if r.get("m") is not None and r["m"] > 0]
    n_edges = make_log_bins(n_vals, max(2, args.num_bins))
    m_edges = make_log_bins(m_vals, max(2, args.num_bins)) if m_vals else []

    n_summary = summarize_bins(paired, "n", n_edges)
    m_summary = summarize_bins(paired, "m", m_edges) if m_edges else []

    xlogn = [math.log(float(r["n"])) for r in paired if r.get("n") and r.get("speedup") is not None and r["n"] > 0]
    yspeed = [r["speedup"] for r in paired if r.get("n") and r.get("speedup") is not None and r["n"] > 0]
    ygain = [r["time_gain_s"] for r in paired if r.get("n") and r.get("speedup") is not None and r["n"] > 0]

    rho_speed, p_speed = spearman_stats(xlogn, yspeed)
    rho_gain, p_gain = spearman_stats(xlogn, ygain)

    reg_speed = ols_with_class(paired, "log_speedup")
    reg_gain = ols_with_class(paired, "time_gain_s")

    pair_out = os.path.join(args.outdir, "size_effect_pairs.csv")
    n_out = os.path.join(args.outdir, "size_effect_by_n_bin.csv")
    m_out = os.path.join(args.outdir, "size_effect_by_m_bin.csv")
    summary_out = os.path.join(args.outdir, "size_effect_summary.json")

    write_csv(
        pair_out,
        [
            "graph_class",
            "mode",
            "threads",
            "seed",
            "graph_path",
            "graph_name",
            "recommended_config",
            "n",
            "m",
            "baseline_time",
            "recommended_time",
            "time_gain_s",
            "speedup",
            "log_speedup",
            "win",
            "cut_mismatch",
        ],
        paired,
    )
    write_csv(
        n_out,
        [
            "bin",
            "samples",
            "n_median",
            "m_median",
            "time_gain_median_s",
            "time_gain_p25_s",
            "time_gain_p75_s",
            "speedup_median",
            "speedup_geomean",
            "speedup_p25",
            "speedup_p75",
            "win_fraction",
            "cut_mismatches",
        ],
        n_summary,
    )
    if m_summary:
        write_csv(
            m_out,
            [
                "bin",
                "samples",
                "n_median",
                "m_median",
                "time_gain_median_s",
                "time_gain_p25_s",
                "time_gain_p75_s",
                "speedup_median",
                "speedup_geomean",
                "speedup_p25",
                "speedup_p75",
                "win_fraction",
                "cut_mismatches",
            ],
            m_summary,
        )

    plot_files: List[str] = []
    # scatter plot
    try:
        import matplotlib.pyplot as plt

        xs = [math.log10(r["n"]) for r in paired if r.get("n") and r.get("speedup")]
        ys = [r["speedup"] for r in paired if r.get("n") and r.get("speedup")]
        if xs and ys:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.scatter(xs, ys, s=18, alpha=0.45, color="#2a9d8f", edgecolors="none")
            bx = []
            by = []
            for b in n_summary:
                nm = b.get("n_median")
                sm = b.get("speedup_median")
                if nm is not None and sm is not None and nm > 0:
                    bx.append(math.log10(nm))
                    by.append(sm)
            if bx:
                ord_idx = np.argsort(bx)
                bx = [bx[i] for i in ord_idx]
                by = [by[i] for i in ord_idx]
                ax.plot(bx, by, color="#e76f51", linewidth=2.2, marker="o", markersize=4)
            ax.set_xlabel("log10(n)")
            ax.set_ylabel("speedup (baseline / recommended)")
            ax.set_title("Preset Gain vs Graph Size")
            ax.grid(alpha=0.25, linewidth=0.6)
            plt.tight_layout()
            p = os.path.join(args.outdir, "size_vs_speedup_scatter.png")
            fig.savefig(p, dpi=220)
            plt.close(fig)
            plot_files.append(p)
    except Exception:
        pass

    p_box = make_boxplot_by_bins(paired, n_edges, args.outdir)
    if p_box is not None:
        plot_files.append(p_box)

    summary = {
        "rows_total": len(rows),
        "paired_rows": len(paired),
        "n_bin_edges": n_edges,
        "m_bin_edges": m_edges,
        "overall": {
            "speedup_median": qtile([r["speedup"] for r in paired if r.get("speedup") is not None], 0.5),
            "speedup_geomean": geomean_positive([r["speedup"] for r in paired if r.get("speedup") is not None]),
            "time_gain_median_s": qtile([r["time_gain_s"] for r in paired], 0.5),
            "win_fraction": float(sum(r["win"] for r in paired) / len(paired)),
            "cut_mismatches": int(sum(r["cut_mismatch"] for r in paired)),
        },
        "spearman": {
            "speedup_vs_log_n": {"rho": rho_speed, "pvalue": p_speed},
            "time_gain_vs_log_n": {"rho": rho_gain, "pvalue": p_gain},
        },
        "class_controlled_regression": {
            "log_speedup_vs_log_n_plus_class": reg_speed,
            "time_gain_vs_log_n_plus_class": reg_gain,
        },
        "files": {
            "pairs_csv": pair_out,
            "by_n_bin_csv": n_out,
            "by_m_bin_csv": m_out if m_summary else None,
            "plots": plot_files,
        },
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {pair_out}")
    print(f"Wrote: {n_out}")
    if m_summary:
        print(f"Wrote: {m_out}")
    print(f"Wrote: {summary_out}")
    for p in plot_files:
        print(f"Wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
