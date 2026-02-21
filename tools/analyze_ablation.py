#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

STEPS = ["lp", "trivial", "pr12", "pr34", "noi_finalize"]


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


def to_int(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except ValueError:
        return None


def normalize_field(value):
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    value = value.replace('""', '"').strip()
    return value


def median(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.median(vals)


def mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.mean(vals)


def geomean(values):
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def quantile(values, q):
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    q = min(1.0, max(0.0, q))
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def robust_centered_bounds(values, center, lo_q=0.05, hi_q=0.95, min_span=0.05):
    lo = quantile(values, lo_q)
    hi = quantile(values, hi_q)
    if lo is None or hi is None:
        return center - 1.0, center + 1.0
    lo = min(lo, center - min_span)
    hi = max(hi, center + min_span)
    d = max(center - lo, hi - center)
    d = max(d, min_span)
    return center - d, center + d


def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, (int,)):
        return str(x)
    return f"{x:.6g}"


def matrix_bounds(matrix):
    vals = [
        v
        for row in matrix
        for v in row
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not vals:
        return 0.0, 1.0
    lo = min(vals)
    hi = max(vals)
    if lo == hi:
        return lo - 1.0, hi + 1.0
    return lo, hi


def format_annot_matrix(matrix, value_fmt):
    out = []
    for row in matrix:
        arow = []
        for v in row:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                arow.append("")
            else:
                arow.append(format(v, value_fmt))
        out.append(arow)
    return out


def parse_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r = {k: normalize_field(v) for k, v in r.items()}
            r["exit_code_int"] = to_int(r.get("exit_code"))
            r["total_time_f"] = to_float(r.get("total_time"))
            r["final_cut_f"] = to_float(r.get("final_cut"))
            for step in STEPS:
                r[f"{step}_time_f"] = to_float(r.get(f"{step}_time"))
                r[f"{step}_n_before_i"] = to_int(r.get(f"{step}_n_before"))
                r[f"{step}_m_before_i"] = to_int(r.get(f"{step}_m_before"))
                r[f"{step}_n_after_i"] = to_int(r.get(f"{step}_n_after"))
                r[f"{step}_m_after_i"] = to_int(r.get(f"{step}_m_after"))
            rows.append(r)
    return rows


def pre_noi_state(row):
    n = row.get("noi_finalize_n_before_i")
    m = row.get("noi_finalize_m_before_i")
    if n is not None and m is not None:
        return n, m
    for step in ["pr34", "pr12", "trivial", "lp"]:
        na = row.get(f"{step}_n_after_i")
        ma = row.get(f"{step}_m_after_i")
        if na is not None and ma is not None:
            return na, ma
    return None, None


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out_prefix, rule_effect_rows, per_rule_group):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
    except ImportError:
        print("matplotlib not available, skipping plot generation.")
        return []
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    outputs = []
    viridis = colormaps["viridis"]
    if sns is not None:
        sns.set_theme(style="white", context="paper", font_scale=1.2)
    by_mode_threads = defaultdict(list)
    for r in rule_effect_rows:
        by_mode_threads[(r["mode"], r["threads"])].append(r)

    if not by_mode_threads:
        print("No rule effect rows available, skipping plot generation.")
        return outputs

    target_mode_threads = max(by_mode_threads.items(), key=lambda x: len(x[1]))[0]
    mode, threads = target_mode_threads
    rows = by_mode_threads[target_mode_threads]
    classes = sorted({r["graph_class"] for r in rows})
    rules = ["lp", "trivial", "pr1", "pr2", "pr3", "pr4"]
    matrix = []
    for gclass in classes:
        row_vals = []
        for rule in rules:
            val = None
            for r in rows:
                if r["graph_class"] == gclass and r["rule_disabled"] == rule:
                    val = r["slowdown_median"]
                    break
            row_vals.append(float(val) if val is not None else float("nan"))
        matrix.append(row_vals)
    speedup_matrix = []
    for row in matrix:
        speedup_row = []
        for v in row:
            if isinstance(v, float) and math.isnan(v):
                speedup_row.append(float("nan"))
            elif v is None or v <= 0:
                speedup_row.append(float("nan"))
            else:
                speedup_row.append(1.0 / v)
        speedup_matrix.append(speedup_row)

    # Heatmap
    fig_h = max(4, 0.4 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    ann_m = format_annot_matrix(matrix, ".2f")
    if sns is not None:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=viridis,
            xticklabels=rules,
            yticklabels=classes,
            annot=ann_m,
            fmt="",
            annot_kws={
                "size": 8,
                "color": "#111111",
                "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.15},
            },
            linewidths=1.25,
            linecolor="#ffffff",
            cbar_kws={"label": "slowdown (disabled / baseline)", "shrink": 0.92},
        )
    else:
        im = ax.imshow(matrix, aspect="auto", cmap=viridis)
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels(rules)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("slowdown (disabled / baseline)")
    ax.set_title(f"Ablation Heatmap (mode={mode}, threads={threads})", pad=10)
    ax.set_xlabel("Disabled rule")
    ax.set_ylabel("Graph class")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    heatmap_path = Path(f"{out_prefix}_heatmap.png")
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(f"{out_prefix}_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    outputs.append(heatmap_path)

    # Gain heatmap with exact percent values.
    gain_matrix = []
    for row in matrix:
        gain_row = []
        for v in row:
            if math.isnan(v):
                gain_row.append(float("nan"))
            else:
                gain_row.append((1.0 - v) * 100.0)
        gain_matrix.append(gain_row)

    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    ann_g = format_annot_matrix(gain_matrix, ".1f")
    if sns is not None:
        sns.heatmap(
            gain_matrix,
            ax=ax,
            cmap=viridis,
            xticklabels=rules,
            yticklabels=classes,
            annot=ann_g,
            fmt="",
            annot_kws={
                "size": 8,
                "color": "#111111",
                "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.15},
            },
            linewidths=1.25,
            linecolor="#ffffff",
            cbar_kws={"label": "runtime gain % (positive means disabling helps)", "shrink": 0.92},
        )
    else:
        im = ax.imshow(gain_matrix, aspect="auto", cmap=viridis)
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels(rules)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("runtime gain % (positive means disabling helps)")
    ax.set_title(f"Exact Gain Heatmap (mode={mode}, threads={threads})", pad=10)
    ax.set_xlabel("Disabled rule")
    ax.set_ylabel("Graph class")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    gain_path = Path(f"{out_prefix}_gain_heatmap.png")
    fig.savefig(gain_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(f"{out_prefix}_gain_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    outputs.append(gain_path)

    # Speedup heatmap (inverse of slowdown).
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    ann_s = format_annot_matrix(speedup_matrix, ".2f")
    if sns is not None:
        sns.heatmap(
            speedup_matrix,
            ax=ax,
            cmap=viridis,
            xticklabels=rules,
            yticklabels=classes,
            annot=ann_s,
            fmt="",
            annot_kws={
                "size": 8,
                "color": "#111111",
                "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.15},
            },
            linewidths=1.25,
            linecolor="#ffffff",
            cbar_kws={"label": "speedup (baseline / disabled)", "shrink": 0.92},
        )
    else:
        im = ax.imshow(speedup_matrix, aspect="auto", cmap=viridis)
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels(rules)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("speedup (baseline / disabled)")
    ax.set_title(f"Speedup Heatmap (mode={mode}, threads={threads})", pad=10)
    ax.set_xlabel("Disabled rule")
    ax.set_ylabel("Graph class")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    speedup_path = Path(f"{out_prefix}_speedup_heatmap.png")
    fig.savefig(speedup_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(f"{out_prefix}_speedup_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    outputs.append(speedup_path)

    # Large grouped bar chart for per-class ablation comparison.
    fig_w = max(12, 1.1 * len(classes))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    colors = [viridis(x) for x in [i / max(1, (len(rules) - 1)) for i in range(len(rules))]]
    x = list(range(len(classes)))
    bar_w = 0.12
    for ridx, rule in enumerate(rules):
        vals = [matrix[cidx][ridx] for cidx in range(len(classes))]
        offset = (ridx - (len(rules) - 1) / 2.0) * bar_w
        xpos = [xi + offset for xi in x]
        ax.bar(xpos, vals, width=bar_w, label=rule, color=colors[ridx])
    ax.axhline(1.0, linestyle="--", linewidth=1, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("slowdown median (disabled / baseline)")
    ax.set_xlabel("graph class")
    ax.set_title(f"Ablation Comparison by Class (mode={mode}, threads={threads})")
    ax.legend(title="disabled rule", ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    class_plot_path = Path(f"{out_prefix}_class_comparison.png")
    fig.savefig(class_plot_path, dpi=150)
    plt.close(fig)
    outputs.append(class_plot_path)

    return outputs


def make_structural_plots(out_prefix, structural_rule_rows):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
    except ImportError:
        return []
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    outputs = []
    viridis = colormaps["viridis"]
    if sns is not None:
        sns.set_theme(style="white", context="paper", font_scale=1.2)

    by_mode_threads = defaultdict(list)
    for r in structural_rule_rows:
        by_mode_threads[(r["mode"], r["threads"])].append(r)
    if not by_mode_threads:
        return outputs

    target_mode_threads = max(by_mode_threads.items(), key=lambda x: len(x[1]))[0]
    mode, threads = target_mode_threads
    rows = by_mode_threads[target_mode_threads]
    classes = sorted({r["graph_class"] for r in rows})
    rules = ["lp", "trivial", "pr1", "pr2", "pr3", "pr4"]

    def matrix_for(field):
        matrix = []
        for gclass in classes:
            row_vals = []
            for rule in rules:
                val = None
                for r in rows:
                    if r["graph_class"] == gclass and r["rule_disabled"] == rule:
                        val = r[field]
                        break
                row_vals.append(float(val) if val is not None else float("nan"))
            matrix.append(row_vals)
        return matrix

    n_gain = matrix_for("n_gain_pct_median")
    m_gain = matrix_for("m_gain_pct_median")
    fig_h = max(4, 0.4 * len(classes) + 2)
    for label, matrix in [("n", n_gain), ("m", m_gain)]:
        fig, ax = plt.subplots(figsize=(8.4, fig_h))
        ann_sm = format_annot_matrix(matrix, ".1f")
        if sns is not None:
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=viridis,
                xticklabels=rules,
                yticklabels=classes,
                annot=ann_sm,
                fmt="",
                annot_kws={
                    "size": 8,
                    "color": "#111111",
                    "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 0.15},
                },
                linewidths=1.25,
                linecolor="#ffffff",
                cbar_kws={"label": f"{label}-reduction gain % (positive means smaller pre-NOI)", "shrink": 0.92},
            )
        else:
            im = ax.imshow(matrix, aspect="auto", cmap=viridis)
            ax.set_xticks(range(len(rules)))
            ax.set_xticklabels(rules)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(f"{label}-reduction gain % (positive means smaller pre-NOI)")
        ax.set_title(f"Pre-NOI {label.upper()} Gain Heatmap (mode={mode}, threads={threads})", pad=10)
        ax.set_xlabel("Disabled rule")
        ax.set_ylabel("Graph class")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        out = Path(f"{out_prefix}_{label}_gain_heatmap.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        fig.savefig(Path(f"{out_prefix}_{label}_gain_heatmap.pdf"), bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)

    return outputs


def main():
    ap = argparse.ArgumentParser(description="Analyze VieCut ablation CSV output")
    ap.add_argument("--input", required=True, help="CSV produced by tools/run_ablation.sh")
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="Prefix for output CSVs (default: <input_without_ext>)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_prefix = Path(args.out_prefix) if args.out_prefix else in_path.with_suffix("")

    rows = parse_rows(in_path)
    ok = [r for r in rows if r["exit_code_int"] == 0 and r["total_time_f"] is not None]

    print(f"Total rows: {len(rows)}")
    print(f"Successful rows: {len(ok)}")

    # Summary by class/mode/threads/config
    summary = []
    grouped = defaultdict(list)
    for r in ok:
        key = (r["graph_class"], r["mode"], r["threads"], r["config_name"])
        grouped[key].append(r)

    for (gclass, mode, threads, config), grp in sorted(grouped.items()):
        total_times = [x["total_time_f"] for x in grp]
        cuts = [x["final_cut_f"] for x in grp]
        summary.append(
            {
                "graph_class": gclass,
                "mode": mode,
                "threads": threads,
                "config_name": config,
                "runs": len(grp),
                "time_mean": mean(total_times),
                "time_median": median(total_times),
                "time_p95": statistics.quantiles(total_times, n=20)[18]
                if len(total_times) >= 20
                else None,
                "cut_mean": mean(cuts),
                "cut_median": median(cuts),
            }
        )

    print("\n== Runtime Summary (class/mode/threads/config) ==")
    for r in summary:
        print(
            f"class={r['graph_class']} mode={r['mode']} threads={r['threads']} "
            f"cfg={r['config_name']} runs={r['runs']} "
            f"time_med={fmt(r['time_median'])} time_mean={fmt(r['time_mean'])} "
            f"cut_med={fmt(r['cut_median'])}"
        )

    # Build baseline lookup for paired comparisons.
    # Key excludes config, includes graph+seed+mode+threads.
    baseline = {}
    for r in ok:
        if r["config_name"] != "baseline":
            continue
        key = (r["graph_path"], r["graph_class"], r["mode"], r["threads"], r["seed"])
        baseline[key] = r

    # Rule-level leave-one-out effects (assuming config names no_*).
    rule_configs = {
        "lp": "no_lp",
        "trivial": "no_trivial",
        "pr1": "no_pr1",
        "pr2": "no_pr2",
        "pr3": "no_pr3",
        "pr4": "no_pr4",
    }

    rule_effect_rows = []
    per_rule_group = defaultdict(list)
    for r in ok:
        for rule, cfg_name in rule_configs.items():
            if r["config_name"] != cfg_name:
                continue
            key = (r["graph_path"], r["graph_class"], r["mode"], r["threads"], r["seed"])
            b = baseline.get(key)
            if not b:
                continue
            if b["total_time_f"] in (None, 0.0):
                continue
            slowdown = r["total_time_f"] / b["total_time_f"]
            cut_delta = None
            if r["final_cut_f"] is not None and b["final_cut_f"] is not None:
                cut_delta = r["final_cut_f"] - b["final_cut_f"]
            gkey = (r["graph_class"], r["mode"], r["threads"], rule)
            per_rule_group[gkey].append((slowdown, cut_delta))

    for (gclass, mode, threads, rule), vals in sorted(per_rule_group.items()):
        slowdowns = [v[0] for v in vals]
        cut_deltas = [v[1] for v in vals if v[1] is not None]
        rule_effect_rows.append(
            {
                "graph_class": gclass,
                "mode": mode,
                "threads": threads,
                "rule_disabled": rule,
                "samples": len(vals),
                "slowdown_mean": mean(slowdowns),
                "slowdown_median": median(slowdowns),
                "slowdown_gt1_rate": sum(1 for s in slowdowns if s > 1.0) / len(slowdowns)
                if slowdowns
                else None,
                "cut_delta_mean": mean(cut_deltas),
                "cut_delta_median": median(cut_deltas),
            }
        )

    print("\n== Leave-One-Out Rule Effects vs Baseline ==")
    for r in rule_effect_rows:
        print(
            f"class={r['graph_class']} mode={r['mode']} threads={r['threads']} "
            f"rule={r['rule_disabled']} n={r['samples']} "
            f"slowdown_med={fmt(r['slowdown_median'])} "
            f"slowdown_mean={fmt(r['slowdown_mean'])} "
            f"P(slowdown>1)={fmt(r['slowdown_gt1_rate'])} "
            f"cut_delta_med={fmt(r['cut_delta_median'])}"
        )

    # Structural effects before NOI stage (paired vs baseline).
    structural_rule_rows = []
    per_rule_struct_group = defaultdict(list)
    per_cfg_struct_group = defaultdict(list)
    for r in ok:
        key = (r["graph_path"], r["graph_class"], r["mode"], r["threads"], r["seed"])
        b = baseline.get(key)
        if not b:
            continue
        rn, rm = pre_noi_state(r)
        bn, bm = pre_noi_state(b)
        if rn is None or rm is None or bn is None or bm is None:
            continue
        if bn <= 0 or bm < 0:
            continue
        n_ratio = rn / bn if bn > 0 else None
        m_ratio = rm / bm if bm > 0 else None
        struct_score = None
        parts = [x for x in (n_ratio, m_ratio) if x is not None and x > 0]
        if parts:
            struct_score = geomean(parts)
        cfg_key = (r["graph_class"], r["mode"], r["threads"], r["config_name"])
        per_cfg_struct_group[cfg_key].append((n_ratio, m_ratio, struct_score))

        for rule, cfg_name in rule_configs.items():
            if r["config_name"] != cfg_name:
                continue
            gkey = (r["graph_class"], r["mode"], r["threads"], rule)
            per_rule_struct_group[gkey].append((n_ratio, m_ratio))

    for (gclass, mode, threads, rule), vals in sorted(per_rule_struct_group.items()):
        n_ratios = [v[0] for v in vals if v[0] is not None]
        m_ratios = [v[1] for v in vals if v[1] is not None]
        n_med = median(n_ratios)
        m_med = median(m_ratios)
        structural_rule_rows.append(
            {
                "graph_class": gclass,
                "mode": mode,
                "threads": threads,
                "rule_disabled": rule,
                "samples": len(vals),
                "n_ratio_median": n_med,
                "m_ratio_median": m_med,
                "n_gain_pct_median": (1.0 - n_med) * 100.0 if n_med is not None else None,
                "m_gain_pct_median": (1.0 - m_med) * 100.0 if m_med is not None else None,
            }
        )

    print("\n== Structural Effects Before NOI (Leave-One-Out vs Baseline) ==")
    for r in structural_rule_rows:
        print(
            f"class={r['graph_class']} mode={r['mode']} threads={r['threads']} "
            f"rule={r['rule_disabled']} n={r['samples']} "
            f"n_gain%={fmt(r['n_gain_pct_median'])} "
            f"m_gain%={fmt(r['m_gain_pct_median'])}"
        )

    # Baseline stage-time shares.
    stage_rows = []
    per_stage_group = defaultdict(list)
    steps = STEPS
    for r in ok:
        if r["config_name"] != "baseline":
            continue
        total = r["total_time_f"]
        if total is None or total <= 0:
            continue
        for s in steps:
            st = r.get(f"{s}_time_f")
            if st is None:
                continue
            per_stage_group[(r["graph_class"], r["mode"], r["threads"], s)].append(st / total)

    for (gclass, mode, threads, step), shares in sorted(per_stage_group.items()):
        stage_rows.append(
            {
                "graph_class": gclass,
                "mode": mode,
                "threads": threads,
                "step": step,
                "samples": len(shares),
                "time_share_mean": mean(shares),
                "time_share_median": median(shares),
            }
        )

    print("\n== Baseline Stage Time Shares ==")
    for r in stage_rows:
        print(
            f"class={r['graph_class']} mode={r['mode']} threads={r['threads']} "
            f"step={r['step']} n={r['samples']} "
            f"share_med={fmt(r['time_share_median'])} share_mean={fmt(r['time_share_mean'])}"
        )

    # Reduction effects from n/m before/after (supports additional rule-quality analysis).
    reduction_rows = []
    reduction_group = defaultdict(list)
    for r in ok:
        for step in steps:
            nb = r.get(f"{step}_n_before_i")
            mb = r.get(f"{step}_m_before_i")
            na = r.get(f"{step}_n_after_i")
            ma = r.get(f"{step}_m_after_i")
            t = r.get(f"{step}_time_f")
            if nb is None or mb is None or na is None or ma is None:
                continue
            if nb <= 0 or mb < 0:
                continue
            node_removed = nb - na
            edge_removed = mb - ma
            node_red_frac = (node_removed / nb) if nb > 0 else None
            edge_red_frac = (edge_removed / mb) if mb > 0 else None
            nodes_per_sec = None
            edges_per_sec = None
            if t is not None and t > 0:
                nodes_per_sec = node_removed / t
                edges_per_sec = edge_removed / t
            gkey = (r["graph_class"], r["mode"], r["threads"], r["config_name"], step)
            reduction_group[gkey].append(
                (node_red_frac, edge_red_frac, nodes_per_sec, edges_per_sec)
            )

    for (gclass, mode, threads, cfg, step), vals in sorted(reduction_group.items()):
        node_fracs = [v[0] for v in vals if v[0] is not None]
        edge_fracs = [v[1] for v in vals if v[1] is not None]
        nps = [v[2] for v in vals if v[2] is not None]
        eps = [v[3] for v in vals if v[3] is not None]
        reduction_rows.append(
            {
                "graph_class": gclass,
                "mode": mode,
                "threads": threads,
                "config_name": cfg,
                "step": step,
                "samples": len(vals),
                "node_reduction_frac_median": median(node_fracs),
                "edge_reduction_frac_median": median(edge_fracs),
                "nodes_removed_per_sec_median": median(nps),
                "edges_removed_per_sec_median": median(eps),
            }
        )

    # Geometric-mean summary (time + paired slowdown vs baseline).
    geomean_rows = []
    per_cfg_paired_slowdowns = defaultdict(list)
    for r in ok:
        key = (r["graph_path"], r["graph_class"], r["mode"], r["threads"], r["seed"])
        b = baseline.get(key)
        if not b or b["total_time_f"] in (None, 0.0):
            continue
        if r["total_time_f"] is None or r["total_time_f"] <= 0:
            continue
        cfg_key = (r["graph_class"], r["mode"], r["threads"], r["config_name"])
        per_cfg_paired_slowdowns[cfg_key].append(r["total_time_f"] / b["total_time_f"])

    for s in summary:
        cfg_key = (s["graph_class"], s["mode"], s["threads"], s["config_name"])
        grp = grouped[cfg_key]
        times = [x["total_time_f"] for x in grp]
        slowdowns = per_cfg_paired_slowdowns.get(cfg_key, [])
        geomean_rows.append(
            {
                "graph_class": s["graph_class"],
                "mode": s["mode"],
                "threads": s["threads"],
                "config_name": s["config_name"],
                "runs": s["runs"],
                "time_geomean": geomean(times),
                "slowdown_geomean_vs_baseline": geomean(slowdowns),
            }
        )

    structural_summary_rows = []
    for s in summary:
        cfg_key = (s["graph_class"], s["mode"], s["threads"], s["config_name"])
        vals = per_cfg_struct_group.get(cfg_key, [])
        n_ratios = [v[0] for v in vals if v[0] is not None]
        m_ratios = [v[1] for v in vals if v[1] is not None]
        scores = [v[2] for v in vals if v[2] is not None]
        structural_summary_rows.append(
            {
                "graph_class": s["graph_class"],
                "mode": s["mode"],
                "threads": s["threads"],
                "config_name": s["config_name"],
                "samples": len(vals),
                "pre_noi_n_ratio_median": median(n_ratios),
                "pre_noi_m_ratio_median": median(m_ratios),
                "pre_noi_structural_score_median": median(scores),
                "pre_noi_n_gain_pct_median": (1.0 - median(n_ratios)) * 100.0
                if median(n_ratios) is not None
                else None,
                "pre_noi_m_gain_pct_median": (1.0 - median(m_ratios)) * 100.0
                if median(m_ratios) is not None
                else None,
            }
        )

    # Per-class recommended preset.
    # Choose fastest config by median time among configs with zero paired cut mismatches.
    paired_cut_stats = defaultdict(lambda: {"samples": 0, "mismatches": 0})
    for r in ok:
        key = (r["graph_path"], r["graph_class"], r["mode"], r["threads"], r["seed"])
        b = baseline.get(key)
        if not b:
            continue
        cfg_key = (r["graph_class"], r["mode"], r["threads"], r["config_name"])
        paired_cut_stats[cfg_key]["samples"] += 1
        if r["final_cut_f"] is None or b["final_cut_f"] is None:
            continue
        if r["final_cut_f"] != b["final_cut_f"]:
            paired_cut_stats[cfg_key]["mismatches"] += 1

    rec_rows = []
    rec_struct_rows = []
    rec_combined_rows = []
    summary_by_key = defaultdict(list)
    for s in summary:
        summary_by_key[(s["graph_class"], s["mode"], s["threads"])].append(s)

    for group_key, configs in sorted(summary_by_key.items()):
        gclass, mode, threads = group_key
        baseline_row = next((c for c in configs if c["config_name"] == "baseline"), None)
        candidates = []
        for c in configs:
            cfg_key = (gclass, mode, threads, c["config_name"])
            stats = paired_cut_stats.get(cfg_key, {"samples": 0, "mismatches": 0})
            if stats["mismatches"] == 0:
                candidates.append((c, stats))
        if not candidates:
            continue
        candidates.sort(
            key=lambda x: (
                x[0]["time_median"] if x[0]["time_median"] is not None else float("inf"),
                x[0]["time_mean"] if x[0]["time_mean"] is not None else float("inf"),
                x[0]["config_name"],
            )
        )
        best, best_stats = candidates[0]
        baseline_med = baseline_row["time_median"] if baseline_row else None
        best_med = best["time_median"]
        speedup_vs_baseline = None
        if (
            baseline_med is not None
            and best_med is not None
            and baseline_med > 0
            and best_med > 0
        ):
            speedup_vs_baseline = baseline_med / best_med

        rec_rows.append(
            {
                "graph_class": gclass,
                "mode": mode,
                "threads": threads,
                "recommended_config": best["config_name"],
                "recommended_time_median": best_med,
                "baseline_time_median": baseline_med,
                "speedup_vs_baseline": speedup_vs_baseline,
                "paired_samples": best_stats["samples"],
                "cut_mismatches_vs_baseline": best_stats["mismatches"],
            }
        )

        # Structural recommendation: smallest pre-NOI structural score.
        struct_candidates = []
        for c in configs:
            cfg_key = (gclass, mode, threads, c["config_name"])
            stats = paired_cut_stats.get(cfg_key, {"samples": 0, "mismatches": 0})
            if stats["mismatches"] != 0:
                continue
            vals = per_cfg_struct_group.get(cfg_key, [])
            scores = [v[2] for v in vals if v[2] is not None]
            nrs = [v[0] for v in vals if v[0] is not None]
            mrs = [v[1] for v in vals if v[1] is not None]
            if not scores:
                continue
            struct_candidates.append(
                (
                    c,
                    stats,
                    median(scores),
                    median(nrs),
                    median(mrs),
                )
            )
        if struct_candidates:
            struct_candidates.sort(
                key=lambda x: (
                    x[2] if x[2] is not None else float("inf"),
                    x[0]["time_median"] if x[0]["time_median"] is not None else float("inf"),
                    x[0]["config_name"],
                )
            )
            b_struct_vals = per_cfg_struct_group.get(
                (gclass, mode, threads, "baseline"), []
            )
            b_struct_scores = [v[2] for v in b_struct_vals if v[2] is not None]
            baseline_struct = median(b_struct_scores)
            sc_best = struct_candidates[0]
            rec_struct_rows.append(
                {
                    "graph_class": gclass,
                    "mode": mode,
                    "threads": threads,
                    "recommended_config": sc_best[0]["config_name"],
                    "recommended_structural_score_median": sc_best[2],
                    "recommended_pre_noi_n_ratio_median": sc_best[3],
                    "recommended_pre_noi_m_ratio_median": sc_best[4],
                    "baseline_structural_score_median": baseline_struct,
                    "structural_gain_pct_vs_baseline": (
                        (1.0 - (sc_best[2] / baseline_struct)) * 100.0
                        if baseline_struct is not None
                        and baseline_struct > 0
                        and sc_best[2] is not None
                        else None
                    ),
                    "paired_samples": sc_best[1]["samples"],
                    "cut_mismatches_vs_baseline": sc_best[1]["mismatches"],
                }
            )

        # Combined recommendation: minimize geometric mean of runtime slowdown and structural score.
        comb_candidates = []
        for c in configs:
            cfg_key = (gclass, mode, threads, c["config_name"])
            stats = paired_cut_stats.get(cfg_key, {"samples": 0, "mismatches": 0})
            if stats["mismatches"] != 0:
                continue
            if baseline_row is None or baseline_row["time_median"] in (None, 0):
                continue
            c_time = c["time_median"]
            if c_time is None or c_time <= 0:
                continue
            vals = per_cfg_struct_group.get(cfg_key, [])
            scores = [v[2] for v in vals if v[2] is not None]
            if not scores:
                continue
            time_slow = c_time / baseline_row["time_median"]
            struct_score = median(scores)
            if struct_score is None or struct_score <= 0:
                continue
            combined_score = math.sqrt(time_slow * struct_score)
            comb_candidates.append((c, stats, time_slow, struct_score, combined_score))
        if comb_candidates:
            comb_candidates.sort(key=lambda x: (x[4], x[0]["config_name"]))
            cbest = comb_candidates[0]
            rec_combined_rows.append(
                {
                    "graph_class": gclass,
                    "mode": mode,
                    "threads": threads,
                    "recommended_config": cbest[0]["config_name"],
                    "combined_score": cbest[4],
                    "time_slowdown_vs_baseline": cbest[2],
                    "structural_score_vs_baseline": cbest[3],
                    "paired_samples": cbest[1]["samples"],
                    "cut_mismatches_vs_baseline": cbest[1]["mismatches"],
                }
            )

    summary_path = Path(f"{out_prefix}_summary.csv")
    rule_path = Path(f"{out_prefix}_rule_effects.csv")
    stage_path = Path(f"{out_prefix}_stage_shares.csv")
    geomean_path = Path(f"{out_prefix}_geomean_summary.csv")
    rec_path = Path(f"{out_prefix}_recommended_presets.csv")
    rec_struct_path = Path(f"{out_prefix}_recommended_structural_presets.csv")
    rec_combined_path = Path(f"{out_prefix}_recommended_combined_presets.csv")
    reduction_path = Path(f"{out_prefix}_reduction_effects.csv")
    structural_summary_path = Path(f"{out_prefix}_structural_summary.csv")
    structural_rule_path = Path(f"{out_prefix}_structural_rule_effects.csv")

    write_csv(
        summary_path,
        [
            "graph_class",
            "mode",
            "threads",
            "config_name",
            "runs",
            "time_mean",
            "time_median",
            "time_p95",
            "cut_mean",
            "cut_median",
        ],
        summary,
    )

    write_csv(
        rule_path,
        [
            "graph_class",
            "mode",
            "threads",
            "rule_disabled",
            "samples",
            "slowdown_mean",
            "slowdown_median",
            "slowdown_gt1_rate",
            "cut_delta_mean",
            "cut_delta_median",
        ],
        rule_effect_rows,
    )

    write_csv(
        stage_path,
        [
            "graph_class",
            "mode",
            "threads",
            "step",
            "samples",
            "time_share_mean",
            "time_share_median",
        ],
        stage_rows,
    )

    write_csv(
        geomean_path,
        [
            "graph_class",
            "mode",
            "threads",
            "config_name",
            "runs",
            "time_geomean",
            "slowdown_geomean_vs_baseline",
        ],
        geomean_rows,
    )

    write_csv(
        rec_path,
        [
            "graph_class",
            "mode",
            "threads",
            "recommended_config",
            "recommended_time_median",
            "baseline_time_median",
            "speedup_vs_baseline",
            "paired_samples",
            "cut_mismatches_vs_baseline",
        ],
        rec_rows,
    )

    write_csv(
        rec_struct_path,
        [
            "graph_class",
            "mode",
            "threads",
            "recommended_config",
            "recommended_structural_score_median",
            "recommended_pre_noi_n_ratio_median",
            "recommended_pre_noi_m_ratio_median",
            "baseline_structural_score_median",
            "structural_gain_pct_vs_baseline",
            "paired_samples",
            "cut_mismatches_vs_baseline",
        ],
        rec_struct_rows,
    )

    write_csv(
        rec_combined_path,
        [
            "graph_class",
            "mode",
            "threads",
            "recommended_config",
            "combined_score",
            "time_slowdown_vs_baseline",
            "structural_score_vs_baseline",
            "paired_samples",
            "cut_mismatches_vs_baseline",
        ],
        rec_combined_rows,
    )

    write_csv(
        reduction_path,
        [
            "graph_class",
            "mode",
            "threads",
            "config_name",
            "step",
            "samples",
            "node_reduction_frac_median",
            "edge_reduction_frac_median",
            "nodes_removed_per_sec_median",
            "edges_removed_per_sec_median",
        ],
        reduction_rows,
    )

    write_csv(
        structural_summary_path,
        [
            "graph_class",
            "mode",
            "threads",
            "config_name",
            "samples",
            "pre_noi_n_ratio_median",
            "pre_noi_m_ratio_median",
            "pre_noi_structural_score_median",
            "pre_noi_n_gain_pct_median",
            "pre_noi_m_gain_pct_median",
        ],
        structural_summary_rows,
    )

    write_csv(
        structural_rule_path,
        [
            "graph_class",
            "mode",
            "threads",
            "rule_disabled",
            "samples",
            "n_ratio_median",
            "m_ratio_median",
            "n_gain_pct_median",
            "m_gain_pct_median",
        ],
        structural_rule_rows,
    )

    plot_paths = make_plots(out_prefix, rule_effect_rows, per_rule_group)
    plot_paths.extend(make_structural_plots(out_prefix, structural_rule_rows))

    print("\nWrote:")
    print(f"  {summary_path}")
    print(f"  {rule_path}")
    print(f"  {stage_path}")
    print(f"  {geomean_path}")
    print(f"  {rec_path}")
    print(f"  {rec_struct_path}")
    print(f"  {rec_combined_path}")
    print(f"  {reduction_path}")
    print(f"  {structural_summary_path}")
    print(f"  {structural_rule_path}")
    for p in plot_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
