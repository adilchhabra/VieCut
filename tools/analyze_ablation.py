#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


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


def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, (int,)):
        return str(x)
    return f"{x:.6g}"


def parse_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r = {k: normalize_field(v) for k, v in r.items()}
            r["exit_code_int"] = to_int(r.get("exit_code"))
            r["total_time_f"] = to_float(r.get("total_time"))
            r["final_cut_f"] = to_float(r.get("final_cut"))
            for step in ["lp", "trivial", "pr12", "pr34", "noi_finalize"]:
                r[f"{step}_time_f"] = to_float(r.get(f"{step}_time"))
            rows.append(r)
    return rows


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    # Baseline stage-time shares.
    stage_rows = []
    per_stage_group = defaultdict(list)
    steps = ["lp", "trivial", "pr12", "pr34", "noi_finalize"]
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

    summary_path = Path(f"{out_prefix}_summary.csv")
    rule_path = Path(f"{out_prefix}_rule_effects.csv")
    stage_path = Path(f"{out_prefix}_stage_shares.csv")

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

    print("\nWrote:")
    print(f"  {summary_path}")
    print(f"  {rule_path}")
    print(f"  {stage_path}")


if __name__ == "__main__":
    main()
