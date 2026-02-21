#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import shlex
import subprocess
import sys


def parse_key_value_line(line):
    parsed = {}
    try:
        tokens = shlex.split(line)
    except ValueError:
        return parsed
    if not tokens:
        return parsed
    for tok in tokens[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        parsed[k] = v
    return parsed


def iso_utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def as_float(v):
    try:
        if v is None or v == "":
            return ""
        return float(v)
    except ValueError:
        return ""


def main():
    ap = argparse.ArgumentParser(
        description="Run graph class guessing over a class,path CSV and export features + guesses"
    )
    ap.add_argument("--input", required=True, help="CSV with columns: class,path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument(
        "--binary",
        default=os.path.join(os.getcwd(), "build", "graph_class_guesser"),
        help="Path to graph_class_guesser binary (default: ./build/graph_class_guesser)",
    )
    ap.add_argument("--verbose", action="store_true", help="Pass -v to guesser binary")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    ap.add_argument(
        "--timeout-sec",
        type=float,
        default=300.0,
        help="Per-graph timeout in seconds (default: 300)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.binary) or not os.access(args.binary, os.X_OK):
        print(f"Guesser binary not found or not executable: {args.binary}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fieldnames = [
        "timestamp",
        "graph_class_true",
        "graph_path",
        "graph_name",
        "exit_code",
        "guess_class",
        "guess_preset",
        "guess_confidence",
        "guess_correct",
        "guess_lp",
        "guess_trivial",
        "guess_pr1",
        "guess_pr2",
        "guess_pr3",
        "guess_pr4",
        "guess_flags",
        "guess_rationale",
        "feature_n",
        "feature_m",
        "feature_avg_degree",
        "feature_density",
        "feature_min_degree",
        "feature_degree_p50",
        "feature_degree_p90",
        "feature_degree_p99",
        "feature_degree_stddev",
        "feature_degree_cv",
        "feature_max_degree",
        "feature_leaf_fraction",
        "feature_isolated_fraction",
        "feature_io_time",
        "feature_time",
        "command",
        "stderr",
    ]

    with open(args.input, newline="", encoding="utf-8") as fin, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None or "class" not in reader.fieldnames or "path" not in reader.fieldnames:
            print("Input CSV must have columns: class,path", file=sys.stderr)
            return 1

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            gt_class = (row.get("class") or "").strip()
            graph_path = (row.get("path") or "").strip()
            if not gt_class or not graph_path:
                continue

            cmd = [args.binary, graph_path]
            if args.verbose:
                cmd.append("-v")
            cmd_str = " ".join(shlex.quote(x) for x in cmd)

            out_row = {k: "" for k in fieldnames}
            out_row["timestamp"] = iso_utc_now()
            out_row["graph_class_true"] = gt_class
            out_row["graph_path"] = graph_path
            out_row["graph_name"] = os.path.basename(graph_path)
            out_row["command"] = cmd_str

            if args.dry_run:
                print(f"DRY-RUN: {cmd_str}", file=sys.stderr)
                out_row["exit_code"] = "0"
                writer.writerow(out_row)
                continue

            if not os.path.isfile(graph_path):
                out_row["exit_code"] = "127"
                out_row["stderr"] = f"missing graph file: {graph_path}"
                writer.writerow(out_row)
                continue

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_sec,
                )
                exit_code = proc.returncode
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
            except subprocess.TimeoutExpired:
                out_row["exit_code"] = "124"
                out_row["stderr"] = f"timeout after {args.timeout_sec}s"
                writer.writerow(out_row)
                continue

            features = {}
            guess = {}
            for line in stdout.splitlines():
                if line.startswith("FEATURES "):
                    features = parse_key_value_line(line)
                elif line.startswith("GUESS "):
                    guess = parse_key_value_line(line)

            out_row["exit_code"] = str(exit_code)
            out_row["stderr"] = stderr.strip()

            out_row["guess_class"] = guess.get("class", "")
            out_row["guess_preset"] = guess.get("preset", "")
            out_row["guess_confidence"] = as_float(guess.get("confidence", ""))
            out_row["guess_lp"] = guess.get("lp", "")
            out_row["guess_trivial"] = guess.get("trivial", "")
            out_row["guess_pr1"] = guess.get("pr1", "")
            out_row["guess_pr2"] = guess.get("pr2", "")
            out_row["guess_pr3"] = guess.get("pr3", "")
            out_row["guess_pr4"] = guess.get("pr4", "")
            out_row["guess_flags"] = guess.get("flags", "")
            out_row["guess_rationale"] = guess.get("rationale", "")
            if out_row["guess_class"]:
                out_row["guess_correct"] = "1" if out_row["guess_class"] == gt_class else "0"

            out_row["feature_n"] = as_float(features.get("n", ""))
            out_row["feature_m"] = as_float(features.get("m", ""))
            out_row["feature_avg_degree"] = as_float(features.get("avg_degree", ""))
            out_row["feature_density"] = as_float(features.get("density", ""))
            out_row["feature_min_degree"] = as_float(features.get("min_degree", ""))
            out_row["feature_degree_p50"] = as_float(features.get("degree_p50", ""))
            out_row["feature_degree_p90"] = as_float(features.get("degree_p90", ""))
            out_row["feature_degree_p99"] = as_float(features.get("degree_p99", ""))
            out_row["feature_degree_stddev"] = as_float(features.get("degree_stddev", ""))
            out_row["feature_degree_cv"] = as_float(features.get("degree_cv", ""))
            out_row["feature_max_degree"] = as_float(features.get("max_degree", ""))
            out_row["feature_leaf_fraction"] = as_float(features.get("leaf_fraction", ""))
            out_row["feature_isolated_fraction"] = as_float(
                features.get("isolated_fraction", "")
            )
            out_row["feature_io_time"] = as_float(features.get("io_time", ""))
            out_row["feature_time"] = as_float(features.get("feature_time", ""))

            writer.writerow(out_row)

    print(f"Wrote graph class guessing results to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

