#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_noigen_viecut_parallel.sh [options]

Options:
  --input <file>        Graph manifest CSV (default: ./generated/noigen_suite/graphs.csv)
  --output <file>       Results CSV (default: ./results/noigen_parallel_jet_compare.csv)
  --binary-dir <dir>    Build dir containing mincut_parallel (default: ./build_jet_gpu)
  --threads <n>         Parallel VieCut thread count (default: 8)
  --seeds <list>        Comma-separated runtime seeds for VieCut (default: 0)
  --jet-config <file>   JET config file (default: extlib/Jet-Partitioner/configs/one_config.txt)
  --jet-iter <n>        Number of JET upper-bound attempts (default: 1)
  --jobs <n>            GNU parallel job count across instances (default: 1)
  --dry-run             Print underlying command only
  --help                Show this help

This is a thin wrapper around:
  tools/run_jet_ub_compare.sh

It runs VieCut in parallel mode on the NOIGEN manifest and compares:
  - baseline parallel VieCut
  - parallel VieCut with JET upper bound
USAGE
}

INPUT="$(pwd)/generated/noigen_suite/graphs.csv"
OUTPUT="$(pwd)/results/noigen_parallel_jet_compare.csv"
BINARY_DIR="$(pwd)/build_jet_gpu"
THREADS=8
SEEDS="0"
JET_CONFIG="extlib/Jet-Partitioner/configs/one_config.txt"
JET_ITER=1
JOBS=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"; shift 2 ;;
    --output)
      OUTPUT="$2"; shift 2 ;;
    --binary-dir)
      BINARY_DIR="$2"; shift 2 ;;
    --threads)
      THREADS="$2"; shift 2 ;;
    --seeds)
      SEEDS="$2"; shift 2 ;;
    --jet-config)
      JET_CONFIG="$2"; shift 2 ;;
    --jet-iter)
      JET_ITER="$2"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

cmd=(
  tools/run_jet_ub_compare.sh
  --input "$INPUT"
  --output "$OUTPUT"
  --binary-dir "$BINARY_DIR"
  --mode par
  --threads "$THREADS"
  --seeds "$SEEDS"
  --jet-config "$JET_CONFIG"
  --jet-iter "$JET_ITER"
  --jobs "$JOBS"
)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "${cmd[*]}"
else
  "${cmd[@]}"
fi
