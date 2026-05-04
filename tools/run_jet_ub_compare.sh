#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_jet_ub_compare.sh --input <graphs.csv> --output <results.csv> [options]

Required:
  --input <file>         CSV with columns: class,path
  --output <file>        Output CSV path

Options:
  --binary-dir <dir>     Build dir containing JET-enabled mincut binaries (default: ./build_jet)
  --mode <seq|par|both>  Run sequential, parallel, or both (default: seq)
  --threads <list>       Comma-separated thread counts for parallel mode (default: 1)
  --seeds <list>         Comma-separated seeds (default: 0)
  --jet-config <file>    Optional JET config file
  --jet-iter <n>         Number of JET upper-bound attempts (default: 1)
  --jobs <n>             Number of runs to execute in parallel via GNU parallel (default: 1)
  --dry-run              Print commands only, do not execute
  --help                 Show this help

Output:
  One CSV row per graph/seed/mode/thread/variant, where variant is either
  baseline or jet_ub. JET timing fields are populated only for the jet_ub row.
USAGE
}

trim() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "$s"
}

csv_escape() {
  local s="$1"
  s=${s//\"/\"\"}
  printf '"%s"' "$s"
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

split_csv_list() {
  local value="$1"
  local -n out_ref=$2
  IFS=',' read -r -a out_ref <<< "$value"
}

run_one() {
  local binary_dir="$1"
  local graph_class="$2"
  local graph_path="$3"
  local seed="$4"
  local run_mode="$5"
  local threads="$6"
  local variant="$7"
  local jet_config="$8"
  local jet_iter="$9"

  local seq_bin="$binary_dir/mincut"
  local par_bin="$binary_dir/mincut_parallel"
  local graph_name
  graph_name="$(basename "$graph_path")"

  local -a cmd
  if [[ "$run_mode" == "seq" ]]; then
    cmd=("$seq_bin" "-r" "$seed")
  else
    cmd=("$par_bin" "-r" "$seed" "-p" "$threads")
  fi

  if [[ "$variant" == "jet_ub" ]]; then
    cmd+=("--jet_ub" "--jet_iter" "$jet_iter")
    [[ -n "$jet_config" ]] && cmd+=("--jet_config" "$jet_config")
  fi

  if [[ "$run_mode" == "seq" ]]; then
    cmd+=("$graph_path" "vc")
  else
    cmd+=("$graph_path" "inexact")
  fi

  local cmd_str="${cmd[*]}"
  local tmp_log
  tmp_log="$(mktemp)"

  set +e
  "${cmd[@]}" >"$tmp_log" 2>&1
  local exit_code=$?
  set -e

  local parsed
  parsed="$(awk -v default_status="$([[ "$variant" == "jet_ub" ]] && echo missing || echo not_run)" '
    function parsekv(start,   i,a,pos,key,val) {
      delete kv
      for (i = start; i <= NF; ++i) {
        pos = index($i, "=")
        if (pos == 0) {
          continue
        }
        key = substr($i, 1, pos - 1)
        val = substr($i, pos + 1)
        kv[key] = val
      }
    }
    BEGIN {
      jet_status = default_status
      jet_cut = ""
      min_degree = ""
      used_cut = ""
      improved = ""
      jet_total = ""
      jet_conversion = ""
      jet_partition = ""
      result_time = ""
      result_cut = ""
      result_n = ""
      result_m = ""
    }
    /^JET_UB / {
      parsekv(2)
      jet_status = kv["status"]
      jet_cut = kv["jet_cut"]
      min_degree = kv["min_degree"]
      used_cut = kv["used_cut"]
      improved = kv["improved_min_degree"]
      jet_total = kv["total_time"]
      jet_conversion = kv["conversion_time"]
      jet_partition = kv["partition_time"]
      next
    }
    /^RESULT / {
      parsekv(2)
      result_time = kv["time"]
      result_cut = kv["cut"]
      result_n = kv["n"]
      result_m = kv["m"]
      next
    }
    END {
      printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n",
        result_time, result_cut, result_n, result_m,
        jet_status, jet_cut, min_degree, used_cut, improved,
        jet_total, jet_conversion, jet_partition
    }
  ' "$tmp_log")"

  rm -f "$tmp_log"

  IFS='|' read -r \
    total_time final_cut result_n result_m \
    jet_status jet_cut min_degree used_cut improved_min_degree \
    jet_total_time jet_conversion_time jet_partition_time \
    <<< "$parsed"

  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local -a row
  row=(
    "$timestamp" "$graph_class" "$graph_path" "$graph_name"
    "$run_mode" "$threads" "$seed" "$variant"
    "$exit_code" "$cmd_str"
    "$total_time" "$final_cut" "$result_n" "$result_m"
    "$jet_status" "$jet_cut" "$min_degree" "$used_cut" "$improved_min_degree"
    "$jet_total_time" "$jet_conversion_time" "$jet_partition_time"
  )

  local i
  for i in "${!row[@]}"; do
    row[$i]="$(csv_escape "${row[$i]}")"
  done

  join_by_comma "${row[@]}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_one "$@"
  exit 0
fi

INPUT=""
OUTPUT=""
BINARY_DIR="$(pwd)/build_jet"
MODE="seq"
THREADS_CSV="1"
SEEDS_CSV="0"
JET_CONFIG=""
JET_ITER="1"
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
    --mode)
      MODE="$2"; shift 2 ;;
    --threads)
      THREADS_CSV="$2"; shift 2 ;;
    --seeds)
      SEEDS_CSV="$2"; shift 2 ;;
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

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  usage
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

if [[ "$MODE" != "seq" && "$MODE" != "par" && "$MODE" != "both" ]]; then
  echo "Invalid --mode: $MODE" >&2
  exit 1
fi

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --jobs: $JOBS" >&2
  exit 1
fi

if ! [[ "$JET_ITER" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --jet-iter: $JET_ITER" >&2
  exit 1
fi

SEQ_BIN="$BINARY_DIR/mincut"
PAR_BIN="$BINARY_DIR/mincut_parallel"

if [[ "$MODE" == "seq" || "$MODE" == "both" ]]; then
  [[ -x "$SEQ_BIN" ]] || { echo "Missing binary: $SEQ_BIN" >&2; exit 1; }
fi
if [[ "$MODE" == "par" || "$MODE" == "both" ]]; then
  [[ -x "$PAR_BIN" ]] || { echo "Missing binary: $PAR_BIN" >&2; exit 1; }
fi

if [[ -n "$JET_CONFIG" && ! -f "$JET_CONFIG" ]]; then
  echo "JET config file not found: $JET_CONFIG" >&2
  exit 1
fi

if (( JOBS > 1 )); then
  command -v parallel >/dev/null 2>&1 || {
    echo "GNU parallel not found in PATH; install it or use --jobs 1" >&2
    exit 1
  }
fi

mkdir -p "$(dirname "$OUTPUT")"

THREADS=()
SEEDS=()
split_csv_list "$THREADS_CSV" THREADS
split_csv_list "$SEEDS_CSV" SEEDS
if [[ ${#THREADS[@]} -eq 0 ]]; then THREADS=("1"); fi
if [[ ${#SEEDS[@]} -eq 0 ]]; then SEEDS=("0"); fi

header=(
  timestamp graph_class graph_path graph_name mode threads seed variant
  exit_code command
  total_time final_cut result_n result_m
  jet_status jet_cut min_degree_cut used_initial_cut improved_min_degree
  jet_total_time jet_conversion_time jet_partition_time
)

tasks_file="$(mktemp)"
trap 'rm -f "$tasks_file"' EXIT

while IFS= read -r raw || [[ -n "$raw" ]]; do
  line="$(trim "$raw")"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  IFS=',' read -r graph_class graph_path_rest <<< "$line"
  graph_class="$(trim "$graph_class")"
  graph_path="$(trim "$graph_path_rest")"

  if [[ "$graph_class" == "class" && "$graph_path" == "path" ]]; then
    continue
  fi
  if [[ -z "$graph_class" || -z "$graph_path" ]]; then
    echo "Skipping malformed line: $line" >&2
    continue
  fi
  if [[ ! -f "$graph_path" ]]; then
    echo "Skipping missing graph: $graph_path" >&2
    continue
  fi

  for seed in "${SEEDS[@]}"; do
    seed="$(trim "$seed")"
    [[ -z "$seed" ]] && continue

    modes=()
    case "$MODE" in
      seq) modes=("seq") ;;
      par) modes=("par") ;;
      both) modes=("seq" "par") ;;
    esac

    for run_mode in "${modes[@]}"; do
      thread_values=("1")
      if [[ "$run_mode" == "par" ]]; then
        thread_values=("${THREADS[@]}")
      fi

      for threads in "${thread_values[@]}"; do
        threads="$(trim "$threads")"
        [[ -z "$threads" ]] && continue

        for variant in baseline jet_ub; do
          if [[ "$DRY_RUN" == "1" ]]; then
            cmd=()
            if [[ "$run_mode" == "seq" ]]; then
              cmd=("$SEQ_BIN" "-r" "$seed")
            else
              cmd=("$PAR_BIN" "-r" "$seed" "-p" "$threads")
            fi
            if [[ "$variant" == "jet_ub" ]]; then
              cmd+=("--jet_ub" "--jet_iter" "$JET_ITER")
              [[ -n "$JET_CONFIG" ]] && cmd+=("--jet_config" "$JET_CONFIG")
            fi
            if [[ "$run_mode" == "seq" ]]; then
              cmd+=("$graph_path" "vc")
            else
              cmd+=("$graph_path" "inexact")
            fi
            echo "DRY-RUN: ${cmd[*]}" >&2
          else
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "$BINARY_DIR" "$graph_class" "$graph_path" "$seed" "$run_mode" \
              "$threads" "$variant" "$JET_CONFIG" "$JET_ITER" >> "$tasks_file"
          fi
        done
      done
    done
  done
done < "$INPUT"

if [[ "$DRY_RUN" == "1" ]]; then
  join_by_comma "${header[@]}" > "$OUTPUT"
  echo "Wrote JET upper-bound comparison results to: $OUTPUT"
  exit 0
fi

{
  join_by_comma "${header[@]}"
  if (( JOBS == 1 )); then
    while IFS=$'\t' read -r binary_dir graph_class graph_path seed run_mode threads variant jet_config jet_iter; do
      "$0" --worker "$binary_dir" "$graph_class" "$graph_path" "$seed" \
        "$run_mode" "$threads" "$variant" "$jet_config" "$jet_iter"
    done < "$tasks_file"
  else
    parallel --jobs "$JOBS" --colsep '\t' --keep-order \
      "$0" --worker '{1}' '{2}' '{3}' '{4}' '{5}' '{6}' '{7}' '{8}' '{9}' \
      :::: "$tasks_file"
  fi
} > "$OUTPUT"

echo "Wrote JET upper-bound comparison results to: $OUTPUT"
