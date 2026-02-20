#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_ablation.sh --input <graphs.csv> --output <results.csv> [options]

Required:
  --input <file>         CSV with columns: class,path
  --output <file>        Output CSV path

Options:
  --binary-dir <dir>     Build dir containing mincut binaries (default: ./build)
  --mode <seq|par|both>  Run sequential, parallel, or both (default: both)
  --threads <list>       Comma-separated thread counts for parallel mode (default: 1)
  --seeds <list>         Comma-separated seeds (default: 0)
  --config-set <name>    baseline | loo | full (default: loo)
  --dry-run              Print commands only, do not execute
  --help                 Show this help

Config sets:
  baseline: all reductions enabled
  loo:      baseline + one-disabled-at-a-time (lp,trivial,pr1,pr2,pr3,pr4)
  full:     all 2^6 on/off combinations
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

build_configs() {
  local config_set="$1"
  local -n out_ref=$2
  out_ref=()

  case "$config_set" in
    baseline)
      out_ref+=("baseline:0:0:0:0:0:0")
      ;;
    loo)
      out_ref+=("baseline:0:0:0:0:0:0")
      out_ref+=("no_lp:1:0:0:0:0:0")
      out_ref+=("no_trivial:0:1:0:0:0:0")
      out_ref+=("no_pr1:0:0:1:0:0:0")
      out_ref+=("no_pr2:0:0:0:1:0:0")
      out_ref+=("no_pr3:0:0:0:0:1:0")
      out_ref+=("no_pr4:0:0:0:0:0:1")
      ;;
    full)
      local mask
      for mask in $(seq 0 63); do
        local dlp=$(( (mask >> 0) & 1 ))
        local dtr=$(( (mask >> 1) & 1 ))
        local d1=$(( (mask >> 2) & 1 ))
        local d2=$(( (mask >> 3) & 1 ))
        local d3=$(( (mask >> 4) & 1 ))
        local d4=$(( (mask >> 5) & 1 ))
        out_ref+=("m${mask}:${dlp}:${dtr}:${d1}:${d2}:${d3}:${d4}")
      done
      ;;
    *)
      echo "Unknown --config-set: $config_set" >&2
      exit 1
      ;;
  esac
}

INPUT=""
OUTPUT=""
BINARY_DIR="$(pwd)/build"
MODE="both"
THREADS_CSV="1"
SEEDS_CSV="0"
CONFIG_SET="loo"
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
    --config-set)
      CONFIG_SET="$2"; shift 2 ;;
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

SEQ_BIN="$BINARY_DIR/mincut"
PAR_BIN="$BINARY_DIR/mincut_parallel"

if [[ "$MODE" == "seq" || "$MODE" == "both" ]]; then
  [[ -x "$SEQ_BIN" ]] || { echo "Missing binary: $SEQ_BIN" >&2; exit 1; }
fi
if [[ "$MODE" == "par" || "$MODE" == "both" ]]; then
  [[ -x "$PAR_BIN" ]] || { echo "Missing binary: $PAR_BIN" >&2; exit 1; }
fi

mkdir -p "$(dirname "$OUTPUT")"

THREADS=()
SEEDS=()
CONFIGS=()
split_csv_list "$THREADS_CSV" THREADS
split_csv_list "$SEEDS_CSV" SEEDS
build_configs "$CONFIG_SET" CONFIGS

if [[ ${#THREADS[@]} -eq 0 ]]; then THREADS=("1"); fi
if [[ ${#SEEDS[@]} -eq 0 ]]; then SEEDS=("0"); fi

header=(
  timestamp graph_class graph_path graph_name mode threads seed config_name
  disable_lp disable_trivial disable_pr1 disable_pr2 disable_pr3 disable_pr4
  exit_code command
  total_time final_cut result_n result_m
  lp_time lp_n_before lp_m_before lp_n_after lp_m_after lp_cut_before lp_cut_after
  trivial_time trivial_n_before trivial_m_before trivial_n_after trivial_m_after trivial_cut_before trivial_cut_after
  pr12_time pr12_n_before pr12_m_before pr12_n_after pr12_m_after pr12_cut_before pr12_cut_after
  pr34_time pr34_n_before pr34_m_before pr34_n_after pr34_m_after pr34_cut_before pr34_cut_after
  noi_finalize_time noi_finalize_n_before noi_finalize_m_before noi_finalize_n_after noi_finalize_m_after noi_finalize_cut_before noi_finalize_cut_after
)

{
  join_by_comma "${header[@]}"

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

    graph_name="$(basename "$graph_path")"

    for seed in "${SEEDS[@]}"; do
      seed="$(trim "$seed")"
      [[ -z "$seed" ]] && continue

      for cfg in "${CONFIGS[@]}"; do
        IFS=':' read -r cfg_name dlp dtr d1 d2 d3 d4 <<< "$cfg"

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

            cmd=()
            if [[ "$run_mode" == "seq" ]]; then
              cmd=("$SEQ_BIN" "-v" "-r" "$seed")
            else
              cmd=("$PAR_BIN" "-v" "-r" "$seed" "-p" "$threads")
            fi

            [[ "$dlp" == "1" ]] && cmd+=("-E")
            [[ "$dtr" == "1" ]] && cmd+=("-F")
            [[ "$d1" == "1" ]] && cmd+=("-A")
            [[ "$d2" == "1" ]] && cmd+=("-B")
            [[ "$d3" == "1" ]] && cmd+=("-C")
            [[ "$d4" == "1" ]] && cmd+=("-D")

            if [[ "$run_mode" == "seq" ]]; then
              cmd+=("$graph_path" "vc")
            else
              cmd+=("$graph_path" "inexact")
            fi

            cmd_str="${cmd[*]}"

            if [[ "$DRY_RUN" == "1" ]]; then
              echo "DRY-RUN: $cmd_str" >&2
              continue
            fi

            tmp_log="$(mktemp)"
            exit_code=0
            if ! "${cmd[@]}" >"$tmp_log" 2>&1; then
              exit_code=$?
            fi

            parsed="$(awk '
              function parsekv(start,   i,a) {
                delete kv
                for (i = start; i <= NF; ++i) {
                  split($i, a, "=")
                  kv[a[1]] = a[2]
                }
              }
              /^STEP / {
                parsekv(2)
                s = kv["step"]
                t[s]  = kv["time"]
                nb[s] = kv["n_before"]
                mb[s] = kv["m_before"]
                na[s] = kv["n_after"]
                ma[s] = kv["m_after"]
                cb[s] = kv["cut_before"]
                ca[s] = kv["cut_after"]
                next
              }
              /^RESULT / {
                parsekv(2)
                rtime = kv["time"]
                rcut  = kv["cut"]
                rn    = kv["n"]
                rm    = kv["m"]
                next
              }
              END {
                steps = "lp trivial pr12 pr34 noi_finalize"
                printf "%s|%s|%s|%s", rtime, rcut, rn, rm
                n = split(steps, arr, " ")
                for (i = 1; i <= n; ++i) {
                  s = arr[i]
                  printf "|%s|%s|%s|%s|%s|%s|%s", t[s], nb[s], mb[s], na[s], ma[s], cb[s], ca[s]
                }
                printf "\n"
              }
            ' "$tmp_log")"

            IFS='|' read -r \
              total_time final_cut result_n result_m \
              lp_time lp_nb lp_mb lp_na lp_ma lp_cb lp_ca \
              tr_time tr_nb tr_mb tr_na tr_ma tr_cb tr_ca \
              pr12_time pr12_nb pr12_mb pr12_na pr12_ma pr12_cb pr12_ca \
              pr34_time pr34_nb pr34_mb pr34_na pr34_ma pr34_cb pr34_ca \
              noi_time noi_nb noi_mb noi_na noi_ma noi_cb noi_ca \
              <<< "$parsed"

            timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

            row=(
              "$timestamp" "$graph_class" "$graph_path" "$graph_name"
              "$run_mode" "$threads" "$seed" "$cfg_name"
              "$dlp" "$dtr" "$d1" "$d2" "$d3" "$d4"
              "$exit_code" "$cmd_str"
              "$total_time" "$final_cut" "$result_n" "$result_m"
              "$lp_time" "$lp_nb" "$lp_mb" "$lp_na" "$lp_ma" "$lp_cb" "$lp_ca"
              "$tr_time" "$tr_nb" "$tr_mb" "$tr_na" "$tr_ma" "$tr_cb" "$tr_ca"
              "$pr12_time" "$pr12_nb" "$pr12_mb" "$pr12_na" "$pr12_ma" "$pr12_cb" "$pr12_ca"
              "$pr34_time" "$pr34_nb" "$pr34_mb" "$pr34_na" "$pr34_ma" "$pr34_cb" "$pr34_ca"
              "$noi_time" "$noi_nb" "$noi_mb" "$noi_na" "$noi_ma" "$noi_cb" "$noi_ca"
            )

            for i in "${!row[@]}"; do
              row[$i]="$(csv_escape "${row[$i]}")"
            done

            join_by_comma "${row[@]}"
            rm -f "$tmp_log"
          done
        done
      done
    done
  done < "$INPUT"
} > "$OUTPUT"

echo "Wrote ablation results to: $OUTPUT"
