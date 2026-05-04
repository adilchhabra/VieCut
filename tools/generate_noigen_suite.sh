#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/generate_noigen_suite.sh [options]

Options:
  --binary-dir <dir>   Build dir containing the noigen binary (default: ./build)
  --output-dir <dir>   Directory to store generated graphs (default: ./generated/noigen_suite)
  --seeds <list>       Comma-separated seeds (default: 0,1,2)
  --k <value>          Decomposition number k for all instances (default: 2)
  --dry-run            Print commands only
  --help               Show this help

Output:
  - Weighted METIS graphs in <output-dir>
  - Manifest CSV at <output-dir>/graphs.csv with columns: class,path

Fixed instance set:
  (300,22425), (400,39900), (500,62375), (600,89850),
  (700,122325), (800,159800), (900,202275), (1000,249750)

Notes:
  - all instances use d = 50%.
  - p is left at the NOIGEN default, i.e. 1/n.
  - cluster output is not written.
USAGE
}

trim() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "$s"
}

split_csv_list() {
  local value="$1"
  local -n out_ref=$2
  IFS=',' read -r -a out_ref <<< "$value"
}

BINARY_DIR="$(pwd)/build"
OUTPUT_DIR="$(pwd)/generated/noigen_suite"
SEEDS_CSV="0,1,2"
K_VALUE=2
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary-dir)
      BINARY_DIR="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --seeds)
      SEEDS_CSV="$2"; shift 2 ;;
    --k)
      K_VALUE="$2"; shift 2 ;;
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

NOIGEN_BIN="$BINARY_DIR/noigen"
if [[ ! -x "$NOIGEN_BIN" ]]; then
  echo "Missing binary: $NOIGEN_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/graphs.csv"

SEEDS=()
split_csv_list "$SEEDS_CSV" SEEDS
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=("0" "1" "2")
fi

PAIRS=(
  "300:22425"
  "400:39900"
  "500:62375"
  "600:89850"
  "700:122325"
  "800:159800"
  "900:202275"
  "1000:249750"
)

DENSITY=50

echo "class,path" > "$MANIFEST"

for pair in "${PAIRS[@]}"; do
  IFS=':' read -r n m <<< "$pair"
  for seed in "${SEEDS[@]}"; do
    seed="$(trim "$seed")"
    [[ -z "$seed" ]] && continue
    out_graph="$OUTPUT_DIR/noigen_n${n}_m${m}_k${K_VALUE}_seed${seed}.graph"
    cmd=("$NOIGEN_BIN" "$n" "$out_graph" "-k" "$K_VALUE" "-d" "$DENSITY" "-s" "$seed")
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY-RUN: ${cmd[*]}"
    else
      "${cmd[@]}"
      echo "noigen,$out_graph" >> "$MANIFEST"
    fi
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo "Wrote NOIGEN graphs to: $OUTPUT_DIR"
  echo "Wrote manifest to: $MANIFEST"
fi
