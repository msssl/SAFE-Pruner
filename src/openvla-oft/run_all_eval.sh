#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

LOG_DIR="${LOG_DIR:-experiments/logs_batch}"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
TASK_SUITES=(libero_object libero_spatial libero_goal libero_10)
USE_SAFE_PRUNER_VALUES=(True False)

total=$((${#TASK_SUITES[@]} * ${#USE_SAFE_PRUNER_VALUES[@]}))
idx=0
failed=0

printf 'Running %d evaluations. Logs: %s/%s_*.txt\n' "$total" "$LOG_DIR" "$RUN_ID"

for task_suite in "${TASK_SUITES[@]}"; do
  for use_safe_pruner in "${USE_SAFE_PRUNER_VALUES[@]}"; do
    idx=$((idx + 1))
    log_file="${LOG_DIR}/${RUN_ID}_${task_suite}_${use_safe_pruner}.txt"

    printf '[%d/%d] %s %s ... ' "$idx" "$total" "$task_suite" "$use_safe_pruner"

    if bash run_eval.sh "$task_suite" "$use_safe_pruner" 2>&1 | tee "$log_file"; then
      printf '[%d/%d] OK -> %s\n' "$idx" "$total" "$log_file"
    else
      printf '[%d/%d] FAIL -> %s\n' "$idx" "$total" "$log_file"
      failed=$((failed + 1))
    fi
  done
done

if [ "$failed" -eq 0 ]; then
  printf 'Done.\n'
else
  printf 'Done with %d failed run(s).\n' "$failed"
  exit 1
fi
