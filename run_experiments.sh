#!/usr/bin/env bash

N=${1:-10}  # default 10 runs if not provided
LOGFILE="rollout_log.txt"

echo "Starting $N runs at $(date)" > "$LOGFILE"

success_count=0
fail_count=0
interrupted=0

trap 'interrupted=1' INT

for ((i=1; i<=N; i++)); do
    if [ "$interrupted" -eq 1 ]; then
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        echo "[$timestamp] Run $i: INTERRUPTED" | tee -a "$LOGFILE"
        break
    fi

    echo "Run $i/$N..."

    uv run -m testing.experiments.pipe_insert.run.dp_data_collection_rollout_2_rot
    exit_code=$?

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    if [ "$interrupted" -eq 1 ]; then
        echo "[$timestamp] Run $i: INTERRUPTED" | tee -a "$LOGFILE"
        break
    fi

    # Bash exit codes are 0–255; -1 is reported as 255.
    if [ $exit_code -eq 255 ]; then
        timesteps=-1
        echo "[$timestamp] Run $i: FAIL timesteps=$timesteps" | tee -a "$LOGFILE"
        ((fail_count++))
    else
        timesteps=$exit_code
        echo "[$timestamp] Run $i: SUCCESS timesteps=$timesteps" | tee -a "$LOGFILE"
        ((success_count++))
    fi
done

echo "--------------------------------" | tee -a "$LOGFILE"
echo "Total Success: $success_count" | tee -a "$LOGFILE"
echo "Total Fail:    $fail_count" | tee -a "$LOGFILE"
echo "Finished at $(date)" | tee -a "$LOGFILE"
