#!/usr/bin/env bash
# ==============================================================================
# scripts/idle_watchdog.sh - Automatic Cloud VM Shutdown on Inactivity
# ==============================================================================
# Monitors CPU and GPU utilization, and whether the pipeline is running.
# If the machine remains continuously idle for IDLE_LIMIT_MINUTES (default: 20),
# it shuts down the instance to prevent runaway cloud charges.
# ==============================================================================
set -euo pipefail

IDLE_LIMIT_MINUTES="${1:-20}"
CHECK_INTERVAL_SECONDS=60
LOG_FILE="/var/log/idle_watchdog.log"

if ! touch "$LOG_FILE" 2>/dev/null; then
    LOG_FILE="idle_watchdog.log"
fi

echo "=== [$(date)] Starting Idle Watchdog (Timeout: ${IDLE_LIMIT_MINUTES} mins, Interval: ${CHECK_INTERVAL_SECONDS}s) ===" | tee -a "$LOG_FILE"

idle_minutes=0

while true; do
    sleep "$CHECK_INTERVAL_SECONDS"

    is_busy=0
    busy_reasons=()

    # 1. Check if run_v2.py or pytest is currently running
    if pgrep -f "run_v2" >/dev/null 2>&1; then
        is_busy=1
        busy_reasons+=("pipeline_running")
    fi

    # 2. Check GPU utilization if nvidia-smi is available
    gpu_util=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || echo "0")
        if [[ "$gpu_util" =~ ^[0-9]+$ ]] && [ "$gpu_util" -ge 5 ]; then
            is_busy=1
            busy_reasons+=("gpu_active_${gpu_util}%")
        fi
    fi

    # 3. Check CPU 1-minute load average against threshold (0.50)
    cpu_load_1m=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo "0.0")
    is_cpu_active=$(awk -v l="$cpu_load_1m" 'BEGIN {print (l >= 0.5) ? 1 : 0}')
    if [ "$is_cpu_active" -eq 1 ]; then
        is_busy=1
        busy_reasons+=("cpu_load_${cpu_load_1m}")
    fi

    if [ "$is_busy" -eq 1 ]; then
        if [ "$idle_minutes" -gt 0 ]; then
            echo "[$(date)] System active (${busy_reasons[*]}). Resetting idle counter from ${idle_minutes} to 0." | tee -a "$LOG_FILE"
        fi
        idle_minutes=0
    else
        idle_minutes=$((idle_minutes + 1))
        echo "[$(date)] System idle: ${idle_minutes}/${IDLE_LIMIT_MINUTES} mins (CPU load: ${cpu_load_1m}, GPU util: ${gpu_util}%)." | tee -a "$LOG_FILE"

        if [ "$idle_minutes" -ge "$IDLE_LIMIT_MINUTES" ]; then
            echo "!!! [$(date)] System idle for ${idle_minutes} consecutive minutes (>= limit of ${IDLE_LIMIT_MINUTES} mins)." | tee -a "$LOG_FILE"
            echo "!!! [$(date)] Executing poweroff to prevent unexpected cloud billing..." | tee -a "$LOG_FILE"
            sync
            sudo poweroff || sudo shutdown -h now
            exit 0
        fi
    fi
done
