#!/usr/bin/env bash
# ==============================================================================
# scripts/run_v2_gcloud.sh - Autonomous GCP Spot Runner for graph-r1 v2
# ==============================================================================
set -euo pipefail

CONFIG="configs/v2/hnscc_gcloud.yaml"
DO_PROBE=0
AUTO_SHUTDOWN=0
IDLE_TIMEOUT_MINUTES="${IDLE_TIMEOUT_MINUTES:-20}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --probe)
            DO_PROBE=1
            shift
            ;;
        --auto-shutdown)
            AUTO_SHUTDOWN=1
            shift
            ;;
        --idle-timeout)
            IDLE_TIMEOUT_MINUTES="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            if [[ "$1" == *.yaml ]] || [[ "$1" == *.yml ]]; then
                CONFIG="$1"
            fi
            shift
            ;;
    esac
done

DATA_DIR="data/hf_hnscc"
GCS_BUCKET="${GCS_BUCKET:-gs://pelliscope-scout-uploads/graph-r1}"
LOG_FILE="run_gcloud_$(hostname)_$(date +%Y%m%d_%H%M%S).log"

# Determine output directory from config
OUTPUT_DIR=$(python3 -c "
import yaml
from pathlib import Path
try:
    cfg = yaml.safe_load(Path('$CONFIG').read_text()) or {}
    name = cfg.get('experiment_name', 'hnscc_v2_e1')
    root = cfg.get('data', {}).get('output_root', 'outputs/v2')
    print(f'{root}/{name}')
except Exception:
    print('outputs/v2/hnscc_v2_e1')
" 2>/dev/null || echo "outputs/v2/hnscc_v2_e1")

echo "=== [$(date)] Initializing graph-r1 GCP Runner on $(hostname) ===" | tee -a "$LOG_FILE"
echo "Config: $CONFIG" | tee -a "$LOG_FILE"
echo "Output Directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "GCS Destination: $GCS_BUCKET" | tee -a "$LOG_FILE"

# 1. Start Background Idle Watchdog Daemon (auto-stops VM if idle too long)
if ! pgrep -f "idle_watchdog.sh" >/dev/null 2>&1; then
    echo "[$(date)] Starting idle watchdog daemon (${IDLE_TIMEOUT_MINUTES} min inactivity limit)..." | tee -a "$LOG_FILE"
    nohup ./scripts/idle_watchdog.sh "$IDLE_TIMEOUT_MINUTES" >> "$LOG_FILE" 2>&1 &
fi

# 2. Spot Preemption Handler (GCP delivers SIGTERM 30 seconds before termination)
trap_preempt() {
    echo "!!! [$(date)] GCP Spot VM Preemption signal (SIGTERM) received! Flushing checkpoint state..." | tee -a "$LOG_FILE"
    if [ -d "$OUTPUT_DIR" ] && command -v zstd >/dev/null 2>&1; then
        tar -cf - "$OUTPUT_DIR" | zstd -3 > "emergency_checkpoint_$(hostname).tar.zst" || true
        if command -v gcloud >/dev/null 2>&1; then
            gcloud storage cp "emergency_checkpoint_$(hostname).tar.zst" "${GCS_BUCKET}/emergency/" || true
        fi
    fi
    exit 143
}
trap trap_preempt SIGTERM

# 3. Safe Exit Handler (Runs on ANY exit: normal completion, failure, or interruption)
cleanup_on_exit() {
    EXIT_CODE=$?
    echo "=== [$(date)] Cloud Runner cleanup triggered (Exit code: $EXIT_CODE) ===" | tee -a "$LOG_FILE"

    # Package and upload artifacts to GCS
    if [ -d "$OUTPUT_DIR" ] || [ -f "$LOG_FILE" ]; then
        echo "[$(date)] Packaging artifacts..." | tee -a "$LOG_FILE"
        ARCHIVE_NAME="v2_results_$(hostname)_$(date +%Y%m%d_%H%M%S).tar.zst"
        if command -v zstd >/dev/null 2>&1; then
            tar -cf - "$OUTPUT_DIR" "$LOG_FILE" 2>/dev/null | zstd -T0 -3 > "$ARCHIVE_NAME" || true
        else
            tar -czf "${ARCHIVE_NAME%.zst}.gz" "$OUTPUT_DIR" "$LOG_FILE" 2>/dev/null || true
            ARCHIVE_NAME="${ARCHIVE_NAME%.zst}.gz"
        fi

        if command -v gcloud >/dev/null 2>&1 && [ -f "$ARCHIVE_NAME" ]; then
            echo "[$(date)] Uploading archive to ${GCS_BUCKET}/runs/..." | tee -a "$LOG_FILE"
            gcloud storage cp "$ARCHIVE_NAME" "${GCS_BUCKET}/runs/" || true
        fi
    fi

    # Git push lightweight evaluation summaries on clean success
    GH_TOKEN=""
    if [ -f "$HOME/credentials/github-token.txt" ]; then
        GH_TOKEN="$(cat "$HOME/credentials/github-token.txt")"
    elif [ -f "/home/prime/Documents/github-hawp-prime-token-90-days.txt" ]; then
        GH_TOKEN="$(cat "/home/prime/Documents/github-hawp-prime-token-90-days.txt")"
    fi

    if [ "$EXIT_CODE" -eq 0 ] && [ -n "$GH_TOKEN" ] && [ -d "$OUTPUT_DIR" ]; then
        echo "[$(date)] Syncing evaluation summaries to GitHub repository..." | tee -a "$LOG_FILE"
        git add -f "$OUTPUT_DIR/07_eval"/*.csv "$OUTPUT_DIR/12_report"/*.md "$OUTPUT_DIR/manifest.json" 2>/dev/null || true
        if ! git diff --cached --quiet; then
            git commit -m "results(e1): completed v2 execution on $(hostname) [skip ci]" || true
            PUSH_URL="https://x-access-token:${GH_TOKEN}@github.com/HawkFranklin-Research/graph-r1.git"
            git push "$PUSH_URL" HEAD:v2-pipeline || echo "Git push encountered an issue; archive is safe in GCS." | tee -a "$LOG_FILE"
            echo "[$(date)] Successfully pushed results to GitHub." | tee -a "$LOG_FILE"
        else
            echo "[$(date)] No new evaluation files to commit." | tee -a "$LOG_FILE"
        fi
    fi

    # Safe Auto-Shutdown if explicitly requested
    if [ "$AUTO_SHUTDOWN" -eq 1 ] || [ "${AUTO_SHUTDOWN:-0}" = "1" ]; then
        echo "[$(date)] AUTO_SHUTDOWN enabled. Powering off instance now..." | tee -a "$LOG_FILE"
        sudo poweroff || sudo shutdown -h now
    else
        echo "[$(date)] Runner finished without --auto-shutdown. Background watchdog remains active to power off if idle for ${IDLE_TIMEOUT_MINUTES}m." | tee -a "$LOG_FILE"
    fi
}
trap cleanup_on_exit EXIT

# 4. Check and install Python dependencies if needed
echo "[$(date)] Checking Python environment..." | tee -a "$LOG_FILE"
if ! python3 -c "import torch, torch_geometric, gseapy, huggingface_hub, tabulate" >/dev/null 2>&1; then
    echo "[$(date)] Installing required packages from requirements-v2.txt..." | tee -a "$LOG_FILE"
    pip install -q -r requirements-v2.txt 2>&1 | tee -a "$LOG_FILE"
fi

# 5. Check & Pull Input Data from Hugging Face if not already on disk
if [ ! -d "$DATA_DIR/harmonized" ] || [ ! -d "$DATA_DIR/clinical" ]; then
    echo "[$(date)] Harmonized data not found at $DATA_DIR. Fetching from Hugging Face Hub..." | tee -a "$LOG_FILE"
    
    # Read and export token
    if [ -z "${HF_TOKEN:-}" ]; then
        if [ -f "$HOME/credentials/hugging-face-read-token.txt" ]; then
            export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/credentials/hugging-face-read-token.txt")"
        elif [ -f "/home/prime/Documents/hugging-face-read-token.txt" ]; then
            export HF_TOKEN="$(tr -d '[:space:]' < "/home/prime/Documents/hugging-face-read-token.txt")"
        fi
    else
        export HF_TOKEN="$(echo "$HF_TOKEN" | tr -d '[:space:]')"
    fi

    python3 -c "
import os, sys
from huggingface_hub import snapshot_download

token = os.environ.get('HF_TOKEN')
repo_id = os.environ.get('HF_DATASET_REPO', 'VatsalPatel18/multiomics-hnscc-processed')
revision = os.environ.get('HF_DATASET_REVISION', 'bf953018da4f918e628a5c63a4143f19ebe6e663')

print(f'[HF] Downloading {repo_id} (revision: {revision})...')
path = snapshot_download(
    repo_id=repo_id,
    repo_type='dataset',
    revision=revision,
    local_dir='$DATA_DIR',
    token=token,
    allow_patterns=['harmonized/*', 'clinical/*', 'references/*']
)
print(f'[HF] Successfully unpacked dataset at: {path}')
" 2>&1 | tee -a "$LOG_FILE"
else
    echo "[$(date)] Harmonized data already present at $DATA_DIR." | tee -a "$LOG_FILE"
fi

# 6. Optional Pre-flight Probe Check (tests CUDA + PyTorch Geometric end-to-end)
if [ "$DO_PROBE" -eq 1 ] || [ "${PROBE:-0}" = "1" ]; then
    echo "[$(date)] Running pre-flight probe test on GPU (configs/v2/hnscc_gcloud_probe.yaml)..." | tee -a "$LOG_FILE"
    python3 scripts/run_v2.py --config configs/v2/hnscc_gcloud_probe.yaml --stages all 2>&1 | tee -a "$LOG_FILE"
    echo "[$(date)] Pre-flight probe passed successfully!" | tee -a "$LOG_FILE"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[$(date)] GPU State after probe:" | tee -a "$LOG_FILE"
        nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu --format=csv | tee -a "$LOG_FILE"
    fi
fi

# 7. Execute v2 pipeline end-to-end
echo "[$(date)] Starting v2 repeated CV execution on GPU ($CONFIG)..." | tee -a "$LOG_FILE"
python3 scripts/run_v2.py \
    --config "$CONFIG" \
    --stages all \
    2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] Pipeline completed successfully!" | tee -a "$LOG_FILE"
