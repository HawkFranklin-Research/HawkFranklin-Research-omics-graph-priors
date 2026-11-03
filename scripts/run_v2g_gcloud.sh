#!/usr/bin/env bash
# Config-driven generalized runner. Never edit the frozen HNSCC runner.
set -euo pipefail
CONFIG="configs/v2g/brca_gcloud.yaml"
DO_PROBE=0
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
IDLE_TIMEOUT_MINUTES="${IDLE_TIMEOUT_MINUTES:-20}"
GRAPH_R1_PYTHON="${GRAPH_R1_PYTHON:-python3}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --probe) DO_PROBE=1; shift ;;
        --auto-shutdown) AUTO_SHUTDOWN=1; shift ;;
        --idle-timeout) IDLE_TIMEOUT_MINUTES="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *.yaml|*.yml) CONFIG="$1"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing cloud run: tracked files differ from HEAD. Commit code and configs first." >&2
    exit 2
fi

# One config read supplies all data, output and probe settings; no scientific values live here.
SETTINGS=$("$GRAPH_R1_PYTHON" - "$CONFIG" <<'PY'
import json, os, re, sys
from pathlib import Path
from graphr1g.config import load_config
cfg = load_config(sys.argv[1])
hf = cfg['data'].get('hf') or {}
repo = os.environ.get('HF_DATASET_REPO') or hf.get('repo')
revision = os.environ.get('HF_DATASET_REVISION') or hf.get('revision') or ''
if not revision or '<' in revision or 'to record' in revision:
    raise SystemExit('Refusing cloud run: record an immutable HF revision first')
if not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
    raise SystemExit('Refusing cloud run: HF revision must be a full commit SHA')
directory = hf.get('local_dir')
if not repo or not directory or not hf.get('allow_patterns'):
    raise SystemExit('Cloud config requires data.hf repo, local_dir and allow_patterns')
if Path(directory).resolve() != Path(cfg['data']['processed_root']).resolve():
    raise SystemExit('data.processed_root must equal data.hf.local_dir')
if Path(cfg['data']['reference_root']).resolve() != (Path(directory)/'references').resolve():
    raise SystemExit('data.reference_root must be data.hf.local_dir/references')
probe = cfg.get('cloud', {}).get('probe_config') or ''
probe_output = ''
if probe:
    test = load_config(probe)
    probe_hf = test['data'].get('hf') or {}
    if (test['data']['processed_root'] != cfg['data']['processed_root']
        or test['data']['reference_root'] != cfg['data']['reference_root']
        or probe_hf.get('repo') != hf.get('repo')
        or probe_hf.get('local_dir') != directory
        or probe_hf.get('allow_patterns') != hf.get('allow_patterns')):
        raise SystemExit('Probe and main configs must use the same HF inputs')
    probe_output = str(Path(test['data']['output_root'])/test['experiment_name'])
values = [repo, revision, directory, json.dumps(hf['allow_patterns']),
          str(Path(cfg['data']['output_root'])/cfg['experiment_name']), probe, probe_output]
for value in values:
    if '\n' in value or '\r' in value: raise SystemExit('Newlines are not allowed in cloud settings')
    print(value)
PY
)
mapfile -t CFG_VALUES <<< "$SETTINGS"
export HF_DATASET_REPO="${CFG_VALUES[0]}"
export HF_DATASET_REVISION="${CFG_VALUES[1]}"
export HF_DATASET_LOCAL_DIR="${CFG_VALUES[2]}"
export HF_DATASET_ALLOW_PATTERNS="${CFG_VALUES[3]}"
OUTPUT_DIR="${CFG_VALUES[4]}"
PROBE_CONFIG="${CFG_VALUES[5]:-}"
PROBE_OUTPUT_DIR="${CFG_VALUES[6]:-}"
GCS_BUCKET="${GCS_BUCKET:-gs://pelliscope-scout-uploads/graph-r1}"
LOG_FILE="run_gcloud_v2g_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
echo "Config: $CONFIG; output: $OUTPUT_DIR; HF: $HF_DATASET_REPO@$HF_DATASET_REVISION" | tee -a "$LOG_FILE"

trap_preempt() {
    echo "[$(date)] Spot preemption: archiving the complete output directory" | tee -a "$LOG_FILE"
    if [[ -d "$OUTPUT_DIR" ]] && command -v zstd >/dev/null 2>&1; then
        checkpoint="emergency_checkpoint_$(hostname).tar.zst"
        tar -cf - "$OUTPUT_DIR" | zstd -3 > "$checkpoint" || true
        if command -v gcloud >/dev/null 2>&1; then
            gcloud storage cp "$checkpoint" "${GCS_BUCKET}/emergency/" || true
        fi
    fi
    exit 143
}
cleanup_on_exit() {
    EXIT_CODE=$?
    trap - EXIT
    set +e
    echo "[$(date)] Cleanup, exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    # No exclusions: weights, embeddings, config, protocol and source diff are all retained.
    archive_items=("$LOG_FILE")
    [[ ! -d "$OUTPUT_DIR" ]] || archive_items+=("$OUTPUT_DIR")
    if [[ -n "$PROBE_OUTPUT_DIR" && -d "$PROBE_OUTPUT_DIR" ]]; then archive_items+=("$PROBE_OUTPUT_DIR"); fi
    ARCHIVE_NAME="v2g_results_$(hostname)_$(date +%Y%m%d_%H%M%S).tar.zst"
    if command -v zstd >/dev/null 2>&1; then
        tar -cf - "${archive_items[@]}" | zstd -T0 -3 > "$ARCHIVE_NAME"
    else
        ARCHIVE_NAME="${ARCHIVE_NAME%.tar.zst}.tar.gz"
        tar -czf "$ARCHIVE_NAME" "${archive_items[@]}"
    fi
    if command -v gcloud >/dev/null 2>&1 && [[ -f "$ARCHIVE_NAME" ]]; then
        gcloud storage cp "$ARCHIVE_NAME" "${GCS_BUCKET}/runs/" || echo "GCS upload failed; retain the local archive on the persistent disk." >&2
    fi
    # Results are delivered via GCS. Git publication remains an explicit author action.
    if [[ "$AUTO_SHUTDOWN" == 1 ]]; then
        sudo poweroff || sudo shutdown -h now
    else
        echo "Runner finished; the idle watchdog remains active." | tee -a "$LOG_FILE"
    fi
    exit "$EXIT_CODE"
}
trap trap_preempt SIGTERM
trap cleanup_on_exit EXIT

if ! pgrep -f "idle_watchdog.sh" >/dev/null 2>&1; then
    nohup ./scripts/idle_watchdog.sh "$IDLE_TIMEOUT_MINUTES" >> "$LOG_FILE" 2>&1 &
fi
if ! "$GRAPH_R1_PYTHON" -c 'import torch, torch_geometric, gseapy, huggingface_hub, tabulate' >/dev/null 2>&1; then
    "$GRAPH_R1_PYTHON" -m pip install -q -r requirements-v2.txt 2>&1 | tee -a "$LOG_FILE"
fi

# Always reconcile the pinned snapshot, even when a previous cancer's files are present.
"$GRAPH_R1_PYTHON" - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
if not os.environ.get('HF_TOKEN'):
    for file in (Path.home()/'credentials/hugging-face-read-token.txt',
                 Path('/home/prime/Documents/hugging-face-read-token.txt')):
        if file.is_file():
            os.environ['HF_TOKEN'] = file.read_text().strip()
            break
if not os.environ.get('HF_TOKEN'): raise SystemExit('A private HF read token is required')
snapshot_download(repo_id=os.environ['HF_DATASET_REPO'], repo_type='dataset',
                  revision=os.environ['HF_DATASET_REVISION'], local_dir=os.environ['HF_DATASET_LOCAL_DIR'],
                  allow_patterns=json.loads(os.environ['HF_DATASET_ALLOW_PATTERNS']), token=os.environ['HF_TOKEN'])
print('Pinned HF snapshot is ready at', os.environ['HF_DATASET_LOCAL_DIR'])
PY
if [[ "$DO_PROBE" == 1 || "${PROBE:-0}" == 1 ]]; then
    [[ -n "$PROBE_CONFIG" ]] || { echo "cloud.probe_config is required for --probe" >&2; exit 2; }
    echo "Running probe: $PROBE_CONFIG" | tee -a "$LOG_FILE"
    "$GRAPH_R1_PYTHON" scripts/run_v2g.py --config "$PROBE_CONFIG" --stages all 2>&1 | tee -a "$LOG_FILE"
    echo "Probe passed; starting the main config." | tee -a "$LOG_FILE"
fi
"$GRAPH_R1_PYTHON" scripts/run_v2g.py --config "$CONFIG" --stages all 2>&1 | tee -a "$LOG_FILE"
echo "Generalized pipeline completed successfully." | tee -a "$LOG_FILE"
