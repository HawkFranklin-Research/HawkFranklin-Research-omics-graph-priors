#!/usr/bin/env bash
# ==============================================================================
# scripts/provision_gcloud_spot.sh - Launch GCP Spot T4 GPU Instance
# ==============================================================================
set -euo pipefail

INSTANCE_NAME="${1:-graph-r1-t4-runner}"
ZONE="${2:-us-central1-a}"
PROJECT="${3:-pelliscope-scout}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"

echo "=== Provisioning GCP Spot VM: $INSTANCE_NAME in $ZONE ($PROJECT) ==="

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="n1-standard-4" \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --maintenance-policy="TERMINATE" \
  --provisioning-model="SPOT" \
  --instance-termination-action="STOP" \
  --boot-disk-size="100GB" \
  --boot-disk-type="pd-balanced" \
  --no-address \
  --image-family="$IMAGE_FAMILY" \
  --image-project="deeplearning-platform-release" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --metadata="install-nvidia-driver=True"

echo ""
echo "=== Instance created successfully! ==="
echo ""
echo "Step 1: Wait ~60s for initial boot & GPU driver initialization, then transfer tokens securely:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT --tunnel-through-iap --command=\"mkdir -p ~/credentials\""
echo "  gcloud compute scp /home/prime/Documents/hugging-face-read-token.txt $INSTANCE_NAME:~/credentials/hugging-face-read-token.txt --zone=$ZONE --project=$PROJECT --tunnel-through-iap"
echo "  gcloud compute scp /home/prime/Documents/github-hawp-prime-token-90-days.txt $INSTANCE_NAME:~/credentials/github-token.txt --zone=$ZONE --project=$PROJECT --tunnel-through-iap"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT --tunnel-through-iap --command=\"chmod 600 ~/credentials/*\""
echo ""
echo "Step 2: Connect to your VM and clone the repository:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT --tunnel-through-iap"
echo "  git clone https://github.com/HawkFranklin-Research/graph-r1.git"
echo "  cd graph-r1 && git checkout v2-pipeline"
echo ""
echo "Step 3: (Recommended) Install system idle-watchdog to auto-stop VM on 20 min inactivity:"
echo "  sudo ./scripts/install_idle_watchdog_service.sh 20"
echo ""
echo "Step 4: Launch probe check and full E1 repeated-CV in detached tmux with auto-shutdown:"
echo "  tmux new -s v2_run"
echo "  ./scripts/run_v2_gcloud.sh --probe --auto-shutdown"
