#!/usr/bin/env bash
# ==============================================================================
# scripts/install_idle_watchdog_service.sh - Install idle watchdog as systemd service
# ==============================================================================
set -euo pipefail

TIMEOUT="${1:-20}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_FILE="/etc/systemd/system/graphr1-idle-watchdog.service"

echo "=== Installing graphr1 idle watchdog service (Timeout: ${TIMEOUT}m) ==="

sudo bash -c "cat << EOF > $SERVICE_FILE
[Unit]
Description=graph-r1 Cloud VM Idle Shutdown Watchdog
After=network.target

[Service]
Type=simple
ExecStart=${REPO_DIR}/scripts/idle_watchdog.sh ${TIMEOUT}
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
sudo systemctl enable graphr1-idle-watchdog.service
sudo systemctl restart graphr1-idle-watchdog.service

echo "=== graphr1-idle-watchdog service successfully installed and active! ==="
sudo systemctl status graphr1-idle-watchdog.service --no-pager || true
