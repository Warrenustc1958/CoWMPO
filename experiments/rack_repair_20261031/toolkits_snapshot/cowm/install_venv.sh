#!/usr/bin/env bash
# CoWM-PO venv: OpenVLA-OFT + LIBERO + Wan (diffsynth fork), on local NVMe.
# --no-root: do not run sys_deps.sh (apt-get) on the shared host.
set -u
REPO=/mnt/nfs/fjj/Code/CoWMPO
VENV=/data_cf/cowmpo/.venv-oft-wan
mkdir -p /data_cf/cowmpo
cd "$REPO"
echo "START $(date)"
bash requirements/install.sh embodied --model openvla-oft --env wan --venv "$VENV" --no-root
rc=$?
echo "INSTALL_EXIT rc=$rc $(date)"
