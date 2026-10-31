#!/usr/bin/env bash
# Run a command inside the CoWM-PO Builder sandbox (plan §3.7C).
#   (1) Builder venv without libero
#   (2) private mount namespace: /mnt/nfs, /data_cf, /root, /home, /tmp, /opt are covered by empty
#       tmpfs; only the ALLOW list is bound back (code ro, evidence ro, outputs rw)
#   (3) private network namespace: no network
#   (4) sitecustomize audit guard (sandbox_guard.py)
# Usage: run_builder.sh <cmd> [args...]     e.g. run_builder.sh python toolkits/cowm/isolation/test_isolation.py
set -eu
SELF="$(readlink -f "$0")"
if [ "${1:-}" != "--inner" ]; then
  exec unshare --net --mount --propagation private "$SELF" --inner "$@"
fi
shift

BUILDER_VENV=/data_cf/cowmpo/.venv-builder
ALLOW=(
  "/mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm:ro"
  "/data_cf/cowmpo/evidence:ro"
  "/data_cf/cowmpo/builder_assets:ro"
  "/data_cf/cowmpo/.venv-builder:ro"
  "/data_cf/cowmpo/.venv-perception:ro"
  "/data_cf/cowmpo/worlds:rw"
  "/root/.local/share/uv/python:ro"
)
HIDE_ROOTS=(/mnt/nfs /data_cf /root /home /tmp /opt)

KEEP=/run/cowm-keep
mkdir -p "$KEEP"
mount -t tmpfs cowm-keep "$KEEP"
i=0
for spec in "${ALLOW[@]}"; do
  p="${spec%:*}"
  mkdir -p "$KEEP/$i"
  mount --bind "$p" "$KEEP/$i"
  i=$((i + 1))
done
for r in "${HIDE_ROOTS[@]}"; do
  [ -d "$r" ] && mount -t tmpfs -o size=256m cowm-hidden "$r"
done
i=0
for spec in "${ALLOW[@]}"; do
  p="${spec%:*}"; m="${spec##*:}"
  mkdir -p "$p"
  mount --bind "$KEEP/$i" "$p"
  [ "$m" = "ro" ] && mount -o remount,bind,ro "$p"
  i=$((i + 1))
done
umount -R -l "$KEEP"

export COWM_SANDBOX=1
export COWM_EVIDENCE=/data_cf/cowmpo/evidence
export COWM_WORLDS=/data_cf/cowmpo/worlds
export COWM_GUARD_LOG=/data_cf/cowmpo/worlds/guard.log
export HOME=/root
export NUMBA_CACHE_DIR=/tmp/numba
export HF_HUB_OFFLINE=1
unset LD_PRELOAD PYTHONHOME
export PYTHONPATH="${COWM_PYTHONPATH:-/mnt/nfs/fjj/Code/CoWMPO/toolkits}"
export PATH="$BUILDER_VENV/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
cd /mnt/nfs/fjj/Code/CoWMPO/toolkits
exec "$@"
