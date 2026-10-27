#!/usr/bin/env bash
# CoWM-PO S4.5: live Code Agent build with the CodeWorld-Safe "default" provider.
# The key is read with hidden input, kept only in this process's environment, never written to disk.
set -u
if [ -z "${CWS_DEFAULT_API_KEY:-}" ]; then
  read -r -s -p "CWS_DEFAULT_API_KEY: " CWS_DEFAULT_API_KEY; echo
  export CWS_DEFAULT_API_KEY
fi
RUN=/data_cf/cowmpo/worlds/s45_live_goal_$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"
cd /mnt/nfs/fjj/Code/CoWMPO/toolkits/cowm/agent
/data_cf/cowmpo/.venv-builder/bin/python build_agent.py --provider default --rounds 4 --run "$RUN" 2>&1 \
  | tee "$RUN/agent.log"
echo "run dir: $RUN"
