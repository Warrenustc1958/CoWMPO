#!/usr/bin/env bash
# Run a command under forensic recording (for the unexplained SIGKILL / exit 137 of 10-29).
# Records at start: pid/pgid/sid, cgroup, session, limits; every 5 s: RSS of the process tree, GPU memory,
# available host memory; at exit: exit code, and if it was killed by a signal: journalctl (system + kernel)
# for the run window, dmesg tail, loginctl sessions, surviving process tree. Nothing is changed on the host.
# Usage: forensic_run.sh <outdir> <command...>
set -u
OUT=$1; shift
mkdir -p "$OUT"
T0=$(date "+%Y-%m-%d %H:%M:%S")
{ echo "start $T0 host $(hostname) shell_pid $$"; echo "cmd $*"; echo "cgroup $(cat /proc/self/cgroup)";
  echo "session ${XDG_SESSION_ID:-none} ssh ${SSH_CONNECTION:-none}"; ulimit -a; } > "$OUT/start.txt"
"$@" > "$OUT/cmd.log" 2>&1 &
PID=$!
echo "child_pid $PID pgid $(ps -o pgid= -p $PID) sid $(ps -o sid= -p $PID)" >> "$OUT/start.txt"
while kill -0 $PID 2>/dev/null; do
  rss=$(ps -o rss= --ppid $PID -p $PID 2>/dev/null | awk '{s+=$1} END {printf "%.2f", s/1048576}')
  echo "$(date +%T) tree_rss_gb=$rss avail_gb=$(free -g | awk '/Mem:/{print $7}') \
gpu_mib=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' ')" >> "$OUT/samples.txt"
  sleep 5
done
wait $PID; RC=$?
T1=$(date "+%Y-%m-%d %H:%M:%S")
echo "end $T1 exit $RC" >> "$OUT/start.txt"
if [ $RC -gt 128 ]; then
  echo "signal $((RC - 128))" >> "$OUT/start.txt"
  journalctl --since "$T0" --until "$T1" --no-pager -o short-iso 2>/dev/null \
    | grep -vE "etcd\[|kubelet\[.*(eviction_manager|container_gc|image_gc)" > "$OUT/journal.txt"
  journalctl -k --since "$T0" --until "$T1" --no-pager -o short-iso > "$OUT/kernel.txt" 2>&1
  dmesg -T 2>/dev/null | tail -50 > "$OUT/dmesg_tail.txt"
  loginctl list-sessions --no-legend > "$OUT/sessions.txt" 2>&1
  ps -eo pid,ppid,pgid,sid,etime,args --forest > "$OUT/ps_forest.txt"
fi
echo "FORENSIC_RUN_DONE exit $RC out $OUT"
exit $RC
