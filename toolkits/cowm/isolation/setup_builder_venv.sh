#!/usr/bin/env bash
# Create the CoWM-PO Builder venv: physics/geometry stack pinned to the Target venv versions, no libero.
set -eu
UV=/root/.local/bin/uv
TARGET_PY=/data_cf/cowmpo/.venv-oft-wan/bin/python
V=/data_cf/cowmpo/.venv-builder
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p /data_cf/cowmpo/evidence /data_cf/cowmpo/worlds /data_cf/cowmpo/builder_assets
$UV venv "$V" --python 3.11.14

# same versions as the Target venv for everything the executor shares with it
PINS=$($UV pip freeze --python "$TARGET_PY" \
  | grep -iE '^(mujoco|numpy|scipy|numba|llvmlite|pillow|opencv-python-headless|opencv-python|termcolor)==' \
  | grep -v -i '^opencv-python==' || true)
echo "pins: $PINS"
$UV pip install --python "$V/bin/python" $PINS "trimesh==4.12.2" "coacd==1.0.11"
$UV pip install --python "$V/bin/python" --no-deps "robosuite==1.4.1"

cp "$HERE/sandbox_guard.py" "$V/lib/python3.11/site-packages/sitecustomize.py"
$UV pip freeze --python "$V/bin/python" > /data_cf/cowmpo/builder_freeze.txt
if "$V/bin/python" -c "import libero" 2>/dev/null; then echo "ERROR: libero importable in builder venv"; exit 1; fi
echo "SETUP_DONE"
