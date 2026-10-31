#!/usr/bin/env bash
# CoWM-PO S5 perception stack (EvidenceAdapter): SAM2.1 + Grounding DINO, pinned. Runs on the host
# (needs network); the Builder sandbox later mounts the venv and weights read-only.
set -u
UV=/root/.local/bin/uv
V=/data_cf/cowmpo/.venv-perception
W=/data_cf/cowmpo/builder_assets/models
export HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_XET=1
mkdir -p "$W"
echo "START $(date)"
$UV venv "$V" --python 3.11.14
$UV pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0" "torchvision==0.26.0"
$UV pip install --python "$V/bin/python" "transformers==5.21.0" "numpy==1.26.4" "pillow==12.3.0" \
  "scipy==1.15.3" "mujoco==3.3.7" "safetensors" "huggingface_hub"
hf download facebook/sam2.1-hiera-large --revision 665f8e2ad61cf5f53d65644ff27c8ee525124610 \
  --local-dir "$W/sam2.1-hiera-large" --exclude "*.pt" && echo SAM2_DONE
hf download IDEA-Research/grounding-dino-base --revision 12bdfa3120f3e7ec7b434d90674b3396eccf88eb \
  --local-dir "$W/grounding-dino-base" --exclude "pytorch_model.bin" && echo GDINO_DONE
$UV pip freeze --python "$V/bin/python" > /data_cf/cowmpo/perception_freeze.txt
"$V/bin/python" -c "import transformers, torch; from transformers import Sam2Model, AutoModelForZeroShotObjectDetection; print('IMPORT_OK', transformers.__version__, torch.__version__)"
echo "SETUP_EXIT $(date)"
