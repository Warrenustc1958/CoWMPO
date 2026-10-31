"""Robot track repair v7: RGB-only per-frame re-detection for robot parts that ENTER the view after frame 0.

First-frame-anchored propagation cannot track robot links invisible at frame 0 (ym_h100: upper arm enters
from the image border). Rule fixed before evaluation (2026-10-30), no GT, no per-probe choice:
  per frame: Grounding DINO on the full frame with prompts PROMPTS (threshold DET_THR) -> SAM2 box masks;
  keep connected components that touch the current track (dilated DIL px) or the image border;
  final = existing track (v5 / v6 union) OR kept components.
Inputs are read-only; writes rack_robot_redetect_v7/<key>/<frame>/{masks.npz, association.json}.
usage: python run_robot_redetect_v7.py   (inside the Builder sandbox, perception venv, GPU)
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, label

from cowm.init.perceive import MODELS

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
E = Path("/data_cf/cowmpo/evidence")
OUT = R / "rack_robot_redetect_v7"
EP = "libero_goal_t0_i0"
PROMPTS = ["robot arm", "robot"]
DET_THR = 0.25
DIL = 3
V6 = R / "rack_robot_union_generic_v6"
V5 = R / "rack_robot_union_v5"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def queries():
    out = []
    for r in json.loads((V6 / "freeze.json").read_text())["queries"]:
        out.append({"key": r["key"], "group": r["group"], "public": r["public"], "steps": r["steps"],
                    "track": str(V6 / "frames_merged" / r["key"])})
    for r in json.loads((V5 / "freeze.json").read_text())["queries"]:
        out.append({"key": r["key"], "group": r["group"], "public": r["public"], "steps": r["steps"],
                    "track": str(V5 / "frames" / r["key"])})
    return out


def main():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gp = AutoProcessor.from_pretrained(f"{MODELS}/grounding-dino-base")
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(f"{MODELS}/grounding-dino-base").to(dev).eval()
    sp = Sam2Processor.from_pretrained(f"{MODELS}/sam2.1-hiera-large")
    sm = Sam2Model.from_pretrained(f"{MODELS}/sam2.1-hiera-large").to(dev).eval()
    OUT.mkdir(exist_ok=False)
    qs = queries()
    (OUT / "freeze.json").write_text(json.dumps({"queries": qs, "prompts": PROMPTS, "det_threshold": DET_THR,
                                                 "dilation": DIL, "script_sha256": sha(__file__),
                                                 "rule": "keep re-detected components touching the track or the "
                                                         "image border; union with track", "GT_inputs": False},
                                                indent=1))
    for q in qs:
        for s in q["steps"]:
            f = f"{EP}__f{s:03d}"
            img = Image.open(Path(q["public"]) / f / "frame0.png").convert("RGB")
            W, H = img.size
            tr_meta = json.loads((Path(q["track"]) / f / "association.json").read_text())
            if tr_meta["masks_sha256"] != sha(Path(q["track"]) / f / "masks.npz"):
                raise ValueError("track mask hash")
            with np.load(Path(q["track"]) / f / "masks.npz") as a:
                track = a["robot"].astype(bool)
            boxes = []
            for pr in PROMPTS:
                inp = gp(images=img, text=pr + ".", return_tensors="pt").to(dev)
                with torch.no_grad():
                    o = gm(**inp)
                r = gp.post_process_grounded_object_detection(o, inp.input_ids, threshold=DET_THR,
                                                              text_threshold=DET_THR, target_sizes=[(H, W)])[0]
                boxes += [[float(v) for v in b] for b in r["boxes"].tolist()]
            det = np.zeros((H, W), bool)
            if boxes:
                inp = sp(images=img, input_boxes=[boxes], return_tensors="pt").to(dev)
                with torch.no_grad():
                    o = sm(**inp, multimask_output=False)
                ms = sp.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
                for k in range(len(boxes)):
                    det |= ms[k, 0].numpy().astype(bool)
            lab, n = label(det & ~track)
            near = binary_dilation(track, iterations=DIL)
            border = np.zeros((H, W), bool)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            added = np.zeros((H, W), bool)
            kept = 0
            for c in range(1, n + 1):
                comp = lab == c
                if (comp & near).any() or (comp & border).any():
                    added |= comp
                    kept += 1
            final = track | added
            d = OUT / q["key"] / f
            d.mkdir(parents=True)
            np.savez_compressed(d / "masks.npz", robot=final)
            (d / "association.json").write_text(json.dumps({
                "episode": f, "algorithm": "robot_redetect_v7",
                "objects": {"robot": {"status": "matched" if final.any() else "missing",
                                      "track_area": int(track.sum()), "added_area": int(added.sum()),
                                      "n_boxes": len(boxes), "components_kept": kept}},
                "masks_sha256": sha(d / "masks.npz"), "track_source": str(Path(q["track"]) / f)}, indent=1))
        print("REDETECT_DONE", q["group"], q["key"], len(q["steps"]), flush=True)
    print("REDETECT_ALL_DONE", sum(len(q["steps"]) for q in qs), flush=True)


if __name__ == "__main__":
    main()
