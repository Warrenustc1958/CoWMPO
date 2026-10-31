"""Robot tracks for the 3 rear rack probes densely exported in codex_rack_vertical_dense_v16 (derived from track_rack_v11_v7b.py), with the FROZEN v7b rule:
whole track (original vertical anchors) U matched part tracks (v4 anchors) U border-touching RGB re-detections.
Builder side, no GT. Writes rack_vertical_v11_tracks/{whole,parts,frames}.
usage: python track_rack_v11_v7b.py   (inside the Builder sandbox; perception venv)
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label

from cowm.init.perceive import MODELS

R = Path("/data_cf/cowmpo/worlds/codex_s56_20261030_1020")
C = Path("/data_cf/cowmpo/worlds/_staging_code/codex_s56_20261030_1020_v10")
E = Path("/data_cf/cowmpo/evidence/codex_rack_accept_dense_v23")
PLAN = R / "rack_accept_dense_plan_v23"
WHOLE_ANCH = R / "rack_vertical_tracking_ready_v3/anchors"
PART_ANCH = R / "rack_robot_parts_cpu_v4/anchors"
OUT = R / "rack_accept_dense_v23_tracks"
EP = "libero_goal_t0_i0"
KEYS = ["acc_x+040_y+000", "acc_x+040_y+080", "acc_x+040_y-080", "acc_x+120_y+000", "acc_x+120_y+080", "acc_x+120_y-080"]
PROMPTS, DET_THR = ["robot arm", "robot"], 0.25          # frozen v7b


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def track(key, anchors, dest):
    ready = OUT / "ready" / key
    cmd = [sys.executable, str(C / "track_probe_video_device.py"), "--data-role", "dev", "--device", "cpu",
           "--cpu-threads", "8", "--episodes", EP, "--anchors", str(anchors), "--initial-evidence",
           str(ready / "initial"), "--evidence", str(E / key), "--probes", str(ready / "probes"), "--out", str(dest)]
    with (OUT / f"{key}_{dest.parent.name}.log").open("w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError((key, dest.parent.name, r.returncode))


def main():
    resume = OUT.exists()
    OUT.mkdir(exist_ok=True)
    done = {k for k in KEYS if (OUT / "frames" / k).is_dir() and
            len(list((OUT / "frames" / k).glob("*/association.json"))) == len(json.loads((PLAN / k / "probe.json").read_text())["frames"])}
    for k in KEYS:   # keep any partial output of an interrupted run as evidence, never delete it
        for sub in ("whole", "parts", "frames"):
            d = OUT / sub / k
            if k not in done and d.exists():
                d.rename(OUT / sub / f"{k}.interrupted_{int(__import__('time').time())}")
    for key in KEYS:
        if (OUT / "ready" / key).exists():
            continue
        ready = OUT / "ready" / key
        for sub in ("initial", "probes"):
            (ready / sub / EP).mkdir(parents=True)
        shutil.copy2(E / key / f"{EP}__f000" / "frame0.png", ready / "initial" / EP / "frame0.png")
        shutil.copy2(PLAN / key / "probe.json", ready / "probes" / EP / "probe.json")
    if not resume:
      (OUT / "freeze.json").write_text(json.dumps({
        "keys": KEYS, "rule": "frozen v7b (whole U matched parts U border-touching re-detections)",
        "tracker_sha256": sha(C / "track_probe_video_device.py"), "script_sha256": sha(__file__),
        "whole_anchors": str(WHOLE_ANCH), "part_anchors": str(PART_ANCH), "GT_inputs": False}, indent=1))
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gp = AutoProcessor.from_pretrained(f"{MODELS}/grounding-dino-base")
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(f"{MODELS}/grounding-dino-base").to(dev).eval()
    sp = Sam2Processor.from_pretrained(f"{MODELS}/sam2.1-hiera-large")
    sm = Sam2Model.from_pretrained(f"{MODELS}/sam2.1-hiera-large").to(dev).eval()
    print("RESUME" if resume else "FRESH", "done:", sorted(done), flush=True)
    for key in KEYS:
        if key in done:
            continue
        track(key, WHOLE_ANCH, OUT / "whole" / key)
        track(key, PART_ANCH, OUT / "parts" / key)
        probe = json.loads((PLAN / key / "probe.json").read_text())
        for s in probe["frames"]:
            f = f"{EP}__f{s:03d}"
            union = np.zeros((256, 256), bool)
            for src, name_key in ((OUT / "whole" / key / f, None), (OUT / "parts" / key / f, None)):
                meta = json.loads((src / "association.json").read_text())
                with np.load(src / "masks.npz") as a:
                    for n, o in meta["objects"].items():
                        if o["status"] == "matched" and n in a.files:
                            union |= a[n].astype(bool)
            img = Image.open(E / key / f / "frame0.png").convert("RGB")
            W, H = img.size
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
            lab, n = label(det & ~union)
            border = np.zeros((H, W), bool)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            added = np.zeros((H, W), bool)
            for c in range(1, n + 1):
                comp = lab == c
                if (comp & border).any():
                    added |= comp
            final = union | added
            d = OUT / "frames" / key / f
            d.mkdir(parents=True)
            np.savez_compressed(d / "masks.npz", robot=final)
            (d / "association.json").write_text(json.dumps({
                "episode": f, "algorithm": "robot_tracking_v7b",
                "objects": {"robot": {"status": "matched" if final.any() else "missing",
                                      "union_area": int(union.sum()), "added_area": int(added.sum())}},
                "masks_sha256": sha(d / "masks.npz")}, indent=1))
        print("V23_TRACK_DONE", key, len(probe["frames"]), flush=True)
    print("V11_TRACK_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
