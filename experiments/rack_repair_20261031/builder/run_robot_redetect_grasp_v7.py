"""Apply the frozen v7 robot re-detection rule to grasp videos (same prompts/threshold/dilation as rack v7).

Input track: <track>/<ep>__fNNN/{masks.npz, association.json} (objects robot, cream_cheese).
Evidence:    <evidence>/<ep>__grasp/fNNN/frame0.png
Output:      <out>/<ep>__fNNN/ with robot = track OR kept re-detected components; every other object copied
             unchanged (mask and status), so the existing Target validator scores it unchanged. No GT.
usage: python run_robot_redetect_grasp_v7.py --track D --evidence D --out D --episodes a,b
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, label

from cowm.init.perceive import MODELS

PROMPTS = ["robot arm", "robot"]   # identical to rack v7
DET_THR = 0.25
DIL = 3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ("--track", "--evidence", "--out", "--episodes"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    track, ev, out = Path(a.track), Path(a.evidence), Path(a.out)
    eps = a.episodes.split(",")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gp = AutoProcessor.from_pretrained(f"{MODELS}/grounding-dino-base")
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(f"{MODELS}/grounding-dino-base").to(dev).eval()
    sp = Sam2Processor.from_pretrained(f"{MODELS}/sam2.1-hiera-large")
    sm = Sam2Model.from_pretrained(f"{MODELS}/sam2.1-hiera-large").to(dev).eval()
    out.mkdir(parents=True, exist_ok=False)
    (out / "freeze.json").write_text(json.dumps({"track": str(track), "evidence": str(ev), "episodes": eps,
                                                 "prompts": PROMPTS, "det_threshold": DET_THR, "dilation": DIL,
                                                 "script_sha256": sha(__file__), "GT_inputs": False}, indent=1))
    n_frames = 0
    for ep in eps:
        frames = sorted(p for p in track.glob(f"{ep}__f*") if (p / "masks.npz").is_file())
        for fd in frames:
            step = int(fd.name.split("__f")[1])
            meta = json.loads((fd / "association.json").read_text())
            if meta.get("masks_sha256") and meta["masks_sha256"] != sha(fd / "masks.npz"):
                raise ValueError(f"track mask hash {fd}")
            with np.load(fd / "masks.npz") as z:
                arrs = {k: z[k] for k in z.files}
            robot = arrs["robot"].astype(bool) if meta["objects"]["robot"]["status"] == "matched" else \
                np.zeros_like(arrs["robot"], bool)
            img = Image.open(ev / f"{ep}__grasp" / f"f{step:03d}" / "frame0.png").convert("RGB")
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
            lab, n = label(det & ~robot)
            near = binary_dilation(robot, iterations=DIL)
            border = np.zeros((H, W), bool)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            added = np.zeros((H, W), bool)
            for c in range(1, n + 1):
                comp = lab == c
                if (comp & near).any() or (comp & border).any():
                    added |= comp
            arrs["robot"] = robot | added
            d = out / fd.name
            d.mkdir()
            np.savez_compressed(d / "masks.npz", **arrs)
            new_meta = dict(meta)
            new_meta["objects"] = dict(meta["objects"])
            new_meta["objects"]["robot"] = dict(meta["objects"]["robot"],
                                                status="matched" if arrs["robot"].any() else "missing",
                                                redetect_added_area=int(added.sum()))
            new_meta["masks_sha256"] = sha(d / "masks.npz")
            new_meta["algorithm_robot"] = "robot_redetect_v7"
            (d / "association.json").write_text(json.dumps(new_meta, indent=1))
            n_frames += 1
        print("GRASP_REDETECT_EP", ep, len(frames), flush=True)
    print("GRASP_REDETECT_DONE", n_frames, flush=True)


if __name__ == "__main__":
    main()
