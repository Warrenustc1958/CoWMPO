"""S5 EvidenceAdapter (Builder side, plan §3.7C / §9.7-C): RGB evidence frame -> per-object 2D masks.

Grounding DINO (text prompt per world object) -> best box -> SAM2 box-prompted mask.
Inputs are only Builder-visible: evidence frame0.png + the world's own object names/categories.
Runs inside the Builder sandbox with the perception venv and local weights (HF_HUB_OFFLINE=1).

Output per episode: <out>/<episode>/masks.npz  (one bool [H, W] per object, absent if not detected)
                    <out>/<episode>/perception.json  {obj: {prompt, score, box, area}}
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

MODELS = "/data_cf/cowmpo/builder_assets/models"
UPSCALE = 3            # 256 px evidence is small for DINO; upsample before detection
BOX_THRESHOLD = 0.20
MAX_OVERLAP = 0.6


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def prompts_for(obj):
    """Text prompts derived only from the world program's own names/categories (no benchmark names)."""
    out = [obj["name"].replace("_", " ")]
    cat = obj["category"].replace("_", " ")
    if cat not in out:
        out.append(cat)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True, help="compiled world dir (object list)")
    p.add_argument("--evidence", required=True, help="evidence root with <episode>/frame0.png")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default="", help="comma list; default = all")
    a = p.parse_args()

    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                              Sam2Processor)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gp = AutoProcessor.from_pretrained(f"{MODELS}/grounding-dino-base")
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(f"{MODELS}/grounding-dino-base").to(dev).eval()
    sp = Sam2Processor.from_pretrained(f"{MODELS}/sam2.1-hiera-large")
    sm = Sam2Model.from_pretrained(f"{MODELS}/sam2.1-hiera-large").to(dev).eval()

    with open(os.path.join(a.world, "world.json")) as f:
        objects = json.load(f)["objects"]
    eps = a.episodes.split(",") if a.episodes else sorted(os.listdir(a.evidence))
    eps = [e for e in eps if os.path.isfile(os.path.join(a.evidence, e, "frame0.png"))]
    print("device", dev, "episodes", len(eps), "objects", [o["name"] for o in objects], flush=True)

    for ep in eps:
        img = Image.open(os.path.join(a.evidence, ep, "frame0.png")).convert("RGB")
        W, H = img.size
        big = img.resize((W * UPSCALE, H * UPSCALE), Image.BICUBIC)
        # all candidate boxes for every object, then greedy one-box-per-object assignment by score;
        # a box overlapping an already assigned box (IoU > MAX_OVERLAP) is rejected (e.g. "box" -> cabinet)
        cands = []
        for o in objects:
            for prompt in prompts_for(o):
                inp = gp(images=big, text=prompt + ".", return_tensors="pt").to(dev)
                with torch.no_grad():
                    outp = gm(**inp)
                r = gp.post_process_grounded_object_detection(
                    outp, inp.input_ids, threshold=BOX_THRESHOLD, text_threshold=BOX_THRESHOLD,
                    target_sizes=[(H, W)])[0]
                for s, b in zip(r["scores"].tolist(), r["boxes"].tolist()):
                    cands.append((float(s), o["name"], prompt, [float(v) for v in b]))
        dets = {}
        for s, name, prompt, box in sorted(cands, key=lambda c: -c[0]):
            if name in dets or any(box_iou(box, d["box"]) > MAX_OVERLAP for d in dets.values()):
                continue
            dets[name] = {"prompt": prompt, "score": s, "box": box}

        masks = {}
        if dets:
            names = list(dets)
            boxes = [[dets[n]["box"] for n in names]]
            inp = sp(images=img, input_boxes=boxes, return_tensors="pt").to(dev)
            with torch.no_grad():
                outp = sm(**inp, multimask_output=False)
            m = sp.post_process_masks(outp.pred_masks.cpu(), inp["original_sizes"].cpu())[0]  # [N, 1, H, W]
            for k, n in enumerate(names):
                masks[n] = m[k, 0].numpy().astype(bool)
                dets[n]["area"] = int(masks[n].sum())
        d = os.path.join(a.out, ep)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "masks.npz"), **masks)
        with open(os.path.join(d, "perception.json"), "w") as fh:
            json.dump(dets, fh, indent=1)
        print(ep, {n: round(v["score"], 2) for n, v in dets.items()},
              "missing", [o["name"] for o in objects if o["name"] not in dets], flush=True)


if __name__ == "__main__":
    main()
