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


BOX_THRESHOLD_LOW = 0.10   # candidates below BOX_THRESHOLD are kept only if they pass the prior filter
SECOND_PASS_THRESHOLD = 0.04
ROBOT_MAX_FRAC = 0.5       # box mostly on the (prior) robot render -> rejected
AREA_RATIO = (0.15, 6.0)   # box area vs. the program's own predicted box area (loose: shapes are not fitted yet)
# adjectives from the program's declared material class (SDK prior table), not from the target
MATERIAL_ADJ = {"wood": "wooden", "metal": "metal", "ceramic": "ceramic", "glass": "glass",
                "plastic": "plastic", "cardboard_package": "cardboard", "filled_bottle": "glass"}


def prompts_for(obj, material_class=None):
    """Text prompts derived only from the world program's own names/categories/materials."""
    out = [obj["name"].replace("_", " ")]
    cat = obj["category"].replace("_", " ")
    if cat not in out:
        out.append(cat)
    adj = MATERIAL_ADJ.get(material_class)
    if adj:
        out.append(f"{adj} {cat.split()[-1]}")
    return out


def plausible(box, name, prior):
    """Loose plausibility check of a candidate box against the Builder-rendered prior."""
    if prior is None:
        return True
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    rob = prior["robot_mask"][max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
    if rob.size and rob.mean() > ROBOT_MAX_FRAC:
        return False
    wh = prior["objects"][name].get("box_wh")
    if wh:
        r = (x1 - x0) * (y1 - y0) / max(wh[0] * wh[1], 1)
        if not (AREA_RATIO[0] <= r <= AREA_RATIO[1]):
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True, help="compiled world dir (object list)")
    p.add_argument("--evidence", required=True, help="evidence root with <episode>/frame0.png")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default="", help="comma list; default = all")
    p.add_argument("--prior", default="", help="perception_prior dir (Builder render of the world)")
    a = p.parse_args()
    prior = None
    if a.prior:
        prior = json.load(open(os.path.join(a.prior, "perception_prior.json")))
        prior["robot_mask"] = np.load(os.path.join(a.prior, "perception_prior.npz"))["robot_mask"]

    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                              Sam2Processor)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gp = AutoProcessor.from_pretrained(f"{MODELS}/grounding-dino-base")
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(f"{MODELS}/grounding-dino-base").to(dev).eval()
    sp = Sam2Processor.from_pretrained(f"{MODELS}/sam2.1-hiera-large")
    sm = Sam2Model.from_pretrained(f"{MODELS}/sam2.1-hiera-large").to(dev).eval()

    with open(os.path.join(a.world, "world.json")) as f:
        wj = json.load(f)
    objects, mats = wj["objects"], wj.get("materials", {})
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
            for prompt in prompts_for(o, mats.get(o.get("material"))):
                inp = gp(images=big, text=prompt + ".", return_tensors="pt").to(dev)
                with torch.no_grad():
                    outp = gm(**inp)
                thr = BOX_THRESHOLD_LOW if prior is not None else BOX_THRESHOLD
                r = gp.post_process_grounded_object_detection(
                    outp, inp.input_ids, threshold=thr, text_threshold=thr, target_sizes=[(H, W)])[0]
                for s, b in zip(r["scores"].tolist(), r["boxes"].tolist()):
                    b = [float(v) for v in b]
                    if plausible(b, o["name"], prior):
                        cands.append((float(s), o["name"], prompt, b))
        dets = {}
        for s, name, prompt, box in sorted(cands, key=lambda c: -c[0]):
            if name in dets or any(box_iou(box, d["box"]) > MAX_OVERLAP for d in dets.values()):
                continue
            dets[name] = {"prompt": prompt, "score": s, "box": box}
        # second pass, only for objects still unassigned: lower score threshold, same plausibility filter,
        # and the box must not overlap any already assigned box at all (low-score boxes are noisy)
        missing = [o for o in objects if o["name"] not in dets]
        if prior is not None and missing:
            low = []
            for o in missing:
                for prompt in prompts_for(o, mats.get(o.get("material"))):
                    inp = gp(images=big, text=prompt + ".", return_tensors="pt").to(dev)
                    with torch.no_grad():
                        outp = gm(**inp)
                    r = gp.post_process_grounded_object_detection(
                        outp, inp.input_ids, threshold=SECOND_PASS_THRESHOLD,
                        text_threshold=SECOND_PASS_THRESHOLD, target_sizes=[(H, W)])[0]
                    for s, b in zip(r["scores"].tolist(), r["boxes"].tolist()):
                        b = [float(v) for v in b]
                        if plausible(b, o["name"], prior):
                            low.append((float(s), o["name"], prompt, b))
            for s, name, prompt, box in sorted(low, key=lambda c: -c[0]):
                if name in dets or any(box_iou(box, d["box"]) > 0.05 for d in dets.values()):
                    continue
                dets[name] = {"prompt": prompt, "score": s, "box": box, "second_pass": True}

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
