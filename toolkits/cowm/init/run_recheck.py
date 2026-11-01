"""GPU runner for cowm.init.recheck on the frozen candidate pool (Builder side; inside the sandbox; no GT).

Per episode: associate (v3 config) -> budgeted local recheck (Grounding DINO on crops, SAM2 on the full
frame for new candidates) -> re-associate. Writes <out>/<episode>/recheck.json and recheck_masks.npz
(new candidates only). The frozen pool (s5_audit_v1) is read-only input.
Crop rule (fixed before the dev run): crop resized so its longer side = CROP_LONG px; detection threshold
DET_THRESHOLD; prompt protocol = all prompts of all world objects (perceive.prompts_for).
usage: python run_recheck.py --pool <s5_audit_v1/<set>> --evidence <evidence root> --world <ref> --prior <prior>
       --out <dir> --episodes a,b,c
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from cowm.init import recheck as RC
from cowm.init.perceive import MODELS, filter_reason, prompts_for

CROP_LONG = 768
DET_THRESHOLD = 0.10


def main():
    p = argparse.ArgumentParser()
    for k in ("--pool", "--evidence", "--world", "--prior", "--out", "--episodes"):
        p.add_argument(k, required=True)
    p.add_argument("--strategy", choices=["presence", "paired-audit"], default="presence")
    p.add_argument("--raw-scores", action="store_true", help="audit only: retain all detector queries before score filtering")
    a = p.parse_args()
    if a.raw_scores and a.strategy != "paired-audit":
        raise ValueError("raw scores are an audit-only experiment")
    threshold = 0.0 if a.raw_scores else DET_THRESHOLD
    from cowm import roles
    roles.require(a.episodes.split(","), {"dev"}, "experimental local recheck")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("local recheck requires the approved GPU2: set CUDA_VISIBLE_DEVICES=2")
    if os.path.exists(a.out):
        raise FileExistsError(f"new output directory required: {a.out}")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gp = AutoProcessor.from_pretrained(f"{MODELS}/grounding-dino-base")
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(f"{MODELS}/grounding-dino-base").to(dev).eval()
    sp = sm = None
    if a.strategy == "presence":
        sp = Sam2Processor.from_pretrained(f"{MODELS}/sam2.1-hiera-large")
        sm = Sam2Model.from_pretrained(f"{MODELS}/sam2.1-hiera-large").to(dev).eval()
    wj = json.load(open(os.path.join(a.world, "world.json")))
    objects, mats = wj["objects"], wj.get("materials", {})
    names = [o["name"] for o in objects]
    protocol = [(o["name"], pr) for o in objects for pr in prompts_for(o, mats.get(o.get("material")))]
    prior = json.load(open(os.path.join(a.prior, "perception_prior.json")))
    prior["robot_mask"] = np.load(os.path.join(a.prior, "perception_prior.npz"))["robot_mask"]
    for ep in a.episodes.split(","):
        t0 = time.time()
        au = json.load(open(os.path.join(a.pool, ep, "audit.json")))
        cmz = np.load(os.path.join(a.pool, ep, "candidate_masks.npz"))
        masks = {c["cid"]: cmz[f"c{c['cid']}"] for c in au["candidates"]}
        img = Image.open(os.path.join(a.evidence, ep, "frame0.png")).convert("RGB")
        W, H = img.size
        fwd = [0]
        detection_log = []

        def detect(crop):
            x0, y0, x1, y1 = crop
            sub = img.crop((x0, y0, x1, y1))
            s = CROP_LONG / max(sub.size)
            big = sub.resize((max(1, round(sub.size[0] * s)), max(1, round(sub.size[1] * s))), Image.BICUBIC)
            out = []
            for obj, pr in protocol:
                inp = gp(images=big, text=pr + ".", return_tensors="pt").to(dev)
                with torch.no_grad():
                    o = gm(**inp)
                fwd[0] += 1
                r = gp.post_process_grounded_object_detection(o, inp.input_ids, threshold=threshold,
                                                              text_threshold=threshold,
                                                              target_sizes=[(sub.size[1], sub.size[0])])[0]
                for sc, b in zip(r["scores"].tolist(), r["boxes"].tolist()):
                    out.append({"obj": obj, "prompt": pr, "score": float(sc), "box": RC.to_original(b, crop)})
            detection_log.append({"crop": crop, "detections": out})
            return out

        def segment(boxes):
            inp = sp(images=img, input_boxes=[boxes], return_tensors="pt").to(dev)
            with torch.no_grad():
                o = sm(**inp, multimask_output=False)
            m = sp.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
            return [m[k, 0].numpy().astype(bool) for k in range(len(boxes))]

        if a.strategy == "paired-audit":
            from cowm.init.paired_recheck import audit_episode
            res = audit_episode(names, au["candidates"], masks, W, H, detect)
        else:
            res = RC.recheck_episode(names, au["candidates"], masks, W, H, detect, segment,
                                     prior_filter=lambda b, n: filter_reason(b, n, prior))
        d = os.path.join(a.out, ep)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "recheck_masks.npz"),
                            **{f"c{c['cid']}": res["masks"][c["cid"]] for c in res["added"]})
        rec = {"episode": ep, "base": res["base"]["objects"], "final": res["final"]["objects"], "log": res["log"],
               "removed": res["removed"], "added": res["added"], "crops": res["model_calls"],
               "dino_forward_passes": fwd[0], "seconds": round(time.time() - t0, 2),
               "changed_untriggered": res["changed_untriggered"],
               "raw_detections": detection_log,
               "config": {"strategy": a.strategy, "crop_long": CROP_LONG, "det_threshold": threshold,
                          "raw_scores": a.raw_scores,
                          "max_objects": RC.MAX_OBJECTS,
                          "max_crops": RC.MAX_CROPS, "region_box_iou": RC.REGION_BOX_IOU, "ctx": RC.CTX}}
        json.dump(rec, open(os.path.join(d, "recheck.json"), "w"), indent=1)
        print(ep, "triggered", list(res["log"]), "crops", res["model_calls"], "fwd", fwd[0], "s", rec["seconds"],
              {n: (e["status_before"], e["status_after"]) for n, e in res["log"].items()}, flush=True)


if __name__ == "__main__":
    main()
