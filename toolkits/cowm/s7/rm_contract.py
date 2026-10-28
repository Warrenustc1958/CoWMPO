"""S7 RM input-contract check on labelled data (read-only; no RM / threshold / preprocessing change).

The frozen shared RM was trained on the RLinf-Wan LIBERO-Goal dataset (<traj>.npy frames with a 0/1 reward
label). Score every labelled frame under the deployed contract (RGB, [-1, 1], image as stored) and under
alternative orientations; report per-task TP/FP/FN/TN. Also compare the dataset's image orientation with the
CoWM-PO evidence frames (same initial scenes) to see which convention each pipeline feeds the RM.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/nfs/fjj/Code/CoWMPO/toolkits")
from diffsynth.models.reward_model import TaskEmbedResnetRewModel, instruction_to_task_id  # noqa: E402

D = "/mnt/nfs/fjj/Models/RLinf-Wan-LIBERO-Goal/dataset"
OUT = "/data_cf/cowmpo/diagnostics/s7_rm_contract"
ORIENT = {"as_stored": lambda x: x, "hflip": lambda x: x[:, ::-1], "vflip": lambda x: x[::-1],
          "rot180": lambda x: x[::-1, ::-1]}


def load(f):
    d = np.load(os.path.join(D, f), allow_pickle=True)
    if d.dtype == object and d.size == 1:
        return [d.reshape(()).item()]
    return list(d)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rm = TaskEmbedResnetRewModel(checkpoint_path="/mnt/nfs/fjj/Models/RLinf-Wan-LIBERO-Goal/taskemb_resnet_rm.pth",
                                 task_suite_name="libero_goal").eval().to(dev)
    frames = []   # (file, step, instruction, label, image)
    for f in sorted(os.listdir(D)):
        for k, s in enumerate(load(f)):
            frames.append((f, k, str(s["instruction"]), float(s["reward"]), s["image"]))
    labels = np.array([fr[3] for fr in frames])
    print("frames", len(frames), "label=1:", int((labels > 0.5).sum()), flush=True)
    rep = {"n_frames": len(frames), "n_pos": int((labels > 0.5).sum()), "orientations": {}}
    for name, fn in ORIENT.items():
        probs = []
        for i in range(0, len(frames), 256):
            chunk = frames[i:i + 256]
            x = np.stack([np.ascontiguousarray(fn(c[4])) for c in chunk]).astype(np.float32) / 127.5 - 1
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
            with torch.no_grad():
                t = instruction_to_task_id([c[2] for c in chunk], "libero_goal").to(dev)
                feat = torch.cat([rm.visual_encoder(x), rm.task_embedding(t)], 1)
                probs.append(rm.fusion_layer(feat).squeeze(1).float().cpu().numpy())
        p = np.concatenate(probs)
        pred = p > 0.5
        pos = labels > 0.5
        per_task = {}
        for task in sorted({fr[2] for fr in frames}):
            m = np.array([fr[2] == task for fr in frames])
            per_task[task] = {"TP": int((pred & pos & m).sum()), "FP": int((pred & ~pos & m).sum()),
                              "FN": int((~pred & pos & m).sum()), "TN": int((~pred & ~pos & m).sum())}
        tp, fp = int((pred & pos).sum()), int((pred & ~pos).sum())
        fn_, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
        rep["orientations"][name] = {
            "TP": tp, "FP": fp, "FN": fn_, "TN": tn,
            "recall": round(tp / max(tp + fn_, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
            "pos_prob_median": round(float(np.median(p[pos])), 4) if pos.any() else None,
            "neg_prob_median": round(float(np.median(p[~pos])), 6), "per_task": per_task}
        print(name, {k: v for k, v in rep["orientations"][name].items() if k != "per_task"}, flush=True)
    # orientation of the dataset vs CoWM-PO evidence frame0 of the same task (best-matching flip, mean abs diff)
    from PIL import Image
    ev = "/data_cf/cowmpo/evidence/libero_goal_dev3"
    first = {}
    for fr in frames:
        if fr[1] == 0 and fr[0].startswith("seed_"):
            first.setdefault(fr[2], fr[4])
    cmp_ = {}
    for ep in sorted(os.listdir(ev)):
        mp = os.path.join(ev, ep, "meta.json")
        if not os.path.isfile(mp):
            continue
        lang = json.load(open(mp))["language"]
        if lang in first and lang not in cmp_:
            a = np.asarray(Image.open(os.path.join(ev, ep, "frame0.png")).convert("RGB")).astype(np.int16)
            cmp_[lang] = {n: round(float(np.abs(a - fn(first[lang]).astype(np.int16)).mean()), 1)
                          for n, fn in ORIENT.items()}
    rep["evidence_vs_dataset_mae"] = cmp_
    print("EVIDENCE_VS_DATASET", json.dumps(cmp_), flush=True)
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print("RM_CONTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
