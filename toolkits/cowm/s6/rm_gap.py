"""E: shared-RM domain gap. Raw (pre-round) RM score on target frames vs the L1 world's render of the
same episode state: first frames (30 dev episodes) and probe frames (target RGB vs L1 replay)."""
import json, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/mnt/nfs/fjj/Code/CoWMPO/toolkits")
from diffsynth.models.reward_model import TaskEmbedResnetRewModel, instruction_to_task_id
from cowm.init.estimate_state import Scene
from cowm.s6.plan_probe import apply_state

FEAT = []
rm = TaskEmbedResnetRewModel(checkpoint_path="/mnt/nfs/fjj/Models/RLinf-Wan-LIBERO-Goal/taskemb_resnet_rm.pth",
                             task_suite_name="libero_goal").eval()
def raw(imgs, instr):
    x = torch.from_numpy(np.stack(imgs).astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)
    with torch.no_grad():
        t = instruction_to_task_id([instr] * len(imgs), "libero_goal")
        v = rm.visual_encoder(x)
        f = torch.cat([v, rm.task_embedding(t)], 1)
        FEAT.append(v.numpy())
        return rm.fusion_layer[:-1](f).squeeze(1).numpy()   # pre-sigmoid logit
EV, ST, PR = "/data_cf/cowmpo/evidence/libero_goal_dev3", "/data_cf/cowmpo/worlds/s5_perception/dev3", "/data_cf/cowmpo/worlds/s6_probe/dev3"
PE = "/data_cf/cowmpo/evidence/libero_goal_dev3_probe"
sc = Scene("/data_cf/cowmpo/worlds/s4_ref_goal_v02")
tgt0, w0, tgtp, wp, l2, COS, FA = [], [], [], [], [], [], []
for ep in sorted(os.listdir(PR)):
    instr = json.load(open(f"{EV}/{ep}/meta.json"))["language"]
    st = json.load(open(f"{ST}/{ep}/state.json"))["objects"]
    pr = json.load(open(f"{PR}/{ep}/probe.json"))
    sc.reset(); apply_state(sc, st, require_matched=True)
    a = np.asarray(Image.open(f"{EV}/{ep}/frame0.png").convert("RGB")); b = sc.rgb()
    tgt0 += list(raw([a], instr)); fa = FEAT[-1][0]
    w0 += list(raw([b], instr)); fb = FEAT[-1][0]
    COS.append(float(fa @ fb / np.linalg.norm(fa) / np.linalg.norm(fb))); FA.append(fa)
    l2.append(float(np.abs(a.astype(int) - b.astype(int)).mean()))
    fr = set(pr["frames"]); ti, wi = [], []
    for k, act in enumerate(pr["actions"], start=1):
        sc.env.step(np.asarray(act))
        if k in fr and os.path.isfile(f"{PE}/{ep}__f{k:03d}/frame0.png"):
            ti.append(np.asarray(Image.open(f"{PE}/{ep}__f{k:03d}/frame0.png").convert("RGB"))); wi.append(sc.rgb())
    tgtp += list(raw(ti, instr)); wp += list(raw(wi, instr))
s = lambda v: {"logit_mean": round(float(np.mean(v)), 2), "logit_max": round(float(np.max(v)), 2), "n": len(v)}
FA = np.array(FA); FA = FA / np.linalg.norm(FA, axis=1, keepdims=True)
cross = FA @ FA.T; tt = float(cross[~np.eye(len(FA), dtype=bool)].mean())
rep = {"first_frame_target": s(tgt0), "first_frame_L1": s(w0), "probe_target": s(tgtp), "probe_L1": s(wp),
       "first_frame_rgb_mae_255": round(float(np.mean(l2)), 1),
       "rm_feat_cos_target_vs_L1_same_episode": round(float(np.mean(COS)), 3),
       "rm_feat_cos_target_vs_target_other_episode": round(tt, 3),
       "pearson_probe": round(float(np.corrcoef(tgtp, wp)[0, 1]), 3)}
json.dump(rep, open("/data_cf/cowmpo/target_eval/e_rm_gap.json", "w"), indent=1)
print("RM_GAP", rep)
