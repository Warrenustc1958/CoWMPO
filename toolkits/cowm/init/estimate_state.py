"""S5 InitialStateEstimate (Builder side, plan §9.7-C): per-episode object poses by render-and-compare.

Inputs (Builder-visible only): compiled world_vN/, evidence frame0.png, EvidenceAdapter masks.npz.
For each episode: reset the world + 15-step warmup (same as the policy's first frame), then for every
object search (x, y, yaw) so that the world's own segmentation render matches the SAM2 masks
(objective = sum of per-object mask IoU, so occlusions between objects are accounted for).
Init: max(IoU) of {program nominal pose, mask-centroid ray hitting the plane z = table + top_z/2}.
Pattern search with shrinking steps, two coordinate-descent sweeps; yaw is skipped for
rotationally symmetric categories. z and orientation other than yaw stay at program values.

Output: <out>/<episode>/state.json  {obj: {pos, yaw, pos_rel_base, iou, observed}}, overlay.png
No target internals are read; the target-side GT comparison is a separate script (s5_gt_dump/compare).
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
from PIL import Image

SYMMETRIC = {"bowl", "plate", "bottle"}
XY_STEPS = [0.04, 0.02, 0.01, 0.005]
YAW_STEPS = [0.4, 0.2, 0.1, 0.05]
HOLD = np.array([0, 0, 0, 0, 0, 0, -1.0])


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class Scene:
    def __init__(self, world_dir):
        from cowm.world.env import CodeWorldEnv

        self.env = CodeWorldEnv(world_dir, use_camera_obs=True, camera_names="agentview", camera_heights=256,
                                camera_widths=256, has_renderer=False, has_offscreen_renderer=True)
        self.world = self.env.world
        self.H, self.W = self.world["camera"]["image_size"]
        m = self.env.sim.model
        self.base = self.env.sim.data.body_xpos[m.body_name2id(self.env.robots[0].robot_model.root_body)].copy()
        # geom id -> object index (via the object's root body)
        root2obj = {m.body_name2id(ob.root_body): k for k, ob in enumerate(self.env.objects)}
        self.geom2obj = np.full(m.ngeom, -1, dtype=np.int64)
        for g in range(m.ngeom):
            self.geom2obj[g] = root2obj.get(int(m.body_rootid[m.geom_bodyid[g]]), -1)
        self.names = [o["name"] for o in self.world["objects"]]
        # robot geoms (shared platform prior): masks that are mostly robot are perception errors
        rroot = m.body_name2id(self.env.robots[0].robot_model.root_body)
        self.robot_geom = np.array([int(m.body_rootid[m.geom_bodyid[g]]) == rroot or
                                    (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith(
                                        ("robot0", "gripper0"))
                                    for g in range(m.ngeom)])

    def reset(self):
        obs = self.env.reset()
        for _ in range(15):
            obs, _, _, _ = self.env.step(HOLD)
        return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

    def set_pose(self, k, xy, yaw):
        o, ob = self.world["objects"][k], self.env.objects[k]
        pos = np.array([xy[0], xy[1], o["pos"][2]])
        quat = yaw_quat(yaw)
        sim = self.env.sim
        if o["free"]:
            sim.data.set_joint_qpos(ob.joints[0], np.concatenate([pos, quat]))
            sim.data.set_joint_qvel(ob.joints[0], np.zeros(6))
        else:
            bid = sim.model.body_name2id(ob.root_body)
            sim.model.body_pos[bid] = pos
            sim.model.body_quat[bid] = quat
        sim.forward()

    def _seg_raw(self):
        s = self.env.sim.render(width=self.W, height=self.H, camera_name="agentview", segmentation=True)
        gid = s[..., 1].astype(np.int64)
        gid[(s[..., 0] != 5) | (gid < 0)] = -1   # mjOBJ_GEOM only
        return gid[::-1, ::-1]

    def seg(self):
        """Object index per pixel in the policy image orientation (-1 = robot/table/background)."""
        gid = self._seg_raw()
        return np.where(gid >= 0, self.geom2obj[np.clip(gid, 0, None)], -1)

    def robot_mask(self):
        gid = self._seg_raw()
        return (gid >= 0) & self.robot_geom[np.clip(gid, 0, None)]

    def rgb(self):
        img = self.env.sim.render(width=self.W, height=self.H, camera_name="agentview")
        return np.ascontiguousarray(img[::-1, ::-1])

    def ray_plane(self, pu, pv, z):
        """Policy-image pixel -> world point on plane z (inverse of CodeWorldEnv.project)."""
        m, d = self.env.sim.model, self.env.sim.data
        cid = m.camera_name2id("agentview")
        R = d.cam_xmat[cid].reshape(3, 3)
        u, v = self.W - 1 - pu, pv
        f = 0.5 * self.H / np.tan(np.radians(m.cam_fovy[cid]) / 2)
        dirw = R @ np.array([(u - self.W / 2) / f, -(v - self.H / 2) / f, -1.0])
        o = d.cam_xpos[cid]
        t = (z - o[2]) / dirw[2]
        return o + t * dirw


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def estimate(scene, masks):
    objs = scene.world["objects"]
    # reject masks that are mostly the robot (rendered from the shared platform prior)
    rob = scene.robot_mask()
    rejected = [n for n, mk in masks.items() if mk.sum() and (mk & rob).sum() / mk.sum() > 0.5]
    masks = {n: mk for n, mk in masks.items() if n not in rejected}
    obs_idx = [k for k, o in enumerate(objs) if o["name"] in masks]
    pose = {k: [list(o["pos"][:2]), float(o.get("yaw", 0.0))] for k, o in enumerate(objs)}
    for k in pose:
        scene.set_pose(k, *pose[k])

    def score():
        s = scene.seg()
        return sum(iou(s == k, masks[objs[k]["name"]]) for k in obs_idx), s

    nominal, _ = score()
    top = scene.world["table"]["height"]
    # joint init: every observed object at its mask-centroid ray (single-object swaps get stuck when the
    # program layout is far from the evidence); keep it if it beats the nominal layout
    ray = {}
    for k in obs_idx:
        ys, xs = np.nonzero(masks[objs[k]["name"]])
        p = scene.ray_plane(xs.mean(), ys.mean(), top + objs[k]["top_z"] / 2)
        ray[k] = [float(p[0]), float(p[1])]
        scene.set_pose(k, ray[k], pose[k][1])
    best, _ = score()
    if best >= nominal:
        for k in obs_idx:
            pose[k][0] = ray[k]
    else:
        best = nominal
        for k in obs_idx:
            scene.set_pose(k, *pose[k])
    # coordinate descent, larger (by mask area) objects first
    order = sorted(obs_idx, key=lambda k: -masks[objs[k]["name"]].sum())
    n_render = 0
    for _ in range(2):
        for k in order:
            sym = objs[k]["category"] in SYMMETRIC
            for xs_, ys_ in zip(XY_STEPS, YAW_STEPS):
                improved = True
                while improved:
                    improved = False
                    moves = [(xs_, 0, 0), (-xs_, 0, 0), (0, xs_, 0), (0, -xs_, 0)]
                    if not sym:
                        moves += [(0, 0, ys_), (0, 0, -ys_)]
                    for dx, dy, dyaw in moves:
                        cand = ([pose[k][0][0] + dx, pose[k][0][1] + dy], pose[k][1] + dyaw)
                        scene.set_pose(k, *cand)
                        sc, _ = score()
                        n_render += 1
                        if sc > best + 1e-6:
                            best, pose[k] = sc, [cand[0], cand[1]]
                            improved = True
                            break
                    scene.set_pose(k, *pose[k])
    _, seg = score()
    res = {}
    for k, o in enumerate(objs):
        xy, yaw = pose[k]
        res[o["name"]] = {"pos": [xy[0], xy[1], o["pos"][2]], "yaw": yaw,
                          "pos_rel_base": [xy[0] - scene.base[0], xy[1] - scene.base[1]],
                          "observed": o["name"] in masks, "mask_rejected": o["name"] in rejected,
                          "iou": iou(seg == k, masks[o["name"]]) if o["name"] in masks else None}
    return res, best, n_render, seg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--masks", required=True, help="EvidenceAdapter output root (<episode>/masks.npz)")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default="")
    a = p.parse_args()
    import time

    scene = Scene(a.world)
    first = scene.reset()
    # orientation self-check: segmentation and RGB renders must agree with the observation pipeline
    if np.abs(scene.rgb().astype(int) - first.astype(int)).mean() > 2.0:
        raise RuntimeError("render orientation mismatch between sim.render and observation image")
    eps = a.episodes.split(",") if a.episodes else sorted(os.listdir(a.masks))
    summary = {}
    for ep in eps:
        t0 = time.time()
        scene.reset()
        m = np.load(os.path.join(a.masks, ep, "masks.npz"))
        masks = {k: m[k] for k in m.files}
        res, best, n, seg = estimate(scene, masks)
        d = os.path.join(a.out, ep)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "state.json"), "w") as fh:
            json.dump({"episode": ep, "world_sha256": scene.world["world_sha256"], "objects": res,
                       "score": best, "renders": n}, fh, indent=1)
        np.savez_compressed(os.path.join(d, "render_masks.npz"),
                            **{n: seg == k for k, n in enumerate(scene.names)})
        tgt = np.asarray(Image.open(os.path.join(a.evidence, ep, "frame0.png")).convert("RGB"))
        Image.fromarray(np.concatenate([tgt, scene.rgb()], axis=1)).save(os.path.join(d, "overlay.png"))
        summary[ep] = {k: (round(v["iou"], 3) if v["iou"] is not None else None) for k, v in res.items()}
        print(ep, f"{time.time() - t0:.0f}s renders={n}", summary[ep], flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
