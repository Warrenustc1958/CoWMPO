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

os.environ.setdefault("MUJOCO_GL", "osmesa")   # Builder sandbox default; RLinf (L1) runs with egl
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np
from PIL import Image

SYMMETRIC = {"bowl", "plate", "bottle"}
XY_STEPS = [0.04, 0.02, 0.01, 0.005]
YAW_STEPS = [0.4, 0.2, 0.1, 0.05]
HOLD = np.array([0, 0, 0, 0, 0, 0, -1.0])
CONTACT_W = 2.0   # weight of the ground-contact term (fraction of image height between bottom rows)
PARK_X = -1.5   # off-table parking spot for unobserved objects contradicted by the evidence


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

    def set_pose(self, k, xy, yaw, z=None):
        o, ob = self.world["objects"][k], self.env.objects[k]
        pos = np.array([xy[0], xy[1], o["pos"][2] if z is None else z], dtype=float)
        if not np.all(np.isfinite(pos)) or not np.isfinite(yaw):
            raise ValueError("object pose must contain finite xyz and yaw")
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
        s = self.env.render_camera_frame(width=self.W, height=self.H, camera_name="agentview", segmentation=True)
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
        img = self.env.render_camera_frame(width=self.W, height=self.H, camera_name="agentview")
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


def estimate(scene, masks, association=None, allow_legacy=False):
    from cowm.init.observation import select_masks

    objs = scene.world["objects"]
    masks, identity = select_masks([o["name"] for o in objs], masks, association, allow_legacy)
    # reject masks that are mostly the robot (rendered from the shared platform prior)
    rob = scene.robot_mask()
    rejected = [n for n, mk in masks.items() if mk.sum() and (mk & rob).sum() / mk.sum() > 0.5]
    masks = {n: mk for n, mk in masks.items() if n not in rejected}
    obs_idx = [k for k, o in enumerate(objs) if o["name"] in masks]
    pose = {k: [list(o["pos"][:2]), float(o.get("yaw", 0.0))] for k, o in enumerate(objs)}
    for k in pose:
        scene.set_pose(k, *pose[k])

    # ground contact: objects rest on the table, so the lowest mask row (closest to the camera) is where the
    # object meets the table. A silhouette alone cannot separate depth from height; this term pins depth.
    mask_bottom = {k: float(np.nonzero(masks[objs[k]["name"]])[0].max()) for k in obs_idx}

    def score():
        s = scene.seg()
        tot = 0.0
        for k in obs_idx:
            mk = masks[objs[k]["name"]]
            rows = np.nonzero(s == k)[0]
            contact = abs(rows.max() - mask_bottom[k]) / scene.H if len(rows) else 1.0
            tot += iou(s == k, mk) - CONTACT_W * contact
        return tot, s

    def apply(layout):
        for k in obs_idx:
            scene.set_pose(k, layout[k], pose[k][1])

    top = scene.world["table"]["height"]
    m_, d_ = scene.env.sim.model, scene.env.sim.data
    cam = d_.cam_xpos[m_.camera_name2id("agentview")].copy()
    from cowm.init.aabb import object_aabb

    # candidate joint layouts (single-object swaps get stuck when the program layout is far from evidence)
    layouts = {"nominal": {k: list(pose[k][0]) for k in obs_idx}, "centroid": {}, "contact": {}}
    for k in obs_idx:
        ys, xs = np.nonzero(masks[objs[k]["name"]])
        p = scene.ray_plane(xs.mean(), ys.mean(), top + objs[k]["top_z"] / 2)
        layouts["centroid"][k] = [float(p[0]), float(p[1])]
        # near edge of the footprint: ray through the bottom mask rows hits the table plane
        sel = ys >= ys.max() - 1
        q = scene.ray_plane(xs[sel].mean(), ys.max(), top)
        dxy = q[:2] - cam[:2]
        dxy /= max(np.linalg.norm(dxy), 1e-9)
        scene.set_pose(k, q[:2], pose[k][1])
        c, h = object_aabb(m_, d_, np.nonzero(scene.geom2obj == k)[0])
        origin = scene.env.sim.data.body_xpos[m_.body_name2id(scene.env.objects[k].root_body)][:2]
        r = abs(dxy[0]) * h[0] + abs(dxy[1]) * h[1]
        layouts["contact"][k] = [float(v) for v in (q[:2] + dxy * r - (c[:2] - origin))]
    scores = {}
    for name, lay in layouts.items():
        apply(lay)
        scores[name], _ = score()
    pick = max(scores, key=scores.get)
    for k in obs_idx:
        pose[k][0] = layouts[pick][k]
    apply(layouts[pick])
    best = scores[pick]
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
    # unobserved objects keep the program's nominal pose unless that pose penetrates an observed object;
    # then the evidence contradicts it and it is parked off the table (flagged, never silently kept)
    parked = []
    sim = scene.env.sim
    for k, o in enumerate(objs):
        if k in obs_idx:
            continue
        gk = set(np.nonzero(scene.geom2obj == k)[0].tolist())
        go = set(np.nonzero(np.isin(scene.geom2obj, obs_idx))[0].tolist())
        hit = any(c.dist < -0.002 and ((c.geom1 in gk and c.geom2 in go) or (c.geom2 in gk and c.geom1 in go))
                  for c in sim.data.contact[:sim.data.ncon])
        if hit:
            pose[k][0] = [PARK_X, 0.6 + 0.3 * len(parked)]
            scene.set_pose(k, *pose[k])
            parked.append(o["name"])
    _, seg = score()
    from cowm.init.aabb import object_aabb

    m, d = scene.env.sim.model, scene.env.sim.data
    res = {}
    for k, o in enumerate(objs):
        xy, yaw = pose[k]
        c, h = object_aabb(m, d, np.nonzero(scene.geom2obj == k)[0])
        res[o["name"]] = {"pos": [xy[0], xy[1], o["pos"][2]], "yaw": yaw,
                          "pos_rel_base": [xy[0] - scene.base[0], xy[1] - scene.base[1]],
                          "observed": o["name"] in masks, "mask_rejected": o["name"] in rejected,
                          "association_status": identity[o["name"]],
                          "parked": o["name"] in parked,
                          "aabb_center_rel_base": (c - scene.base).tolist() if c is not None else None,
                          "aabb_half": h.tolist() if h is not None else None,
                          "iou": iou(seg == k, masks[o["name"]]) if o["name"] in masks else None}
    return res, best, n_render, seg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--masks", required=True, help="EvidenceAdapter output root (<episode>/masks.npz)")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default="")
    p.add_argument("--allow-legacy-masks", action="store_true",
                   help="diagnostic only: fit masks without association metadata; state stays unverified")
    a = p.parse_args()
    import time
    from cowm.world.artifact import tree_sha, verify_tree

    world_hash = tree_sha(a.world)
    scene = Scene(a.world)
    first = scene.reset()
    # orientation self-check: segmentation and RGB renders must agree with the observation pipeline
    if np.abs(scene.rgb().astype(int) - first.astype(int)).mean() > 2.0:
        raise RuntimeError("render orientation mismatch between sim.render and observation image")
    eps = a.episodes.split(",") if a.episodes else sorted(
        e for e in os.listdir(a.masks) if os.path.isfile(os.path.join(a.masks, e, "masks.npz")))
    summary = {}
    for ep in eps:
        t0 = time.time()
        scene.reset()
        m = np.load(os.path.join(a.masks, ep, "masks.npz"))
        masks = {k: m[k] for k in m.files}
        from cowm.init.observation import load_association
        association = load_association(os.path.join(a.masks, ep), ep)
        res, best, n, seg = estimate(scene, masks, association, a.allow_legacy_masks)
        d = os.path.join(a.out, ep)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "state.json"), "w") as fh:
            json.dump({"episode": ep, "world_sha256": scene.world["world_sha256"], "objects": res,
                       "world_tree_sha256": world_hash,
                       "score": best, "renders": n}, fh, indent=1)
        np.savez_compressed(os.path.join(d, "render_masks.npz"),
                            **{n: seg == k for k, n in enumerate(scene.names)})
        tgt = np.asarray(Image.open(os.path.join(a.evidence, ep, "frame0.png")).convert("RGB"))
        Image.fromarray(np.concatenate([tgt, scene.rgb()], axis=1)).save(os.path.join(d, "overlay.png"))
        summary[ep] = {k: (round(v["iou"], 3) if v["iou"] is not None else None) for k, v in res.items()}
        print(ep, f"{time.time() - t0:.0f}s renders={n}", summary[ep], flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    verify_tree(a.world, world_hash)
    scene.env.close()


if __name__ == "__main__":
    main()
