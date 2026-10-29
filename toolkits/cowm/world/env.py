"""CoWM-PO runtime: load a compiled world_vN/ and execute it with the shared Panda + OSC executor.

No Agent, no LLM: MuJoCo advances the state from the policy's actions (plan §3.11 L-Run).
Executor semantics validated in S2 (reset pose, OSC cache refresh, gripper integrator).
Camera = declared platform prior, placed relative to the robot base.
"""
import copy
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask

from .priors import BASE_Z

# shared prior (plan §3.7A): target platform reset pose; also the OSC nullspace reference
RESET_QPOS = np.array([0.0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


class CodeWorldEnv(SingleArmEnv):
    def __init__(self, world_dir, robots="Panda", controller_configs=None, **kwargs):
        self.world_dir = world_dir
        with open(os.path.join(world_dir, "world.json")) as f:
            self.world = json.load(f)
        t = self.world["table"]
        self.table_full_size = tuple(t["size"])
        self.table_friction = tuple(t["friction"])
        self.table_offset = np.array([0.0, 0.0, float(t["height"])])
        if controller_configs is None:
            import robosuite as suite
            controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
        kwargs.setdefault("initialization_noise", None)
        kwargs.setdefault("control_freq", 20)
        kwargs.setdefault("hard_reset", False)
        super().__init__(robots=robots, controller_configs=controller_configs, **kwargs)
        self.robots[0].init_qpos = RESET_QPOS.copy()
        self.reset()
        bz = self.sim.data.body_xpos[self.sim.model.body_name2id(self.robots[0].robot_model.root_body)][2]
        if abs(bz - BASE_Z) > 1e-3:
            raise RuntimeError(f"robot base z {bz:.4f} != prior BASE_Z {BASE_Z}: camera prior would be misplaced")

    def reward(self, action=None):
        return 0.0

    # ---- render-context ownership (audit 10.28): robosuite does not switch GL contexts by itself, so with
    # several worlds in one process every render/step/reset/close must bind THIS world's context first ----
    def _make_render_context_current(self):
        ctx = getattr(getattr(self, "sim", None), "_render_context_offscreen", None)
        if ctx is not None:
            ctx.gl_ctx.make_current()

    def _get_observations(self, force_update=False):
        self._make_render_context_current()
        return super()._get_observations(force_update=force_update)

    def step(self, action):
        self._make_render_context_current()
        return super().step(action)

    def reset(self):
        self._make_render_context_current()
        return super().reset()

    def _destroy_sim(self):
        self._make_render_context_current()
        super()._destroy_sim()

    def render_camera_frame(self, width, height, camera_name="agentview", segmentation=False):
        """Render this world's camera in its own offscreen context."""
        self._make_render_context_current()
        return self.sim.render(width=width, height=height, camera_name=camera_name, segmentation=segmentation)

    def _load_model(self):
        super()._load_model()
        base_x = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(base_x)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        cam = self.world.get("camera")
        if cam:
            base = np.array([base_x[0], base_x[1], BASE_Z])   # base orientation is identity (measured)
            pos = (base + np.array(cam["rel_pos"])).tolist()
            arena.set_camera(camera_name=cam["name"], pos=pos, quat=list(cam["rel_quat"]),
                             camera_attribs={"fovy": str(cam["fovy"])})
        self.objects = []
        for o in self.world["objects"]:
            obj = MujocoXMLObject(os.path.join(self.world_dir, o["xml"]), name=o["name"],
                                  joints=[dict(type="free", damping="0.0005")] if o["free"] else None,
                                  obj_type="all", duplicate_collision_geoms=False)
            if not o["free"]:
                root = obj.get_obj()
                root.set("pos", " ".join(f"{v:.5f}" for v in o["pos"]))
                root.set("quat", " ".join(f"{v:.6f}" for v in o["quat"]))
            self.objects.append(obj)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=self.objects)
        # Multi-contact for convex meshes: the default single-contact path reports ~4.5 mm false penetration
        # for wide flat CoACD hulls (plate). SDK-level engine setting; the target does not set it (note for S6).
        root = self.model.root
        option = root.find("option")
        if option is None:
            option = ET.Element("option")
            root.insert(0, option)
        flag = option.find("flag")
        if flag is None:
            flag = ET.SubElement(option, "flag")
        flag.set("multiccd", "enable")
        self._table_pairs = []   # [(object index, object geom name)] in <pair> declaration order
        if self.world["table"].get("contact_rule", "legacy") == "object_table_pair":
            self._add_object_table_pairs(root)

    def _add_object_table_pairs(self, root):
        """Opt-in contact rule (SDK 0.2.2): object-table contacts become explicit <pair>s whose friction is the
        object's own material value. Dynamic object-table contacts are excluded body by body; every other
        contact (robot-table, fixture-table, object-object, gripper-object) keeps MuJoCo's default mixing and
        no geom priority changes. Pair friction 5-vector = [mu, mu, max spin, max roll, max roll], condim =
        max of the two geoms, solref/solimp = MuJoCo defaults (what the default mixing gives for these geoms)."""
        table_geom = next((g for g in root.iter("geom") if (g.get("name") or "").startswith("table_collision")), None)
        if table_geom is None:
            raise RuntimeError("contact_rule object_table_pair: table collision geom not found")
        parent = {c: p for p in root.iter() for c in p}
        table_body = parent[table_geom]
        tf = [float(v) for v in (table_geom.get("friction") or "1 0.005 0.0001").split()]
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for k, (ob, o) in enumerate(zip(self.objects, self.world["objects"])):
            if not o["free"]:
                continue
            bodies = set()
            for i, g in enumerate(ob.get_obj().iter("geom")):
                if g.get("group", "0") != "0" or g.get("contype") == "0":
                    continue
                if not g.get("name"):
                    g.set("name", f"{o['name']}_col{i}")
                of = [float(v) for v in (g.get("friction") or "1 0.005 0.0001").split()]
                cd = max(int(g.get("condim", "3")), int(table_geom.get("condim", "3")))
                contact.append(ET.Element("pair", {
                    "geom1": table_geom.get("name"), "geom2": g.get("name"), "condim": str(cd),
                    "friction": f"{of[0]:.5f} {of[0]:.5f} {max(of[1], tf[1]):.5f} {max(of[2], tf[2]):.6f} "
                                f"{max(of[2], tf[2]):.6f}"}))
                self._table_pairs.append((k, g.get("name")))
                bodies.add(parent[g].get("name"))
            for b in sorted(bodies):
                contact.append(ET.Element("exclude", {"body1": table_body.get("name"), "body2": b}))

    def _reset_internal(self):
        super()._reset_internal()
        for obj, o in zip(self.objects, self.world["objects"]):
            if o["free"]:
                self.sim.data.set_joint_qpos(obj.joints[0], np.array(o["pos"] + o["quat"]))
        self.sim.forward()

    # ---- snapshot / restore (KIR, branching rollouts; review item) ----
    _DATA_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
                    "mocap_pos", "mocap_quat")
    _CTRL_FIELDS = ("goal_pos", "goal_ori", "relative_ori", "ori_ref", "initial_joint", "initial_ee_pos",
                    "initial_ee_ori_mat", "torques", "new_update")
    _OBS_FIELDS = ("_time_since_last_sample", "_current_delay", "_current_observed_value", "_sampled")

    def snapshot(self):
        """Everything that influences future steps: MuJoCo data (incl. solver warm start), fixture poses set
        at runtime (model.body_pos/quat), OSC controller goal state, gripper integrator, env step counter,
        world version. Restoring and replaying the same actions reproduces the same states bitwise."""
        d, m = self.sim.data, self.sim.model
        r = self.robots[0]
        snap = {"world_sha256": self.world["world_sha256"], "time": float(d.time),
                "timestep": int(self.timestep), "cur_time": self.cur_time, "done": bool(self.done),
                "data": {f: np.array(getattr(d, f)).copy() for f in self._DATA_FIELDS},
                "body_pos": np.array(m.body_pos).copy(), "body_quat": np.array(m.body_quat).copy(),
                # runtime-mutable model fields (S6 apply_params edits friction / mass / inertia)
                "geom_friction": np.array(m.geom_friction).copy(), "body_mass": np.array(m.body_mass).copy(),
                "body_inertia": np.array(m.body_inertia).copy(), "pair_friction": np.array(m.pair_friction).copy(),
                "ctrl": {}, "gripper_action": np.array(r.gripper.current_action).copy()}
        for f in self._CTRL_FIELDS:
            v = getattr(r.controller, f, None)
            snap["ctrl"][f] = None if v is None else (np.array(v).copy() if not isinstance(v, bool) else v)
        # robosuite observables keep their own sampling clock, delay and cached value; the policy reads these,
        # so they are state too (default corrupter/filter/delayer are stateless)
        snap["observables"] = {name: {f: copy.deepcopy(getattr(ob, f)) for f in self._OBS_FIELDS}
                               for name, ob in self._observables.items()}
        snap["obs_cache"] = copy.deepcopy(self._obs_cache)
        return snap

    def restore(self, snap):
        if snap["world_sha256"] != self.world["world_sha256"]:
            raise ValueError("snapshot belongs to a different world version")
        d, m = self.sim.data, self.sim.model
        r = self.robots[0]
        m.body_pos[:] = snap["body_pos"]
        m.body_quat[:] = snap["body_quat"]
        m.geom_friction[:] = snap["geom_friction"]
        m.body_mass[:] = snap["body_mass"]
        m.body_inertia[:] = snap["body_inertia"]
        m.pair_friction[:] = snap["pair_friction"]
        for f, v in snap["data"].items():
            getattr(d, f)[:] = v
        d.time = snap["time"]
        self.timestep = snap["timestep"]
        # robosuite refuses step() while done=True: restoring a pre-terminal snapshot must clear it
        self.cur_time, self.done = snap["cur_time"], snap["done"]
        self.sim.forward()
        for f, v in snap["ctrl"].items():
            setattr(r.controller, f, None if v is None else (v.copy() if isinstance(v, np.ndarray) else v))
        r.gripper.current_action = snap["gripper_action"].copy()
        r.controller.update(force=True)
        # restore observable clocks/caches verbatim (a forced re-sample here would itself change their state)
        for name, st in snap["observables"].items():
            ob = self._observables[name]
            for f, v in st.items():
                setattr(ob, f, copy.deepcopy(v))
        self._obs_cache = copy.deepcopy(snap["obs_cache"])

    def obj(self, name):
        return next(ob for ob in self.objects if ob.name == name)

    def object_pos(self, name):
        return self.sim.data.body_xpos[self.sim.model.body_name2id(self.obj(name).root_body)].copy()

    def eef_pos(self):
        return self._get_observations(force_update=True)["robot0_eef_pos"].copy()

    def min_contact_dist(self):
        d = self.sim.data
        return min((d.contact[i].dist for i in range(d.ncon)), default=0.0)

    def project(self, p, cam="agentview"):
        """World point -> (u, v) pixel in the policy image (after the 180 deg flip), depth along view axis."""
        m, d = self.sim.model, self.sim.data
        cid = m.camera_name2id(cam)
        R = d.cam_xmat[cid].reshape(3, 3)
        pc = R.T @ (np.asarray(p) - d.cam_xpos[cid])       # camera frame: x right, y up, -z forward
        depth = -pc[2]
        H, W = self.world["camera"]["image_size"]
        f = 0.5 * H / np.tan(np.radians(m.cam_fovy[cid]) / 2)
        u = W / 2 + f * pc[0] / max(depth, 1e-6)
        v = H / 2 - f * pc[1] / max(depth, 1e-6)
        # The raw MuJoCo render is bottom-up (OpenGL); the policy image is raw[::-1, ::-1], which in top-down
        # pixel coordinates is a horizontal mirror only (verified against segmentation-render centroids, S5).
        return W - 1 - u, v, depth
