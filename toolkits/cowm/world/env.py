"""CoWM-PO runtime: load a compiled world_vN/ and execute it with the shared Panda + OSC executor.

No Agent, no LLM: MuJoCo advances the state from the policy's actions (plan §3.11 L-Run).
Executor semantics validated in S2 (reset pose, OSC cache refresh, gripper integrator).
Camera = declared platform prior, placed relative to the robot base.
"""
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

    def snapshot(self):
        """Everything that influences future steps: MuJoCo data (incl. solver warm start), fixture poses set
        at runtime (model.body_pos/quat), OSC controller goal state, gripper integrator, env step counter,
        world version. Restoring and replaying the same actions reproduces the same states bitwise."""
        d, m = self.sim.data, self.sim.model
        r = self.robots[0]
        snap = {"world_sha256": self.world["world_sha256"], "time": float(d.time),
                "timestep": int(self.timestep),
                "data": {f: np.array(getattr(d, f)).copy() for f in self._DATA_FIELDS},
                "body_pos": np.array(m.body_pos).copy(), "body_quat": np.array(m.body_quat).copy(),
                "geom_friction": np.array(m.geom_friction).copy(),
                "ctrl": {}, "gripper_action": np.array(r.gripper.current_action).copy()}
        for f in self._CTRL_FIELDS:
            v = getattr(r.controller, f, None)
            snap["ctrl"][f] = None if v is None else (np.array(v).copy() if not isinstance(v, bool) else v)
        return snap

    def restore(self, snap):
        if snap["world_sha256"] != self.world["world_sha256"]:
            raise ValueError("snapshot belongs to a different world version")
        d, m = self.sim.data, self.sim.model
        r = self.robots[0]
        m.body_pos[:] = snap["body_pos"]
        m.body_quat[:] = snap["body_quat"]
        m.geom_friction[:] = snap["geom_friction"]
        for f, v in snap["data"].items():
            getattr(d, f)[:] = v
        d.time = snap["time"]
        self.timestep = snap["timestep"]
        self.sim.forward()
        for f, v in snap["ctrl"].items():
            setattr(r.controller, f, None if v is None else (v.copy() if isinstance(v, np.ndarray) else v))
        r.gripper.current_action = snap["gripper_action"].copy()
        r.controller.update(force=True)

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
