"""CoWM-PO runtime: load a compiled world_vN/ and execute it with the shared Panda + OSC executor.

No Agent, no LLM: MuJoCo advances the state from the policy's actions (plan §3.11 L-Run).
Executor semantics validated in S2 (reset pose, OSC cache refresh, gripper integrator).
"""
import json
import os

import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask

# shared prior (plan §3.7A): target platform reset pose; also the OSC nullspace reference
RESET_QPOS = np.array([0.0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])


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

    def reward(self, action=None):
        return 0.0

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        cam = self.world.get("camera")
        if cam:
            arena.set_camera(camera_name=cam["name"], pos=cam["pos"], quat=cam["quat"],
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
        # for wide flat CoACD hulls (plate). SDK-level engine setting; LIBERO does not set it (note for S6).
        import xml.etree.ElementTree as ET

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

    def object_pos(self, name):
        obj = next(ob for ob in self.objects if ob.name == name)
        return self.sim.data.body_xpos[self.sim.model.body_name2id(obj.root_body)].copy()

    def min_contact_dist(self):
        d = self.sim.data
        return min((d.contact[i].dist for i in range(d.ncon)), default=0.0)
