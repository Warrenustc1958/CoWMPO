"""CoWM-PO Code Agent prompts: SDK documentation + task framing. Text only; no LIBERO files."""

SYSTEM = """You are the world-building Code Agent of CoWM-PO. You write a Python world program that a frozen \
SDK compiles into an executable MuJoCo world. A Franka Panda robot (fixed, provided by the SDK) stands at the -x \
edge of the table, facing +x. Execution never calls you: the world must evolve only through physics driven by \
the robot's actions. Return ONLY one ```python code block containing the full program."""

SDK_DOC = """# CoWM-PO WorldAPI (the only interface you may use)

The program is exactly one function `def build(w):` and nothing else. No imports, no other functions, no while
loops, no lambdas, no attribute access except `w.*` and `math.*`. Allowed builtins: range len min max abs float
int round list dict tuple enumerate zip sum.

w.table(size=(L, W, thickness), height=0.8, friction=(1.0, 0.005, 0.0001))
    Table centred at the origin; top surface at z = height. Robot base is at x = -0.16 - L/2.

ref = w.p(name, init, bounds)
    Declare a physical/geometric parameter to be identified later from target evidence. Use the returned
    reference anywhere a number is expected. Declare a parameter whenever you are unsure of a value.

w.add(category, name, xy=(x, y), yaw=0.0, **attrs)
    Place one template instance on the table top. (x, y) is the object's bottom-centre in table coordinates.
    Templates and their attrs (defaults in brackets); front of fixtures faces -x (towards the robot):
      bowl            radius[0.07] height[0.05] wall[0.006] density[1500] friction[0.9]          (free)
      plate           radius[0.10] height[0.018] density[1500] friction[0.9]                     (free)
      bottle          radius[0.035] height[0.28] neck_radius[0.013] density[900] friction[0.9]   (free)
      box             size_x[0.07] size_y[0.05] size_z[0.025] density[600] friction[0.9]         (free)
      drawer_cabinet  width[0.30] depth[0.26] height[0.36] n_drawers[3] drawer_damping[5.0]
                      drawer_travel[0.16]   joints: drawer_bottom, drawer_middle, drawer_top     (fixed)
      stove           width[0.24] depth[0.22] height[0.04] knob_damping[0.05]  joint: knob       (fixed)
      rack            width[0.24] depth[0.16]  (bottle cradle)                                   (fixed)
    Footprints must not overlap (fixtures: depth along x, width along y; free objects: radius or size).
    Keep everything within the table and within ~0.85 m of the robot base.

w.camera(pos=(x, y, z), quat=(w, x, y, z), fovy=45.0)
    Third-person 'agentview' camera.

w.mechanism(kind, obj, value, joint=None)
    Constant mechanism from the library: "joint_damping" (needs joint=), "joint_frictionloss" (needs joint=),
    "geom_friction". No time-dependent or scripted behaviour exists, by design.
"""


def build_prompt(evidence, previous=None, report=None):
    u = [SDK_DOC, "\n# Evidence\n", evidence]
    if previous is not None:
        u += ["\n# Your previous program\n```python\n", previous, "\n```\n",
              "# Verifier report (fix every error; keep what already works)\n", report]
    u.append("\nWrite the complete world program now.")
    return "".join(u)
