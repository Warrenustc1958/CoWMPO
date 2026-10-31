"""CoWM-PO Code Agent prompts: SDK documentation + task framing. No benchmark names, no target files."""

SYSTEM = """You are the world-building Code Agent of CoWM-PO. You write a Python world program that a frozen \
SDK compiles into an executable MuJoCo world. A Franka Panda robot (fixed, provided by the SDK) stands at the -x \
edge of the table, facing +x. Execution never calls you: the world must evolve only through physics driven by \
the robot's actions. Use only the evidence given (camera images and task sentences). Return ONLY one ```python \
code block containing the full program."""

SDK_DOC = """# CoWM-PO WorldAPI (SDK 0.2) -- the only interface you may use

The program is exactly one function `def build(w):` and nothing else. No imports, no other functions, no while
loops, no lambdas, no attribute access except `w.*` and `math.*`. Allowed builtins: range len min max abs float
int round list dict tuple enumerate zip sum.

Fixed by the platform (you cannot change them): robot, controller, camera ('agentview', the view in the evidence
images), table-top height. The robot base is at x = -0.16 - L/2, y = 0, about 0.01 m above the table top; it
reaches roughly 0.8 m from its base.

w.table(size=(L, W, thickness), material=m)
    Table centred at the origin.

m = w.material(name, cls)
    Declare a material handle. cls is one of: ceramic, glass, wood, metal, plastic, cardboard_package,
    filled_bottle, table_surface. density and friction of every object using this handle are ONE shared pair
    of parameters, identified later from robot trajectories within the class range. Objects made of the same
    stuff should share a handle; do not create one handle per object without a reason.

ref = w.p(name, init, bounds)
    Declare an uncertain GEOMETRY or MECHANISM value (size, wall thickness, damping, travel...). Bounds must be at
    least 20% of |init| wide: do not claim precision the evidence cannot support.

w.add(category, name, xy=(x, y), yaw=0.0, material=m, **attrs)
    Place one template instance. xy/yaw are plain numbers: a NOMINAL placement. The real placement of every
    object is re-estimated from the first camera frame of each episode, so placements are state, not parameters.
    (x, y) is the object's bottom-centre in table coordinates. Front of fixtures faces -x (towards the robot).
    Templates and attrs (defaults in brackets):
      bowl            radius[0.07] height[0.05] wall[0.006]                                     (free)
      plate           radius[0.10] height[0.018]                                                (free)
      bottle          radius[0.035] height[0.28] neck_radius[0.013]                             (free)
      box             size_x[0.07] size_y[0.05] size_z[0.025]                                   (free)
      drawer_cabinet  width[0.30] depth[0.26] height[0.36] n_drawers[3] drawer_damping[5.0]
                      drawer_travel[0.16]   joints: drawer_bottom, drawer_middle, drawer_top     (fixed)
      stove           width[0.24] depth[0.22] height[0.04] knob_damping[0.05]  joint: knob       (fixed)
      rack            width[0.24] depth[0.16]  (bottle cradle)                                   (fixed)
    Footprints must not overlap and must stay on the table; every object, and the cabinet drawer handles, must be
    reachable by the robot and visible in the agentview camera.

w.mechanism(kind, obj, value, joint=)
    Constant mechanism from the library: "joint_damping", "joint_frictionloss". No time-dependent or scripted
    behaviour exists, by design.

The verifier checks: lint, compile, footprints, settling, zero-action stillness, determinism, reachability,
articulation (drawers/knob move under a push), camera visibility. You will see its report and a render of your
world from the agentview camera if a check fails.

After those checks pass, a structural check compares your world's silhouettes with the evidence object by
object: an evidence object you did not declare (missing), an object with no evidence counterpart (unobserved),
or a template whose silhouette shape class differs from the evidence (mismatch: e.g. a round object built with
a rectangular template). Fix these by adding/removing/renaming objects or changing the template; sizes are fitted
later and need not be exact.
"""


def build_prompt(evidence, previous=None, report=None):
    u = [SDK_DOC, "\n# Evidence\n", evidence]
    if previous is not None:
        u += ["\n# Your previous program\n```python\n", previous, "\n```\n",
              "# Verifier report (fix every error; keep what already works)\n", report]
    u.append("\nWrite the complete world program now.")
    return "".join(u)
