"""CoWM-PO SDK: declared priors (frozen, human-written; plan §3.7A, §3.11 L-SDK).

1. MATERIALS: generic engineering ranges per material class (not fitted to any target). Physical parameters
   of objects can only come from a material class; the Agent picks the class, S6 identifies the value inside
   these bounds. density = effective density of the modelled solid (kg/m^3); friction = MuJoCo sliding
   coefficient.
2. PLATFORM: robot base height of the generic Panda + mount (measured), and the target platform's camera
   pose relative to the robot base and table-top height relative to the base (exported once by
   init/extract_camera_prior.py as a declared shared prior). The Agent cannot change either.
"""
import json
import os

MATERIALS = {
    #                     density (init, lo, hi)          sliding friction (init, lo, hi)
    "ceramic":           {"density": (2300.0, 1800.0, 2700.0), "friction": (0.70, 0.40, 1.10)},
    "glass":             {"density": (2500.0, 2300.0, 2700.0), "friction": (0.60, 0.30, 1.00)},
    "wood":              {"density": (600.0, 350.0, 950.0),    "friction": (0.70, 0.40, 1.10)},
    "metal":             {"density": (5000.0, 2600.0, 8100.0), "friction": (0.60, 0.30, 1.00)},
    "plastic":           {"density": (1100.0, 850.0, 1450.0),  "friction": (0.60, 0.30, 1.00)},
    "cardboard_package": {"density": (500.0, 150.0, 1100.0),   "friction": (0.70, 0.40, 1.10)},
    "filled_bottle":     {"density": (1000.0, 600.0, 1400.0),  "friction": (0.60, 0.30, 1.00)},
    "table_surface":     {"density": (600.0, 350.0, 950.0),    "friction": (1.00, 0.50, 1.50)},
}

MIN_REL_BOUND_WIDTH = 0.2   # a declared parameter's (hi - lo) must be >= 0.2 * |init|

BASE_Z = 0.912              # generic robosuite Panda + RethinkMount base height (measured 2026-10-28)
PRIOR_DIR = os.environ.get("COWM_PRIORS", "/data_cf/cowmpo/builder_assets/priors")


def platform_prior():
    with open(os.path.join(PRIOR_DIR, "camera_prior.json")) as f:
        cam = json.load(f)
    return {"camera": {"name": "agentview", "rel_pos": cam["rel_pos"], "rel_quat": cam["rel_quat"],
                       "fovy": cam["fovy"], "image_size": cam["image_size"], "image_flip": cam["image_flip"]},
            "table_height": BASE_Z + float(cam["table_top_z_rel_base"]),
            "source": cam["source"]}
