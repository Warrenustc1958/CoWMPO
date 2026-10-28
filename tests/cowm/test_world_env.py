"""Runtime contracts for independent compiled worlds.

Set COWM_TEST_WORLD_DIR and COWM_TEST_OTHER_WORLD_DIR to compiled fixtures.
Select MUJOCO_GL before starting pytest; both OSMesa and EGL are supported.
"""

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def scenes():
    paths = [os.getenv("COWM_TEST_WORLD_DIR"), os.getenv("COWM_TEST_OTHER_WORLD_DIR")]
    if any(not path or not (Path(path) / "world.json").is_file() for path in paths):
        pytest.skip("Two compiled CodeWorld fixtures are required")
    pytest.importorskip("robosuite")
    from cowm.init.estimate_state import Scene

    opened = []

    def create(index: int = 0):
        scene = Scene(paths[index])
        opened.append(scene)
        scene.reset()
        return scene

    yield create
    for scene in opened:
        scene.env.close()


def test_render_after_creating_other_world(scenes) -> None:
    first = scenes()
    reference_rgb, reference_seg = first.rgb(), first.seg()
    second = scenes(1)
    second_rgb, second_seg = second.rgb(), second.seg()
    for _ in range(4):
        np.testing.assert_array_equal(first.rgb(), reference_rgb)
        np.testing.assert_array_equal(first.seg(), reference_seg)
        np.testing.assert_array_equal(second.rgb(), second_rgb)
        np.testing.assert_array_equal(second.seg(), second_seg)


def test_step_observations_match_independent_world(scenes) -> None:
    first = scenes()
    other = scenes(1)
    reference = scenes()
    action = np.array([0.2, -0.1, 0.1, 0.0, 0.0, 0.0, -1.0])
    for _ in range(8):
        expected, _, _, _ = reference.env.step(action)
        other.env.step(-action)
        observed, _, _, _ = first.env.step(action)
        np.testing.assert_array_equal(observed["agentview_image"], expected["agentview_image"])
        np.testing.assert_allclose(first.env.sim.data.qpos, reference.env.sim.data.qpos, atol=1e-12, rtol=0)


def test_closing_other_world_preserves_render(scenes) -> None:
    first = scenes()
    other = scenes(1)
    reference_rgb, reference_seg = first.rgb(), first.seg()
    other.env.close()
    np.testing.assert_array_equal(first.rgb(), reference_rgb)
    np.testing.assert_array_equal(first.seg(), reference_seg)
    other.env.close()


def test_table_material_controls_effective_contact_friction(scenes) -> None:
    from cowm.s6.g2_replay import set_friction

    scene = scenes(1)
    model, data = scene.env.sim.model, scene.env.sim.data
    for table_friction in [1.0, 0.5]:
        set_friction(scene, "table_surface", table_friction)
        for bowl_friction in [0.4, 0.7, 1.0]:
            set_friction(scene, "ceramic", bowl_friction)
            scene.reset()
            contacts = []
            for contact in data.contact[:data.ncon]:
                names = [model.geom_id2name(int(geom)) for geom in [contact.geom1, contact.geom2]]
                if "table_collision" in names and any(name and name.startswith("bowl_") for name in names):
                    contacts.append(contact.friction[0])
            assert contacts
            np.testing.assert_allclose(contacts, max(table_friction, bowl_friction), rtol=0, atol=1e-12)


def test_invalid_material_override_is_rejected(scenes) -> None:
    from cowm.s6.g2_replay import set_friction

    scene = scenes()
    original = scene.env.sim.model.geom_friction.copy()
    with pytest.raises(ValueError, match="No collision geometry"):
        set_friction(scene, "absent_material", 0.7)
    for invalid in [-1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="finite and non-negative"):
            set_friction(scene, "ceramic", invalid)
    np.testing.assert_array_equal(scene.env.sim.model.geom_friction, original)
