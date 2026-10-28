"""SDK self-test fixture v03, written by a HUMAN. NOT Agent output, NOT a method result.

Same categories/layout as the v02 fixture, but every object's size is a declared parameter with a generic
category range (not tuned to the target), so S5/S6 can identify shapes from evidence. Placements are
nominal; per-episode poses come from S5.
"""


def build(w):
    w.table(size=(1.0, 1.2, 0.05), material=w.material("tabletop", "table_surface"))
    ceramic = w.material("ceramic", "ceramic")
    wood = w.material("wood", "wood")
    w.add("drawer_cabinet", "cabinet", xy=(0.05, -0.30), material=wood,
          drawer_damping=w.p("drawer_damping", 5.0, (0.5, 50.0)),
          width=w.p("cabinet_width", 0.30, (0.15, 0.50)),
          depth=w.p("cabinet_depth", 0.26, (0.13, 0.45)),
          height=w.p("cabinet_height", 0.36, (0.15, 0.60)))
    w.add("stove", "stove", xy=(0.05, 0.30), material=w.material("steel", "metal"),
          width=w.p("stove_width", 0.24, (0.12, 0.45)),
          depth=w.p("stove_depth", 0.22, (0.11, 0.40)),
          height=w.p("stove_height", 0.04, (0.01, 0.12)))
    w.add("rack", "rack", xy=(-0.05, 0.05), material=wood,
          width=w.p("rack_width", 0.24, (0.10, 0.60)),
          depth=w.p("rack_depth", 0.16, (0.06, 0.40)),
          height=w.p("rack_height", 0.063, (0.03, 0.50)))
    w.add("plate", "plate", xy=(-0.28, 0.20), material=ceramic, radius=w.p("plate_radius", 0.10, (0.04, 0.16)))
    w.add("bowl", "bowl", xy=(-0.28, -0.22), material=ceramic,
          radius=w.p("bowl_radius", 0.07, (0.04, 0.10)),
          height=w.p("bowl_height", 0.05, (0.025, 0.09)))
    w.add("box", "cream_cheese", xy=(-0.38, 0.0), material=w.material("package", "cardboard_package"),
          size_x=w.p("box_size_x", 0.07, (0.03, 0.14)),
          size_y=w.p("box_size_y", 0.05, (0.025, 0.12)),
          size_z=w.p("box_size_z", 0.025, (0.012, 0.08)))
    w.add("bottle", "wine_bottle", xy=(-0.25, -0.05), material=w.material("bottle", "filled_bottle"),
          radius=w.p("bottle_radius", 0.035, (0.012, 0.06)),
          height=w.p("bottle_height", 0.28, (0.08, 0.40)))
    w.mechanism("joint_damping", "stove", w.p("knob_damping", 0.05, (0.005, 0.5)), joint="knob")
