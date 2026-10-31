"""SDK self-test fixture, written by a HUMAN to exercise WorldAPI 0.2. NOT Agent output, NOT a method result.

Object categories follow the task language only (drawer, bowl, stove, plate, cream cheese, wine bottle,
rack). Placements are hand-picked nominal guesses that pass the footprint/reach checks, not target values.
"""


def build(w):
    w.table(size=(1.0, 1.2, 0.05), material=w.material("tabletop", "table_surface"))
    ceramic = w.material("ceramic", "ceramic")
    wood = w.material("wood", "wood")
    w.add("drawer_cabinet", "cabinet", xy=(0.05, -0.30), material=wood,
          drawer_damping=w.p("drawer_damping", 5.0, (0.5, 50.0)))
    w.add("stove", "stove", xy=(0.05, 0.30), material=w.material("steel", "metal"))
    w.add("rack", "rack", xy=(-0.05, 0.05), material=wood)
    w.add("plate", "plate", xy=(-0.28, 0.20), material=ceramic)
    w.add("bowl", "bowl", xy=(-0.28, -0.22), material=ceramic, radius=w.p("bowl_radius", 0.07, (0.05, 0.09)))
    w.add("box", "cream_cheese", xy=(-0.38, 0.0), material=w.material("package", "cardboard_package"))
    w.add("bottle", "wine_bottle", xy=(-0.25, -0.05), material=w.material("bottle", "filled_bottle"))
    w.mechanism("joint_damping", "stove", w.p("knob_damping", 0.05, (0.005, 0.5)), joint="knob")
