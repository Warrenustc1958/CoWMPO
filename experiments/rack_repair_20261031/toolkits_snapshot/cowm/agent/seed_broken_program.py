"""Deliberately broken seed (forces the repair loop): the stove overlaps the cabinet, the bottle is out of reach.
Used only to test that the Agent repairs a world from verifier reports."""


def build(w):
    w.table(size=(1.0, 1.2, 0.05), material=w.material("tabletop", "table_surface"))
    wood = w.material("wood", "wood")
    ceramic = w.material("ceramic", "ceramic")
    w.add("drawer_cabinet", "cabinet", xy=(0.05, -0.30), material=wood)
    w.add("stove", "stove", xy=(0.10, -0.22), material=w.material("steel", "metal"))
    w.add("rack", "rack", xy=(0.25, 0.10), material=wood)
    w.add("plate", "plate", xy=(0.0, 0.04), material=ceramic)
    w.add("bowl", "bowl", xy=(-0.17, -0.06), material=ceramic)
    w.add("box", "cream_cheese", xy=(-0.16, 0.14), material=w.material("package", "cardboard_package"))
    w.add("bottle", "wine_bottle", xy=(0.46, 0.50), material=w.material("bottle", "filled_bottle"))
