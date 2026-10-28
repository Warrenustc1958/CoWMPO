# Structurally wrong seed (reference placements, so it passes the verifier gates): the plate is built with the
# 'box' template and the cream cheese is missing. Used only to test the Agent's structural-repair loop.
def build(w):
    w.table(size=(1.0, 1.2, 0.05), material=w.material("tabletop", "table_surface"))
    wood = w.material("wood", "wood")
    ceramic = w.material("ceramic", "ceramic")
    w.add("drawer_cabinet", "cabinet", xy=(0.05, -0.30), material=wood)
    w.add("stove", "stove", xy=(0.05, 0.30), material=w.material("steel", "metal"))
    w.add("rack", "rack", xy=(-0.05, 0.05), material=wood)
    w.add("box", "plate", xy=(-0.28, 0.20), material=ceramic, size_x=0.2, size_y=0.2, size_z=0.018)
    w.add("bowl", "bowl", xy=(-0.28, -0.22), material=ceramic)
    w.add("bottle", "wine_bottle", xy=(-0.25, -0.05), material=w.material("bottle", "filled_bottle"))
