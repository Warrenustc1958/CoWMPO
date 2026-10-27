"""SDK self-test fixture, written by a HUMAN to exercise WorldAPI. NOT Agent output, NOT a method result.

Object categories follow the LIBERO-goal task language only (drawer, bowl, stove, plate, cream cheese,
wine bottle, rack). Positions are hand-picked collision-free guesses, not LIBERO region values.
"""


def build(w):
    w.table(size=(1.0, 1.2, 0.05), height=0.8)
    mu = w.p("obj_friction", 0.9, (0.3, 1.5))
    damp = w.p("drawer_damping", 5.0, (0.5, 50.0))
    w.add("drawer_cabinet", "cabinet", xy=(0.05, -0.30), drawer_damping=damp)
    w.add("stove", "stove", xy=(0.05, 0.30))
    w.add("rack", "rack", xy=(0.25, 0.10))
    w.add("plate", "plate", xy=(0.0, 0.04), friction=mu)
    w.add("bowl", "bowl", xy=(-0.17, -0.06), radius=w.p("bowl_radius", 0.07, (0.05, 0.09)), friction=mu)
    w.add("box", "cream_cheese", xy=(-0.16, 0.14), friction=mu)
    w.add("bottle", "wine_bottle", xy=(0.10, -0.06), friction=mu)
    w.camera(pos=(0.6, 0.0, 1.4), quat=(0.653, 0.271, 0.271, 0.653), fovy=w.p("cam_fovy", 45.0, (35.0, 60.0)))
    w.mechanism("joint_damping", "stove", 0.05, joint="knob")
