import sys
import diag_rack_contact_rgb_v31 as D
D.SETS = dict(D.SETS, far34=("rack_far_dense_plan_v34", "codex_rack_far_dense_v34"))
D.OUT = D.R / "diag_rack_all_v36"
D.run(sys.argv[1])
