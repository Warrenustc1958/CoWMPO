# CoWM-PO isolated experiments, 2026-10-31 night to 2026-11-01 (not integrated)

Nothing here replaces `toolkits/cowm`. Builder scripts import the isolated v10 SDK snapshot in
`experiments/rack_repair_20261031/toolkits_snapshot/cowm`. Target-private outputs (`target_eval/...`, GT) are not included.

## rack: contact acceptance passed; S5 static localization is a known limitation
- Accepted rack: `results/rack/frozen_v48_iter5` (contact-surface fit). Visual geometry unchanged; upper-tier collision
  replaced by invisible 1 cm collision columns (group 0, rgba alpha 0; group 3 is dropped by the robosuite loader).
  Column height = 5-ray prior + residual field fitted to public deceleration onsets of 54 public probes, attached at the
  gripper (eef) local xy at first contact, 5 counterexample-guided layers with leave-one-out CV
  (`builder/fit_rack_hfield_v42/43/46/48.py`).
- Acceptance (rule fixed before execution, <=5 steps, <=2 cm, >=4 valid, all pass): v49 6/6, v50 hard regions 6/6
  (older models <=5/6, v8 2/6).
- S5 static (Target-scored, not included): original metric 3.7 cm mean (3/32, 2/28 < 3 cm); collision-only 20/32,
  18/28; before repair 0/60 at 6.1 cm. Remaining dy is visual-only GT geometry, dx is at the near end where every
  gripper probe is blocked by the forearm.

## stove knob: not passed (best 33/34)
- Public signal: burner turns red (>=20 red px in the public stove mask); no occlusion, a one-step jump.
- v51-v58: two-well detent -> hysteresis latch (`builder/latch_fixture.py`) -> short fin + joint Coulomb friction;
  v58 d58_067 10/10 in-sample, but independent acceptance v59 on 12 never-run probes REJECT 9/12.
- v60-v64 (22 probes): stuck at 21/22; episode pose (v62), knob offset (v63), axis flip (v64) do not help.
- v65: 12 new public probes changing only twist direction/magnitude (`builder/stove_probe_v65.py`); t7_i0 yaw:
  positive twist turns on, negative does not.
- v66 one-way friction 31/34; v67 twist-driven knob (fin removed, coaxial collision cylinder + knob torsional friction,
  hinge limit gives one-way) 32/34; v68 wider bounds, threshold in [s_up, on-well]: best r3_056 33/34, onset max 13
  steps (rule: all final correct and onset <= 8), only wrong t7_i0__v65_a2_m05. World in `results/stove/best_v68_*`.
- Target stove exports must use OSMesa (EGL breaks the bit-exact prefix check).
- Long hold and formal G2 (trials 30-39) not started.
