# Rack repair experiments (2026-10-30 / 31), isolated, not integrated

Isolated CoWM-PO experiments that repaired the LIBERO-goal wine rack geometry until it passed an independent
functional acceptance. Nothing here replaces `toolkits/cowm`; integration into the production SDK and the full
regression are still to do.

Result: frozen rack `rack_rib_dev_v31/cand18` passed the v32 acceptance 6/6 (rule fixed before execution:
first-contact step error <= 5, eef error <= 2 cm, >= 4 valid probes and all valid pass). Max error 2 steps / 1.45 cm.
Same positions: v29c10 4/6, v25c01 3/6, v23c03 2/6, v21c07 1/6, unrepaired v8 0/6.

## Layout

- `builder/` Builder-side scripts (run inside `toolkits/cowm/isolation/run_builder.sh`, no GT): candidate families
  (`fit_rack_*`), dense ROI-IoU scorers (`score_rack_dense_v*`), transition scorer (`score_rack_transition_v31.py`),
  acceptance planners (`plan_rack_accept_v*`), trackers (`track_rack_accept_dense_v*_v7b.py`, `--workers/--threads`
  for per-probe CPU parallelism), diagnostics (`diag_rack_*`).
- `target/` Target-side scripts (LIBERO, `.venv-oft-wan`): dense public RGB export and acceptance runs.
- `toolkits_snapshot/cowm/` the isolated v10 SDK the builder scripts import (differs from `toolkits/cowm`).
- `results/` public Builder-side outputs only: dense/transition score summaries, family and acceptance freeze files,
  diagnostics, and the frozen v31 cand18 rack (`world.json`, `rack_assets/`, `result.json`). Target-private outputs
  (`target_eval/...`, GT masks, contact rows, acceptance summaries) are intentionally not included.

## Version history

| version | change | outcome |
|---|---|---|
| v16-v21 | rear posts / bar from dense rear probes | v21c07 frozen; v22 acceptance 2/6 REJECT |
| v23 | + rear top panel | v24 acceptance 4/6 REJECT (fails on -y) |
| v25 | + sloped -y board | v26 acceptance 3/6 REJECT |
| v27/v27b | global roll about +y bottom/top edge x y shift | top = v25c01, not frozen |
| v29 | flat -y half-top plate (+3 cm) | v28 acceptance 5/6 REJECT (x+050_y-060) |
| v30 | -y edge extension / lip | ROI IoU saturated, not frozen |
| v31 | narrow -y rib + shorter rear plate, selected by transition-level signal | **v32 acceptance 6/6 ACCEPT** |

## Key lessons

- Fit the judged transition, not an appearance proxy (as in WorldCoder / GIF-MCTS style code-world-model repair).
  Acceptance judges first contact; ROI mask IoU saturates at the -y edge. `diag_rack_contact_rgb_v31.py` measures the
  deceleration onset (frame change < 0.5 x descent peak) with the same rule on public RGB and on the model render;
  on the model it lags physical first contact by 0-1 steps.
- The two acc26 y-060 probes are real slide-offs along the -y outer edge (`diag_rack_ym060_v28.json`); their ROI IoU
  is kept, their stall metric is not used.
- The v7b tracker lags arm entry into the rack ROI by 3-8 steps on every probe (common to all models).
- Data roles: every acceptance set becomes fit evidence once its Target result is seen; each new acceptance used
  positions never used before (33 used before v32).
