# Shape‑Aware Weight Merge for pi0 Fine‑Tuning (6‑Dim Actions)

Context
- After a merge, strict shape validation during checkpoint load caused a failure when fine‑tuning `pi0_clear_tray_fine_tune` (action_dim=6) from the `pi0_base` checkpoint (action_dim=32).
- Error example: `Shape mismatch at ['action_in_proj']['kernel']: expected (6, 1024), got (32, 1024)`.

Root Cause
- The base model was trained with a 32‑dim action space (multi‑embodiment). Our task uses 6 dims (single arm + gripper). Only a few head layers depend on `action_dim`.

Change Implemented
- File: `src/openpi/training/weight_loaders.py`
- Function: `_merge_params`
- Behavior: Start from the model’s reference param tree and overwrite with checkpoint weights only when shapes match; log a warning and keep the initialized param when shapes differ.
- Effect: Backbone and compatible layers load from pretraining; action‑dependent heads (e.g., `state_proj`, `action_in_proj`, `action_out_proj`) re‑initialize to 6‑dim and are learned during fine‑tuning.

Why This Is Safe
- Preserves structure and dtypes; skips only a small set of head params.
- Prevents runtime mismatch while retaining maximum benefit from pretraining.

How to Verify
- Run training:
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_clear_tray_fine_tune --exp-name=my_first_pi0jax_run --resume`
- Expect warnings like: `Skipping param action_in_proj/kernel: shape mismatch (32, 1024) != (6, 1024)` and normal progress thereafter.

Alternatives Considered
- Train from scratch (`NoOpWeightLoader`) — not desired.
- Change `action_dim` to 32 — incompatible with dataset/transforms.
- Slice/resize checkpoint heads — complex and unnecessary for this use case.

Impact
- Slight warm‑up to learn new heads; overall training proceeds and benefits from pretrained backbone.
