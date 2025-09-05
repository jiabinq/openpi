# Review: src/openpi/policies/so100_policy.py (vs. SO100 plan)

## What it currently does
- Target: Dual-arm SO100 style.
- State/actions: Example uses 12-dim state; outputs slice to 12 (`actions[:, :12]`).
- Cameras: 3 inputs mapped to `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`.
- Model type: Accepts `model_type` but uses it only to set an unused variable (`mask_padding`).
- Training: Pads `state`/`actions` to `action_dim`; passes `prompt` through.

## Alignment with our plan
- Our plan targets single-arm (5 joints + 1 gripper = 6 actions) with 3 cameras.
- This file is suitable for dual-arm training/configs (e.g., `pi0_fast_so100` with `action_dim=12`).
- For single-arm, we plan a separate adapter (`so100_policy_single.py`) that slices to 6 and uses a 5-joint delta mask.

## Must-change items (only if reusing this file for single-arm)
- Change output slicing from 12 to 6 in `S0100Outputs`.
- Update the example and any comments to reflect 6-dim state.
- Optionally consider model_type-specific image naming (like `DroidInputs`) if your FAST model requires it.

If we keep this file for dual-arm and implement a separate single-arm policy, no changes are required.

## Optional cleanups (not required to function)
- Remove or use `mask_padding` (currently unused).
- Fix docstring in `make_so100_example` (mentions Libero).
- Add a small guard/log for missing image keys (mirrors other policies).

## Recommendation
- Do not modify `src/openpi/policies/so100_policy.py`.
- Add a new `so100_policy_single.py` for 6-dim single-arm, per plan.

