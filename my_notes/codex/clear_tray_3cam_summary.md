# Dataset Summary: JiabinQ/clear_tray_3cam

Source: `/home/jiabin/.cache/huggingface/lerobot/JiabinQ/clear_tray_3cam`

## Observed Features (from meta/info.json)
- Actions (`action`): shape [6]
  - Names: shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos
- State (`observation.state`): shape [6]
  - Same names/order as actions
- Cameras (video):
  - `observation.images.top`   (480x640x3)
  - `observation.images.side`  (480x640x3)
  - `observation.images.wrist` (480x640x3)

## Recommended Repack Mapping (to SO100 3-cam schema)
Map dataset keys → policy-expected observation keys:
- `observation/images.top`   → `observation.images.main.left`      (base view)
- `observation/images.wrist` → `observation.images.secondary_0`    (wrist/close-up)
- `observation/images.side`  → `observation.images.secondary_1`    (aux view)
- `observation/state`        → `observation.state`
- `action`                   → `action`
- `prompt`                   → `prompt`

This aligns with `S0100Inputs` image names emitted to the model:
- `base_0_rgb`  (from main.left)
- `left_wrist_0_rgb`  (from secondary_0)
- `right_wrist_0_rgb` (from secondary_1)

## Action Dimensions and Masks
- Single arm: 5 joints + 1 gripper = 6 dims total.
- Use delta actions for joints only (not gripper):
  - Delta mask: `_transforms.make_bool_mask(5, -1)`
  - Apply `DeltaActions(mask)` on inputs and `AbsoluteActions(mask)` on outputs in the data pipeline.

## Compatibility with current `so100_policy.py`
- Inputs: OK. It expects 3 cameras (mapped above) and pads state/actions to `action_dim`.
- Outputs: It slices to 12 by default, which is safe for 6-dim data (returns 6 when present). No change strictly required for function.
- If training single-arm, prefer a data config that uses a 5-joint delta mask (e.g., a "single" variant) rather than the dual-arm 6-joint mask.

## PI0 vs PI0_FAST
- Both can be supported. Image parsing converts to uint8 HWC; `ModelTransformFactory` handles tokenization and FAST action targets.
- For FAST, you may keep all image masks true; current `S0100Inputs` sets all three masks true.

## Conclusion
- Repack mapping above + 5-joint delta mask is the key.
- No changes needed in `src/openpi/policies/so100_policy.py` to parse this dataset.
- For clarity and future-proofing, we still recommend a dedicated `so100_policy_single.py` with an explicit 6-dim output slice and any single-arm specifics, but it’s not strictly required to use this dataset.

