# Plan: SO100 Policy + Config Implementation

## Goals
- Implement a SO100 single-arm policy: 5 joints + 1 gripper = 6 actions.
- Support 3 cameras (base + two wrists) for PI0 and PI0_FAST.
- Provide data/config plumbing to train and run inference using LeRobot datasets.

## Assumptions
- Dataset provides: images (3 views), state (6 dims), actions (T,6), optional prompt.
- Gripper is [0,1] (absolute). Joints use deltas during training; gripper remains absolute.
- No special sign/unit flips needed (Droid-style). If needed, we’ll add an Aloha-like adapter.

## Data Schema (target keys)
- Images (parsed to uint8, HWC):
  - PI0: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` (mask unused when missing)
  - PI0_FAST: `base_0_rgb`, `base_1_rgb`, `wrist_0_rgb` (no masking of padding images)
- State: `observation/state` → pad/truncate to `action_dim`.
- Actions: `actions` (T, action_dim); during training only.
- Prompt: pass-through (`prompt`).

## Policy Adapters
- Inputs: new `S0100SingleInputs` (or use existing if matching exactly)
  - Args: `action_dim: int`, `model_type: ModelType`.
  - Steps:
    1) Read and pad state to `action_dim`.
    2) Parse each camera to uint8 HWC and map names as above per model_type.
    3) Build `image` and `image_mask` dicts.
    4) If `actions` present, pad to `action_dim`.
    5) If `prompt` present, copy through.
- Outputs: new `S0100SingleOutputs`
  - Slice training targets to 6 dims: `actions[:, :6]`.
- If your runtime needs unit/sign conversions, add `adapt_to_pi` toggles (like Aloha).

## Transforms
- Delta actions (apply to joints only):
  - Mask: `_transforms.make_bool_mask(5, -1)`
  - Training-time pipeline augment:
    - Inputs: `_transforms.DeltaActions(mask)`
    - Outputs: `_transforms.AbsoluteActions(mask)`
- Normalization:
  - Prefer loading norm stats via `assets.asset_id`.
  - Otherwise, enable quantile normalization for PI0_FAST or train without if needed.

## DataConfig(s)
- New data config class: `LeRobotSO100Single3CamDataConfig` (if current ones don’t fit exactly).
  - Repack mapping (example):
    - `"observation/images.main.left"     -> observation.images.main.left`
    - `"observation/images.secondary_0"  -> observation.images.secondary_0`
    - `"observation/images.secondary_1"  -> observation.images.secondary_1`
    - `"observation/state"               -> observation.state`
    - `"actions"                         -> action`
    - `"prompt"                          -> prompt`
  - Data transforms:
    - Inputs: `S0100SingleInputs(action_dim, model_type)`
    - Outputs: `S0100SingleOutputs()`
    - Push delta/absolute pair using the 5-joint mask.
  - Model transforms: `ModelTransformFactory` (prompt tokenize, resize, FAST extraction).

## Train Configs
- PI0 (single step):
  - `model=pi0.Pi0Config(action_dim=6)`
  - `data=LeRobotSO100Single3CamDataConfig(repo_id=..., base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("action",)))`
  - Weights: `pi0_base` checkpoint.
- PI0_FAST (action horizon):
  - `model=pi0_fast.Pi0FASTConfig(action_dim=6, action_horizon=H)`
  - `data=LeRobotSO100Single3CamDataConfig(...)`
  - Weights: `pi0_fast_base` checkpoint.
  - Optional: LoRA variant + freeze filter for low-memory fine-tuning.

## Inference Bridge
- Unnormalize outputs.
- Convert deltas → absolute using current joint state (cumulative).
- Clamp to joint/gripper limits.
- Map gripper [0,1] to driver command.
- Send to robot API/driver at control rate.

## Files to Create/Edit
- `src/openpi/policies/so100_policy_single.py`
  - Implement `S0100SingleInputs` and `S0100SingleOutputs` (6-dim slice).
- `src/openpi/training/config_phospho.py`
  - Add `LeRobotSO100Single3CamDataConfig` if needed (or reuse existing singles).
  - Add new `TrainConfig` entries for PI0 and PI0_FAST variants.
- Assets (optional but recommended)
  - `assets/<asset_id>/` with norm stats for images/state/actions.

## Testing
- Unit tests:
  - Synthetic example generator matching dataset shape.
  - Verify image parsing to uint8 HWC and mask logic per model_type.
  - Verify state/actions padding and output slicing to 6 dims.
  - Check delta/absolute transforms invert correctly on random sequences.
- Dry-run training:
  - Overfit a tiny subset (e.g., 64–256 samples) to see loss → near-zero.
- Inference sanity:
  - Run a single step with real observation; validate command ranges and camera feed mapping.

## Droid vs Aloha Choice
- Default to Droid-style (simple mapping, no unit/sign flips).
- If the robot needs sign flips or nonlinear gripper transforms, add `adapt_to_pi` like Aloha.

## Step-by-Step Checklist
1) Create `so100_policy_single.py` with inputs/outputs (6 dims).
2) Add/verify data repack mapping for 3 cameras, state, actions, prompt.
3) Wire data transforms + 5-joint delta mask.
4) Add/train configs (PI0 and PI0_FAST) with correct `action_dim` and assets.
5) Implement inference postprocess (delta→absolute, limits, gripper mapping).
6) Add quick tests and perform a short overfit run.
7) Iterate on assets (norm stats) and camera mappings as needed.

