# Key Insights from Phospho's SO-100 Implementation

## Overview

Phospho has implemented SO-100 support with two variants:
1. **Dual-arm SO-100** (`so100_policy.py`) - 12 DOF (2 robots with 6 DOF each)
2. **Single-arm SO-100** (`so100_policy_single.py`) - 6 DOF (5 joints + 1 gripper)

## Key Implementation Details

### 1. State and Action Dimensions

**Single-arm SO-100:**
- State dimension: 6 (5 joint positions + 1 gripper position)
- Action dimension: 6 (matching state dimension)
- Joint configuration: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`

**Dual-arm SO-100:**
- State dimension: 12 (2x single-arm)
- Action dimension: 12

### 2. Camera Configuration

The implementation expects specific camera naming:
- `observation/images.main.left` - Main/base camera
- `observation/images.secondary_0` - First wrist camera (or right wrist for dual)
- `observation/images.secondary_1` - Second wrist camera (left wrist for dual, not used in single)

Camera images are mapped to model inputs as:
- `base_0_rgb` - Base camera view
- `left_wrist_0_rgb` - Left wrist camera
- `right_wrist_0_rgb` - Right wrist camera (dual-arm only)

### 3. Image Processing

```python
def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image
```

Key points:
- Handles both float32 and uint8 inputs
- Converts from LeRobot's (C,H,W) format to model's expected (H,W,C)
- Ensures uint8 output for the model

### 4. State Padding Strategy

The implementation uses intelligent padding based on model type:
- **Pi0**: Pads state to match action dimension
- **Pi0-FAST**: No padding needed (uses exact dimensions)

### 5. Dataset Configuration

Phospho uses the `LegrandFrederic/dual-setup` dataset from HuggingFace, which appears to be a dual SO-100 setup dataset.

### 6. Training Configurations

Three main configurations:
1. `pi0_so100` - Standard Pi0 for dual-arm
2. `pi0_fast_so100` - Pi0-FAST for dual-arm (action_dim=12, horizon=10)
3. `pi0_so100_single` - Single-arm variant

## Key Differences from Our Previous Approach

1. **Camera Naming**: Phospho uses a more structured naming convention with main/secondary cameras
2. **Dual-arm Support**: Phospho's implementation supports both single and dual SO-100 setups
3. **Image Format Handling**: Robust handling of LeRobot's image format conversion
4. **Model-specific Padding**: Different padding strategies for Pi0 vs Pi0-FAST

## Recommendations for OpenPi SO-100 Implementation

Based on Phospho's approach, we should:

1. **Support Both Variants**: Implement both single-arm (6 DOF) and dual-arm (12 DOF) configurations

2. **Use Consistent Camera Naming**:
   ```python
   # Single-arm
   cameras = {
       "main.left": "base_camera",
       "secondary_0": "wrist_camera"
   }
   
   # Dual-arm
   cameras = {
       "main.left": "base_camera",
       "secondary_0": "right_wrist_camera",
       "secondary_1": "left_wrist_camera"
   }
   ```

3. **Handle Image Format Conversion**: Ensure proper conversion between LeRobot and model formats

4. **Action Chunking**: Use 10-step chunks for Pi0-FAST (vs 16 in GR00T)

5. **Dataset Structure**: Follow LeRobot's naming conventions for compatibility

## Example Usage Pattern

```python
# Create example input
example = {
    "observation/state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # 6 DOF
    "observation/images.main.left": front_camera_image,  # (224, 224, 3) uint8
    "observation/images.secondary_0": wrist_camera_image,  # (224, 224, 3) uint8
    "prompt": "Pick up the red block"
}

# Transform for model
inputs = S0100SingleInputs(action_dim=6, model_type=ModelType.PI0_FAST)
model_input = inputs(example)

# Get actions
outputs = S0100SingleOutputs()
actions = outputs(model_output)  # Returns 6 DOF actions
```

This implementation provides a clean, modular approach that can be easily adapted for different SO-100 configurations.