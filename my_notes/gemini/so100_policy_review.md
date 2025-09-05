# Review of `so100_policy.py`

This document summarizes the necessary changes for the existing file `src/openpi/policies/so100_policy.py` to align with our implementation plan for a **single-arm SO-100 robot** (5 joints, 1 gripper, 3 cameras).

### Overall Assessment
The existing file is a good starting template but is configured for a **dual-arm robot (12 actions)**. The changes below are mandatory to make it work for your **single-arm robot (6 actions)**.

--- 

## Mandatory Changes

Here are the parts of the code that have to be changed.

### 1. State Handling in `S0100Inputs`

**Problem:** The current code assumes the dataset provides a single, pre-built state vector in `data["observation/state"]`. Our plan, which is more robust, is to build the state from separate joint and gripper position keys. This is the most critical change.

**Action:** Replace the current state handling logic.

**Current Code:**
```python
# For the SO100 the state is of size 6 so we pad
state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
```

**Required Change:**
```python
# --- REQUIRED CHANGE ---
# Construct the state from separate joint and gripper keys.
# IMPORTANT: You must verify these keys match your dataset!
state = np.concatenate([
    data["observation/joint_position"], # Should be an array of 5 numbers
    data["observation/gripper_position"] # Should be an array of 1 number
])
# The state is now correctly sized to 6, so padding to action_dim=6 will not change it.
state = transforms.pad_to_dim(state, self.action_dim)
```

### 2. Image Keys in `S0100Inputs`

**Problem:** The camera names are hardcoded. They must match the keys used in your LeRobot dataset.

**Action:** Update the dictionary keys to match your dataset.

**Current Code:**
```python
base_image = _parse_image(data["observation/images.main.left"])
wrist_image_right = _parse_image(data["observation/images.secondary_0"])
wrist_image_left = _parse_image(data["observation/images.secondary_1"])
```

**Required Change (Example):**
```python
# --- REQUIRED CHANGE ---
# IMPORTANT: Replace with the actual camera names from your dataset!
base_image = _parse_image(data["observation/images/main_cam"])
wrist_1_image = _parse_image(data["observation/images/wrist_cam_1"])
wrist_2_image = _parse_image(data["observation/images/wrist_cam_2"])

# Also update the image_dict to use your new variables
images = {
    "base_0_rgb": base_image,
    "left_wrist_0_rgb": wrist_1_image,
    "right_wrist_0_rgb": wrist_2_image,
}
```

### 3. Action Trimming in `S0100Outputs`

**Problem:** The current code trims the model's output to 12 actions, which is for a dual-arm robot. This will fail for your 6-action single-arm robot.

**Action:** Change the trimming value from 12 to 6.

**Current Code:**
```python
return {"actions": np.asarray(data["actions"][:, :12])}
```

**Required Change:**
```python
# --- REQUIRED CHANGE ---
return {"actions": np.asarray(data["actions"][:, :6])}
```

---

## Optional but Recommended Change

### `make_so100_example()` function

**Problem:** This helper function currently generates a random example for a 12-dimension robot, which is misleading.

**Action:** Update it to generate an example that matches your 6-DoF robot.

**Current Code:**
```python
def make_so100_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(12),
        # ... images
    }
```

**Recommended Change:**
```python
# --- RECOMMENDED CHANGE ---
def make_so100_example() -> dict:
    """Creates a random input example for the SO-100 single-arm policy."""
    return {
        # Use the separate keys that your S0100Inputs class now expects
        "observation/joint_position": np.random.rand(5),
        "observation/gripper_position": np.random.rand(1),
        # Use the camera names that your S0100Inputs class now expects
        "observation/images/main_cam": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/wrist_cam_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/wrist_cam_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }
```
