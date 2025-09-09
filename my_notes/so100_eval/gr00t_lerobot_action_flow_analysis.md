# Gr00t Client Action Processing Analysis (`eval_lerobot.py`)

This document analyzes the action processing flow in the `eval_lerobot.py` script, which shows how the Gr00t/Nvidia client controls an SO100 arm using the `lerobot` framework. This provides a reference for building a custom SO100 client.

---

### 1. Action Processing Flow

The script's logic is centered around the `Gr00tRobotInferenceClient` class, which adapts the output from a generic policy server to the specific format required by the `lerobot` robot API.

#### Step 1: Receive Action Chunk

The process starts in the `get_action` method, which calls `self.policy.get_action(obs_dict)`. This retrieves an action chunk from the server.

The chunk is a **dictionary of NumPy arrays**, where keys represent different action modalities. For the SO100, these are:
*   `action.single_arm`: An array of shape `(horizon, 5)` for the 5 arm joints.
*   `action.gripper`: An array of shape `(horizon, 1)` for the gripper.

#### Step 2: Transform the Entire Chunk into `lerobot` Actions

Unlike the DROID example, this script immediately processes the *entire* action chunk into a list of `lerobot`-compatible action dictionaries. It does not process one action at a time in the main loop.

```python
# convert the action chunk to a list of dict[str, float]
lerobot_actions = []
action_horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
for i in range(action_horizon):
    action_dict = self._convert_to_lerobot_action(action_chunk, i)
    lerobot_actions.append(action_dict)
return lerobot_actions
```

#### Step 3: Convert to a `lerobot` Action Dictionary

This is the most critical transformation step, performed by the `_convert_to_lerobot_action` method. For each timestep `i` in the chunk, it does the following:

1.  **Concatenates** the values from the different modalities (`action.single_arm` and `action.gripper`) into a single, flat NumPy array of 6 floats.
2.  **Creates a dictionary** by mapping these 6 float values to the string names of the robot's motors (e.g., `'shoulder_pan.pos'`, `'gripper.pos'`).

The resulting `action_dict` is the format required by the `lerobot` API:

```python
# Example output for a single timestep:
{
    'shoulder_pan.pos': 0.123,
    'shoulder_lift.pos': -0.456,
    'elbow_flex.pos': 0.789,
    'wrist_flex.pos': -0.111,
    'wrist_roll.pos': 0.222,
    'gripper.pos': 0.999
}
```

#### Step 4: Execute Actions in the Main Loop

The main `eval` function receives the fully processed list of action dictionaries. It then iterates through this list, sending one dictionary at a time to the `lerobot` robot object.

```python
# The main eval function gets the list of processed actions
action_chunk = policy.get_action(observation_dict, language_instruction)

# It then iterates and sends each action dictionary to the robot
for i in range(cfg.action_horizon):
    action_dict = action_chunk[i]
    robot.send_action(action_dict)
    time.sleep(0.02)
```

---

### 2. Implications for Building an SO100 Client

This analysis reveals a clear pattern for controlling an SO100 arm within the `lerobot` ecosystem:

1.  **Action Format is a Dictionary:** The `lerobot` hardware API (`robot.send_action`) requires a **dictionary**, not a simple numerical array. The keys must be the string names of the motors, and the values are their target positions.

2.  **Transformation is Required:** A client must transform the numerical array(s) received from the policy server into this specific dictionary format.

3.  **Motor Names are Key:** To build the action dictionary correctly, the client needs to know the exact names for each motor axis (e.g., `'shoulder_pan.pos'`). The `eval_lerobot.py` script retrieves these dynamically from the `robot` object, which is a robust approach.
