# Minimal SO-100 Implementation: Keep OpenPi, Change Only What's Needed

## Strategy: Minimal OpenPi Changes

Instead of recreating everything, let's **keep OpenPi's existing infrastructure** and only change the **robot-specific parts**.

## What to Keep from OpenPi (No Changes Needed)

### ✅ **Server Side** (Keep 100%)
```bash
# OpenPi server works as-is with phospho's SO-100 policies
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_so100_single \
  --policy.dir=path/to/so100/checkpoint
```

### ✅ **Client Infrastructure** (Keep ~90%)
- `openpi_client.websocket_client_policy` - Keep as-is
- `openpi_client.image_tools` - Keep as-is  
- Action chunking logic - Keep DROID's pattern
- Video recording, CSV export - Keep DROID's implementation
- Error handling, timing - Keep DROID's approach

## What Must Change (Robot-Specific Parts)

### 🔄 **1. Robot Interface** 
**What**: Replace DROID's custom `RobotEnv` with LeRobot SO-100
**Reference**: **GR00T eval_lerobot.py** (lines 207-221)

```python
# REMOVE: DROID's custom robot interface
env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")

# ADD: LeRobot SO-100 interface (from GR00T)
from lerobot.robots import make_robot_from_config, RobotConfig

robot_config = RobotConfig(
    type="so100_follower",
    port="/dev/ttyACM0", 
    cameras={
        "front": {"type": "opencv", "index": 0},
        "wrist": {"type": "opencv", "index": 1}
    }
)
robot = make_robot_from_config(robot_config)
robot.connect()
```

### 🔄 **2. Observation Extraction**
**What**: Replace DROID's complex camera handling with simple LeRobot observations
**Reference**: **GR00T eval_lerobot.py** (lines 238-241)

```python
# REMOVE: DROID's _extract_observation function (60+ lines)
def _extract_observation(args, obs_dict, save_to_disk=False):
    # ... 60 lines of complex camera ID matching

# ADD: Simple LeRobot observation (from GR00T)  
observation_dict = robot.get_observation()
# Gets: {"front": image, "wrist": image, "joint1.pos": val, ...}
```

### 🔄 **3. Data Format for Policy Server**
**What**: Change request format for SO-100 
**Reference**: **Phospho so100_policy_single.py** (make_so100_example)

```python
# CHANGE: DROID's request format
request_data = {
    "observation/exterior_image_1_left": resize_image(external_cam),
    "observation/wrist_image_left": resize_image(wrist_cam), 
    "observation/joint_position": joint_positions,      # 7 DOF
    "observation/gripper_position": gripper_position,   # 1 DOF
    "prompt": instruction,
}

# TO: Phospho's SO-100 format
request_data = {
    "observation/state": np.array([                     # 6 DOF combined
        obs["shoulder_pan.pos"], obs["shoulder_lift.pos"],
        obs["elbow_flex.pos"], obs["wrist_flex.pos"], 
        obs["wrist_roll.pos"], obs["gripper.pos"]
    ]),
    "observation/images.main.left": resize_image(obs["front"]),
    "observation/images.secondary_0": resize_image(obs["wrist"]),
    "prompt": instruction,
}
```

### 🔄 **4. Action Dimensions**
**What**: Change from 8 DOF to 6 DOF
**Reference**: **DROID main.py** (adapt dimensions)

```python
# CHANGE: DROID's 8 DOF
assert pred_action_chunk.shape == (10, 8)  # 7 joints + 1 gripper

# TO: SO-100's 6 DOF  
assert pred_action_chunk.shape == (16, 6)  # 5 joints + 1 gripper
```

### 🔄 **5. Action Execution**
**What**: Change robot command format
**Reference**: **GR00T eval_lerobot.py** (lines 243-247)

```python
# CHANGE: DROID's action execution
env.step(action)  # Direct numpy array

# TO: LeRobot's action execution (from GR00T)
action_dict = {
    "shoulder_pan.pos": action[0],
    "shoulder_lift.pos": action[1], 
    "elbow_flex.pos": action[2],
    "wrist_flex.pos": action[3],
    "wrist_roll.pos": action[4],
    "gripper.pos": action[5],
}
robot.send_action(action_dict)
```

### 🔄 **6. Configuration Parameters**
**What**: Update hardware-specific parameters
**Reference**: **GR00T eval_lerobot.py** (Args class)

```python
# CHANGE: DROID's complex camera config
left_camera_id: str = "<your_camera_id>"
right_camera_id: str = "<your_camera_id>"
wrist_camera_id: str = "<your_camera_id>" 
external_camera: str = "left"

# TO: Simple SO-100 config (from GR00T)
robot_type: str = "so100_follower"
robot_port: str = "/dev/ttyACM0"
camera_front_id: int = 0
camera_wrist_id: int = 1
```

## Implementation Plan: Modify DROID's main.py

### **Step 1**: Copy DROID as Template
```bash
cp examples/droid/main.py examples/so100/main.py
```

### **Step 2**: Replace Robot Interface (10 lines changed)
Replace lines 80-81 and observation extraction function with LeRobot calls.

### **Step 3**: Update Data Format (5 lines changed)  
Modify request_data dictionary to use phospho's format.

### **Step 4**: Fix Action Dimensions (3 lines changed)
Update shape assertions and action indexing.

### **Step 5**: Replace Action Execution (5 lines changed)
Change from `env.step()` to `robot.send_action()`.

### **Step 6**: Update Configuration (10 lines changed)
Replace camera ID parameters with SO-100 config.

## Best References for Each Component

| Component | Best Reference | Why |
|-----------|---------------|-----|
| **Robot Interface** | **GR00T eval_lerobot.py** | Only SO-100 specific example |
| **LeRobot Integration** | **GR00T eval_lerobot.py** | Production-quality LeRobot usage |
| **Data Format** | **Phospho so100_policy_single.py** | Defines expected OpenPi SO-100 format |
| **Action Chunking** | **DROID main.py** | Keep OpenPi's proven chunking logic |
| **Video/CSV Export** | **DROID main.py** | Keep OpenPi's evaluation features |
| **Error Handling** | **DROID main.py** | Keep OpenPi's robust error handling |
| **Configuration** | **GR00T eval_lerobot.py** | Clean, production-ready config |

## Expected Code Changes

**Total Lines Modified**: ~40 lines out of 247 (16% change)

### **Core Function Structure** (Keep from DROID)
```python
def main(args):
    # KEEP: OpenPi's evaluation loop structure
    while True:
        instruction = input("Enter instruction: ")
        
        for t_step in range(args.max_timesteps):
            # KEEP: Action chunking logic
            if actions_completed >= args.open_loop_horizon:
                # CHANGE: Data format only
                action_chunk = client.infer(new_request_format)
            
            # CHANGE: Action execution only  
            robot.send_action(action_dict)
            
            # KEEP: Timing, video recording, etc.
```

## Dependencies to Add

```python
# Add to imports (top of file)
from lerobot.robots import make_robot_from_config, RobotConfig

# Add to requirements.txt
lerobot[so100_follower]
```

## File Structure
```
examples/so100/
├── main.py              # Modified DROID main.py (~40 line changes)
├── README.md            # Usage instructions (new)
└── requirements.txt     # Add lerobot dependency (new)
```

## Usage (Same as DROID)

### **Server** (No changes)
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_so100_single \  
  --policy.dir=path/to/checkpoint
```

### **Client** (Minimal parameter changes)
```bash
python examples/so100/main.py \
  --robot_type=so100_follower \
  --robot_port=/dev/ttyACM0 \
  --remote_host=GPU_SERVER_IP \
  --remote_port=8000
```

## Summary: Minimal Changes Strategy

1. ✅ **Keep 90% of DROID's code** - proven evaluation framework
2. 🔄 **Replace robot interface** - Use GR00T's LeRobot approach  
3. 🔄 **Adapt data format** - Use phospho's SO-100 format
4. 🔄 **Update dimensions** - Change 8 DOF → 6 DOF
5. ✅ **Keep all evaluation features** - video, CSV, error handling

**Result**: Full SO-100 support with minimal changes to OpenPi's proven infrastructure!

**Best References**:
- **Robot Interface**: GR00T (SO-100 specific)
- **Data Format**: Phospho (OpenPi compatible)  
- **Everything Else**: Keep DROID (proven framework)