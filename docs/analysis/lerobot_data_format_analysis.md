# LeRobot Data Format and Configuration Analysis

## Key Finding: LeRobot Uses Its Own Dataset Format (Not Direct OpenPi Format)

After examining LeRobot's implementation, here's what I found about their data format and configuration:

## LeRobot's Data Format Implementation

### **1. Data Format Conversion Pipeline**

LeRobot uses a **3-stage conversion process**:

```python
# Stage 1: Robot Hardware → Raw Observation
observation = robot.get_observation()
# Returns: {"front": image, "wrist": image, "shoulder_pan.pos": 0.1, ...}

# Stage 2: Raw Observation → Dataset Format  
observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
# Converts to: {"observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "observation.images.front": image}

# Stage 3: Dataset Format → Policy Format (when using policy)
action_values = predict_action(observation_frame, policy, device, ...)
# Policy expects LeRobot's internal format, not OpenPi format
```

### **2. LeRobot's Dataset Format Structure**

```python
# LeRobot automatically creates this structure
def hw_to_dataset_features(hw_features, prefix, use_video=True):
    features = {}
    joint_fts = {key: ftype for key, ftype in hw_features.items() if ftype is float}
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    # For observations - combines all joints into one array
    if joint_fts and prefix == "observation":
        features[f"{prefix}.state"] = {
            "dtype": "float32", 
            "shape": (len(joint_fts),),  # (6,) for SO-100
            "names": list(joint_fts),    # ["shoulder_pan.pos", "shoulder_lift.pos", ...]
        }
    
    # For images - prefixes with observation.images
    for key, shape in cam_fts.items():
        features[f"{prefix}.images.{key}"] = {
            "dtype": "video" if use_video else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
```

### **3. Actual SO-100 Data Format in LeRobot**

Based on the code analysis, LeRobot's SO-100 format is:

```python
# Raw robot observation (from robot.get_observation())
raw_observation = {
    "front": np.array(...),           # Camera images
    "wrist": np.array(...),
    "shoulder_pan.pos": 0.1,          # Individual joint positions
    "shoulder_lift.pos": 0.2,
    "elbow_flex.pos": 0.3,
    "wrist_flex.pos": 0.4,
    "wrist_roll.pos": 0.5,
    "gripper.pos": 0.6,
}

# LeRobot dataset format (after build_dataset_frame)
dataset_frame = {
    "observation.state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # Combined state
    "observation.images.front": front_image,
    "observation.images.wrist": wrist_image,
}
```

## LeRobot Configuration System

### **1. SO-100 Configuration Structure**

```python
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SO100FollowerConfig(RobotConfig):
    port: str                                    # "/dev/ttyACM0"
    disable_torque_on_disconnect: bool = True
    max_relative_target: int | None = None       # Safety limits
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = False                    # Radians by default
```

### **2. Camera Configuration**

```python
# LeRobot's camera config (from so100_client.py example)
cameras_config = {}
for k, v in cameras.items():
    cameras_config[k] = OpenCVCameraConfig(**v)

# Usage example:
cameras = {
    "front": {"type": "opencv", "index": 0, "width": 640, "height": 480},
    "wrist": {"type": "opencv", "index": 1, "width": 640, "height": 480}
}
```

## Comparison with OpenPi Requirements

### **Data Format Compatibility**

| Component | LeRobot Format | OpenPi DROID Format | Phospho SO-100 Format |
|-----------|----------------|---------------------|----------------------|
| **State** | `observation.state` (6D array) | `observation/joint_position` + `observation/gripper_position` | `observation/state` (6D array) |
| **Images** | `observation.images.front` | `observation/exterior_image_1_left` | `observation/images.main.left` |
| **Wrist Cam** | `observation.images.wrist` | `observation/wrist_image_left` | `observation/images.secondary_0` |

### **Key Insights:**

1. **LeRobot ≠ Phospho Format**: LeRobot uses `observation.images.front` vs Phospho's `observation/images.main.left`

2. **LeRobot ≠ DROID Format**: LeRobot combines state vs DROID separates joints and gripper

3. **LeRobot Uses Own Standard**: LeRobot has its own dataset format that's different from all OpenPi variants

## OpenPi Integration in LeRobot Aug

The `/home/jiabin/Documents/lerobot_aug/openpi_pytorch/` directory contains **OpenPi integration files**:

```python
# From so100_client.py - Shows how to convert LeRobot → OpenPi
class SO100Env(gym.Env):
    def get_observation(self):
        raw_observation = self.robot.get_observation()
        # Convert LeRobot format → OpenPi format
        state = np.array([
            raw_observation['shoulder_pan.pos'],    # LeRobot individual joints
            raw_observation['shoulder_lift.pos'],
            raw_observation['elbow_flex.pos'],
            raw_observation['wrist_flex.pos'],
            raw_observation['wrist_roll.pos'],
            raw_observation['gripper.pos'],
        ])
        observation = {'state': state}               # OpenPi combined state
        
        # Resize images for OpenPi
        for camera_name in self.camera_names:
            _image = cv2.resize(raw_observation[camera_name], (224, 224))
            observation[camera_name] = _image
        
        return observation
```

## Recommendations for SO-100 OpenPi Implementation

### **1. Don't Use LeRobot's Data Format Directly**

LeRobot's `record.py` uses **LeRobot's internal dataset format**, not OpenPi format. The formats are incompatible:

- LeRobot: `observation.images.front` 
- OpenPi: `observation/exterior_image_1_left` or `observation/images.main.left`

### **2. Use LeRobot for Robot Interface, Convert Format**

**Best approach**: Use LeRobot's robot interface but convert to OpenPi format:

```python
# Use LeRobot for robot setup (superior to GR00T)
robot = make_robot_from_config(SO100FollowerConfig(...))
robot.connect()

# Get observation in LeRobot format
raw_obs = robot.get_observation()

# Convert to Phospho's OpenPi format  
openpi_obs = {
    "observation/state": np.array([
        raw_obs["shoulder_pan.pos"],
        raw_obs["shoulder_lift.pos"], 
        raw_obs["elbow_flex.pos"],
        raw_obs["wrist_flex.pos"],
        raw_obs["wrist_roll.pos"],
        raw_obs["gripper.pos"]
    ]),
    "observation/images.main.left": resize_image(raw_obs["front"]),
    "observation/images.secondary_0": resize_image(raw_obs["wrist"]),
    "prompt": task_instruction
}

# Send to OpenPi server
actions = openpi_client.infer(openpi_obs)["actions"]
```

### **3. Use LeRobot Config, Adapt for OpenPi**

LeRobot's configuration system **is superior** and should be used:

```python
@dataclass
class SO100OpenPiEvalConfig:
    # Use LeRobot's robot config (superior)
    robot: SO100FollowerConfig
    
    # OpenPi-specific config
    openpi_host: str = "localhost"
    openpi_port: int = 8000
    
    # Evaluation config
    max_episodes: int = 10
    action_horizon: int = 16
    task_instruction: str = "Pick up the object"
```

## Final Recommendation

**For SO-100 OpenPi implementation:**

1. ✅ **Robot Interface**: Use LeRobot's SO-100 setup (superior to GR00T)
2. 🔄 **Data Format**: Convert LeRobot format → Phospho OpenPi format  
3. ✅ **Configuration**: Use LeRobot's config system (superior to GR00T)
4. ✅ **Evaluation Features**: Keep DROID's video/CSV features

**The conversion layer is the key** - LeRobot gives you the best robot interface, but you need to convert the data format to work with OpenPi's inference server.

LeRobot's `record.py` is excellent for **robot interface patterns** but **not directly compatible** with OpenPi's data format requirements.