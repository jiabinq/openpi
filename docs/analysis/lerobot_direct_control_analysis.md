# LeRobot Direct Robot Control Analysis

Analysis of how LeRobot's client directly controls real robots like SO-100 **without format conversion** to external policy servers.

## Overview

LeRobot uses a **local inference approach** where the policy model runs directly on the same system as the robot controller, eliminating the need for format conversion to external servers like OpenPi.

## Direct Control Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   SO-100    │ ←→ │   LeRobot    │ ←→ │   Policy    │ ←→ │   Model     │
│  Hardware   │    │   Robot      │    │  (Local)    │    │ (π₀, ACT,   │
│             │    │  Interface   │    │  Inference  │    │  SmolVLA)   │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
```

**Key Insight**: No network communication or format conversion needed - everything runs locally.

## SO-100 Direct Control Implementation

### **1. Robot Hardware Interface**

```python
# From SO100Follower class
class SO100Follower(Robot):
    def __init__(self, config: SO100FollowerConfig):
        # Direct hardware connection via serial port
        self.bus = FeetechMotorsBus(
            port=self.config.port,  # "/dev/ttyACM0"
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            }
        )
        self.cameras = make_cameras_from_configs(config.cameras)
```

**Direct hardware control**: No middleware, direct serial communication to motors.

### **2. Native Observation Format**

```python
def get_observation(self) -> dict[str, Any]:
    """LeRobot's native observation format - no conversion needed."""
    
    # Read joint positions directly
    obs_dict = self.bus.sync_read("Present_Position")
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
    # Result: {"shoulder_pan.pos": 0.1, "shoulder_lift.pos": 0.2, ...}
    
    # Read camera images directly  
    for cam_key, cam in self.cameras.items():
        obs_dict[cam_key] = cam.async_read()
    # Result: {"front": image_array, "wrist": image_array, ...}
    
    return obs_dict
```

**Native format**: Exactly what LeRobot policies expect - no conversion layer.

### **3. Local Policy Inference**

```python
# From record.py - shows local inference workflow
def record_loop(robot, policy, ...):
    while timestamp < control_time_s:
        # 1. Get observation in LeRobot native format
        observation = robot.get_observation()
        # {"shoulder_pan.pos": 0.1, "front": image, "wrist": image, ...}
        
        # 2. Convert to dataset frame format (internal LeRobot conversion)
        observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
        # {"observation.shoulder_pan.pos": tensor, "observation.front": tensor, ...}
        
        # 3. Local policy inference (no network calls)
        action_values = predict_action(
            observation_frame,
            policy,                    # Local policy object
            device,                   # Local GPU/CPU
            use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )
        
        # 4. Convert actions back to robot format
        action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        # {"shoulder_pan.pos": 0.15, "shoulder_lift.pos": 0.25, ...}
        
        # 5. Send directly to robot hardware
        robot.send_action(action)
```

**Local inference**: Policy runs on same machine - no network latency or format conversion.

### **4. Action Execution**

```python
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    """Direct action execution - no format conversion."""
    
    # Extract joint positions (remove ".pos" suffix)
    goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
    
    # Safety checking
    if self.config.max_relative_target is not None:
        present_pos = self.bus.sync_read("Present_Position")
        goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
    
    # Send directly to hardware
    self.bus.sync_write("Goal_Position", goal_pos)
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}
```

**Direct execution**: Actions go straight to hardware with safety checks.

## Data Flow Comparison

### **LeRobot Direct Control**

```python
# Single system, no network
observation = robot.get_observation()
# {"shoulder_pan.pos": 0.1, "front": array(...), "wrist": array(...)}

actions = policy.predict(observation)  # Local inference
# tensor([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])

robot.send_action(actions)  # Direct to hardware
```

### **OpenPi Remote Inference (What We're Implementing)**

```python
# Two systems, network communication required
raw_obs = robot.get_observation()  # LeRobot format
# {"shoulder_pan.pos": 0.1, "front": array(...), "wrist": array(...)}

openpi_obs = convert_to_openpi(raw_obs)  # Format conversion needed
# {"observation/state": array([0.1,0.2,0.3,0.4,0.5,0.6]), 
#  "observation/images.main.left": array(...), "prompt": "task"}

actions = openpi_client.infer(openpi_obs)  # Network call
# {"actions": array([[0.15, 0.25, 0.35, 0.45, 0.55, 0.65], ...])}

robot_actions = convert_to_lerobot(actions[0])  # Format conversion needed
# {"shoulder_pan.pos": 0.15, "shoulder_lift.pos": 0.25, ...}

robot.send_action(robot_actions)  # Same hardware interface
```

## Why LeRobot Doesn't Need Format Conversion

### **1. Unified Ecosystem**
- **Robot interface** and **policy models** use the same native format
- **No impedance mismatch** between components
- **Direct tensor operations** without serialization/deserialization

### **2. Local Execution Benefits**
- **No network latency** (critical for 30Hz robot control)
- **No serialization overhead** (numpy arrays stay as PyTorch tensors)
- **No connection reliability issues** (no WebSocket failures)
- **Shared GPU memory** (efficient for large models)

### **3. Built-in Format Compatibility**

```python
# LeRobot observation features match policy expectations
robot.observation_features = {
    "shoulder_pan.pos": float,
    "shoulder_lift.pos": float,
    "front": (480, 640, 3),
    "wrist": (480, 640, 3),
    # ... exactly what policy expects
}

robot.action_features = {
    "shoulder_pan.pos": float,
    "shoulder_lift.pos": float,
    # ... exactly what policy outputs
}
```

## Internal LeRobot Format Conversion

LeRobot **does** do format conversion, but **internally** for model compatibility:

```python
def predict_action(observation, policy, device, use_amp, task=None, robot_type=None):
    """Internal LeRobot format conversion for model inference."""
    
    # Convert numpy → PyTorch tensors
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        
        if "image" in name:
            # Normalize images: uint8 [0,255] → float32 [0,1]
            observation[name] = observation[name].float() / 255.0
            
            # Add batch dimension: (H,W,C) → (1,H,W,C)
            if observation[name].ndim == 3:
                observation[name] = observation[name].unsqueeze(0)
                
            # Rearrange: (1,H,W,C) → (1,C,H,W) for CNN models
            if observation[name].shape[-1] == 3:
                observation[name] = observation[name].permute(0, 3, 1, 2)
    
    # Move to GPU
    observation = {k: v.to(device) for k, v in observation.items()}
    
    # Policy inference
    with torch.inference_mode():
        actions = policy.predict_action_chunk(observation, task=task)
    
    # Convert back to numpy
    return actions.cpu().numpy()
```

**Internal conversion**: Optimized for PyTorch models, no network overhead.

## Key Advantages of LeRobot Direct Control

| Aspect | LeRobot Direct | OpenPi Remote | Winner |
|--------|----------------|---------------|---------|
| **Latency** | ~1-5ms (local) | ~20-50ms (network + inference) | ✅ **LeRobot** |
| **Reliability** | Hardware only failure points | Network + hardware failure points | ✅ **LeRobot** |
| **Memory Efficiency** | Shared GPU memory | Serialization overhead | ✅ **LeRobot** |
| **Setup Complexity** | Single system | Two systems + network config | ✅ **LeRobot** |
| **Scalability** | One robot per machine | Multiple robots per server | ✅ **OpenPi** |
| **Model Flexibility** | Any LeRobot model | Only OpenPi-compatible models | ✅ **LeRobot** |

## Why Use Remote Inference Then?

Despite LeRobot's advantages, remote inference (like OpenPi) has specific benefits:

### **1. Computational Resources**
```python
# Robot machine: Limited resources (Jetson, NUC)
robot_specs = {
    "gpu": "Jetson Orin (limited VRAM)",
    "ram": "8-16GB", 
    "storage": "Limited SSD"
}

# Server machine: High-end resources
server_specs = {
    "gpu": "RTX 4090 (24GB VRAM)",
    "ram": "64-128GB",
    "storage": "High-speed NVMe"
}
```

### **2. Multiple Robot Support**
```python
# One powerful server → Many robots
server_connections = {
    "robot_1": "SO-100 Assembly Line",
    "robot_2": "SO-100 Quality Control", 
    "robot_3": "ALOHA Kitchen Station",
    "robot_4": "DROID Warehouse Pick"
}
```

### **3. Model Management**
```python
# Server: Easy model updates
server.load_model("openpi/pi0-fast-latest")  # Instant deployment

# vs LeRobot: Must update each robot individually  
for robot in robot_fleet:
    robot.update_model("new_model")  # Time-consuming
```

## Implications for Our SO-100 Implementation

### **Understanding the Trade-off**
- **LeRobot direct**: Better performance, simpler setup
- **OpenPi remote**: Better scalability, centralized compute

### **Why We Need Format Conversion**
We're **choosing remote inference** for:
1. **Powerful server**: Better models, faster inference
2. **Multiple robots**: Future scalability  
3. **Centralized updates**: Easier model management

**Cost**: Must handle format conversion between LeRobot's excellent robot interface and OpenPi's inference server.

### **Architecture Justification**
Our hybrid approach makes sense:
- ✅ **Use LeRobot's superior robot interface** (production-grade hardware control)
- ✅ **Use OpenPi's powerful inference server** (centralized compute)
- ✅ **Handle format conversion** (bridge the gap)

## Conclusion

**LeRobot doesn't need format conversion** because it uses **local inference** with a **unified ecosystem**. Everything speaks the same "language" natively.

**We need format conversion** because we're choosing **remote inference** for its scalability benefits, requiring us to bridge between LeRobot's robot interface format and OpenPi's server format.

Our implementation is essentially **building a bridge** between two excellent but incompatible systems:
- **LeRobot**: Best robot hardware interface
- **OpenPi**: Best remote inference server  

The format conversion is the **necessary translation layer** to get the best of both worlds.