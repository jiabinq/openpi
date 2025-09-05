# Data Format Clarification: OpenPi vs GR00T

## You're Partially Right - But There Are Important Differences

While both OpenPi and GR00T work with robots via LeRobot, they use **different data formats** for the policy server communication.

## Data Format Comparison

### **Robot Observation Layer** (LeRobot) ✅ SAME
Both use LeRobot's standard robot interface:

```python
# Both GR00T and OpenPi use this
from lerobot.robots import make_robot_from_config
robot = make_robot_from_config(robot_config)
observation_dict = robot.get_observation()

# LeRobot format (both get this):
observation_dict = {
    "front": np.array(...),        # Front camera image
    "wrist": np.array(...),        # Wrist camera image  
    "shoulder_pan.pos": 0.1,       # Individual joint positions
    "shoulder_lift.pos": 0.2,
    "elbow_flex.pos": 0.3,
    "wrist_flex.pos": 0.4,
    "wrist_roll.pos": 0.5,
    "gripper.pos": 0.6,
}
```

### **Policy Server Communication** ❌ DIFFERENT

#### **OpenPi DROID Format** (WebSocket)
```python
request_data = {
    "observation/exterior_image_1_left": resized_image,   # OpenPi naming
    "observation/wrist_image_left": resized_wrist_image,
    "observation/joint_position": np.array([...]),        # Combined joints
    "observation/gripper_position": np.array([...]),      # Separate gripper
    "prompt": instruction,
}
```

#### **Phospho SO-100 Format** (Expected by OpenPi server)
```python
# What phospho's SO-100 policy expects
{
    "observation/state": np.array(6),                    # All 6 DOF combined
    "observation/images.main.left": image,               # Different naming
    "observation/images.secondary_0": wrist_image,
    "prompt": "do something",
}
```

#### **GR00T Format** (ZMQ)
```python
# GR00T's internal format
obs_dict = {
    "video.front": observation_dict["front"],            # video.* prefix
    "video.wrist": observation_dict["wrist"], 
    "state.single_arm": state[:5],                       # Semantic grouping
    "state.gripper": state[5:6],
    "annotation.human.task_description": lang,           # Different prompt key
}
```

## Key Differences Identified

### 1. **Image Key Naming**
| System | Front Camera | Wrist Camera |
|--------|-------------|--------------|
| **OpenPi DROID** | `observation/exterior_image_1_left` | `observation/wrist_image_left` |
| **Phospho SO-100** | `observation/images.main.left` | `observation/images.secondary_0` |
| **GR00T** | `video.front` | `video.wrist` |

### 2. **State Representation**
| System | Joint Format | Gripper Format |
|--------|-------------|----------------|
| **OpenPi DROID** | `observation/joint_position` (7 DOF) | `observation/gripper_position` (1 DOF) |
| **Phospho SO-100** | `observation/state` (6 DOF combined) | Included in state |
| **GR00T** | `state.single_arm` (5 DOF) | `state.gripper` (1 DOF) |

### 3. **Prompt/Instruction Key**
| System | Key Name |
|--------|----------|
| **OpenPi DROID** | `prompt` |
| **Phospho SO-100** | `prompt` |
| **GR00T** | `annotation.human.task_description` |

## Implications for SO-100 Implementation

### **Option 1: Use Phospho's Format** (Easiest)
If using phospho's SO-100 policies, the client should send:
```python
request_data = {
    "observation/state": np.array([joint1, joint2, joint3, joint4, joint5, gripper]),  # 6 DOF
    "observation/images.main.left": front_camera_image,
    "observation/images.secondary_0": wrist_camera_image,
    "prompt": task_instruction,
}
```

### **Option 2: Use DROID's Format** (Requires new policy)
If using DROID-style format, need to implement new SO-100 policy:
```python  
request_data = {
    "observation/exterior_image_1_left": front_camera_image,
    "observation/wrist_image_left": wrist_camera_image,
    "observation/joint_position": joint_positions,  # 5 DOF
    "observation/gripper_position": gripper_position,  # 1 DOF  
    "prompt": task_instruction,
}
```

### **Option 3: Adapt GR00T's Format** (Most work)
Convert LeRobot observations to GR00T format, then adapt for OpenPi WebSocket.

## Revised Implementation Strategy

Since the data formats are different, the work required is:

### **Easy Path: Use Phospho's Policies**
1. ✅ Use LeRobot for robot interface (same as GR00T)
2. 🔄 **Format data for phospho's expected input** (different from GR00T)
3. ✅ Use existing OpenPi WebSocket server with phospho policies

### **Medium Path: Create DROID-style SO-100 Policy**
1. ✅ Use LeRobot for robot interface  
2. 🔄 **Create new SO-100 policy following DROID pattern**
3. 🔄 **Adapt DROID's data format for 6 DOF**

### **Hard Path: Full GR00T Adaptation**
1. ✅ Use LeRobot for robot interface
2. 🔄 **Implement GR00T's data format conversion**  
3. 🔄 **Adapt OpenPi server for GR00T-style inputs**

## Recommended Approach

**Use Phospho's format** since:
1. ✅ Policies already exist and work
2. ✅ Only need client-side format conversion  
3. ✅ Minimal changes to OpenPi server
4. ✅ Can still use GR00T's clean evaluation structure

```python
# Hybrid approach: GR00T structure + Phospho format + OpenPi WebSocket
class OpenPiSO100Client:
    def get_action(self, observation_dict, lang):
        # Convert LeRobot obs → Phospho format  
        joint_state = np.array([
            observation_dict[k] for k in [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
            ]
        ])
        
        request = {
            "observation/state": joint_state,  # Phospho expects this
            "observation/images.main.left": observation_dict["front"],
            "observation/images.secondary_0": observation_dict["wrist"],
            "prompt": lang,
        }
        
        return self.client.infer(request)["actions"]
```

## Conclusion

You were right that both use LeRobot for **robot interface**, but they use **different formats for policy communication**. The format differences are manageable and the hybrid approach (GR00T structure + Phospho format) is the most practical solution.