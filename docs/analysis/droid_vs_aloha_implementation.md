# DROID vs ALOHA Implementation Comparison

## Architecture Overview

| Aspect | DROID | ALOHA |
|--------|--------|--------|
| **Control Pattern** | Manual control loop | Runtime framework |
| **Complexity** | Simple, direct | Abstracted, modular |
| **Code Length** | 247 lines | 52 lines |
| **User Interaction** | Interactive prompts | Automated episodes |

## Detailed Comparison

### 1. **Control Architecture**

#### DROID (Manual Control Loop)
```python
# Direct manual control
while True:
    instruction = input("Enter instruction: ")
    
    for t_step in range(max_timesteps):
        # Get observation
        obs = env.get_observation()
        
        # Get action chunk when needed
        if need_new_chunk():
            action_chunk = client.infer(request_data)["actions"]
        
        # Execute single action
        action = action_chunk[current_index]
        env.step(action)
        
        # Manual timing control
        time.sleep(control_frequency)
```

#### ALOHA (Runtime Framework)
```python
# Uses Runtime framework for abstraction
runtime = Runtime(
    environment=AlohaRealEnvironment(),
    agent=PolicyAgent(
        policy=ActionChunkBroker(
            policy=WebsocketClientPolicy()
        )
    ),
    max_hz=50,
    num_episodes=1
)
runtime.run()  # Everything happens inside framework
```

### 2. **Action Chunking**

#### DROID
- **Manual chunking**: Tracks `actions_from_chunk_completed`
- **Open-loop horizon**: 8 steps (0.5 seconds at 15Hz)
- **Action shape**: `(10, 8)` - 10 timesteps, 8 DOF (7 joints + 1 gripper)
- **Manual action processing**: Gripper binarization, clipping

#### ALOHA
- **Automatic chunking**: Uses `ActionChunkBroker`
- **Action horizon**: 25 steps
- **Framework handles**: All action processing internally

### 3. **Robot Interface**

#### DROID
```python
# Direct robot environment
env = RobotEnv(
    action_space="joint_velocity",  # Joint velocities
    gripper_action_space="position"  # Gripper position
)

# Manual state extraction
joint_position = robot_state["joint_positions"]  # 7 DOF
gripper_position = [robot_state["gripper_position"]]  # 1 DOF
```

#### ALOHA
```python
# Custom environment wrapper
environment = AlohaRealEnvironment(
    reset_position=metadata.get("reset_pose")
)
# All robot interface hidden in environment class
```

### 4. **Camera Handling**

#### DROID (Multi-camera with selection)
```python
# Hardware parameters
left_camera_id: str = "24259877"
right_camera_id: str = "24514023" 
wrist_camera_id: str = "13062452"
external_camera: str = "left"  # User selects which to use

# Camera processing
request_data = {
    "observation/exterior_image_1_left": resize_image(external_cam),
    "observation/wrist_image_left": resize_image(wrist_cam),
    "observation/joint_position": joint_pos,
    "observation/gripper_position": gripper_pos,
    "prompt": instruction,
}
```

#### ALOHA (Framework handles cameras)
- Camera setup handled in `AlohaRealEnvironment`
- No explicit camera configuration in main script

### 5. **Evaluation & Metrics**

#### DROID (Comprehensive tracking)
```python
# Video recording
video = []
video.append(current_frame)
ImageSequenceClip(video).write_videofile("rollout.mp4")

# Success tracking
success = input("Did rollout succeed? (y/n or 0-100)")
df = df.append({"success": success, "duration": t_step, "video": filename})
df.to_csv("results/eval_timestamp.csv")

# Interactive evaluation
while True:
    # Run episode
    if input("Do one more eval?").lower() != "y":
        break
```

#### ALOHA (Framework handles evaluation)
- Evaluation metrics handled by Runtime framework
- No explicit success tracking in main script

### 6. **Control Frequency**

#### DROID
- **15 Hz** (matches dataset collection frequency)
- Manual timing: `time.sleep(1/15 - elapsed_time)`

#### ALOHA  
- **50 Hz** (specified in Runtime)
- Framework handles timing

### 7. **Error Handling**

#### DROID
```python
# Sophisticated interrupt handling
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    # Delays Ctrl+C during server calls
    
# Graceful error handling
try:
    # Control loop
except KeyboardInterrupt:
    break
```

#### ALOHA
- Error handling done by Runtime framework

## SO-100 Implementation Recommendations

Based on this comparison, for SO-100 you should choose:

### **Approach 1: DROID-style (Recommended for learning)**
- More explicit control over everything
- Easier to debug and understand
- Better for custom robots like SO-100
- Full control over action processing

### **Approach 2: ALOHA-style (For production)**
- Cleaner, more maintainable
- Requires implementing SO-100 environment class
- Better for standardized deployments

## SO-100 Implementation Template (DROID-style)

```python
@dataclasses.dataclass
class SO100Args:
    # Camera parameters
    front_camera_id: int = 0
    wrist_camera_id: int = 1
    
    # Robot parameters  
    robot_port: str = "/dev/ttyACM0"
    robot_type: str = "so100_follower"
    
    # Control parameters
    max_timesteps: int = 500
    open_loop_horizon: int = 8  # Match DROID
    control_frequency: int = 30  # SO-100 recommended freq
    
    # Server parameters
    remote_host: str = "localhost"
    remote_port: int = 8000

def main(args: SO100Args):
    # Initialize SO-100 robot (using LeRobot)
    robot_config = RobotConfig(
        type=args.robot_type,
        port=args.robot_port,
        cameras={
            "front": {"type": "opencv", "index": args.front_camera_id},
            "wrist": {"type": "opencv", "index": args.wrist_camera_id}
        }
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()
    
    # Connect to policy server
    client = WebsocketClientPolicy(args.remote_host, args.remote_port)
    
    # DROID-style control loop
    while True:
        instruction = input("Enter task: ")
        actions_completed = 0
        action_chunk = None
        
        for step in range(args.max_timesteps):
            # Get observation from robot
            obs = robot.get_observation()
            
            # Get new action chunk when needed
            if actions_completed == 0 or actions_completed >= args.open_loop_horizon:
                request = {
                    "observation/image": resize_image(obs["front"]),
                    "observation/wrist_image": resize_image(obs["wrist"]),
                    "observation/state": get_joint_positions(obs),  # 6 DOF
                    "prompt": instruction
                }
                
                action_chunk = client.infer(request)["actions"]  # Shape: (16, 6)
                actions_completed = 0
            
            # Execute single action (6 DOF: 5 joints + 1 gripper)
            action = action_chunk[actions_completed]
            robot.send_action(dict(zip(robot.motor_names, action)))
            actions_completed += 1
            
            # Maintain control frequency
            time.sleep(1.0 / args.control_frequency)
```

## Key Takeaways for SO-100

1. **Use DROID pattern** for explicit control and learning
2. **6 DOF actions** instead of DROID's 8 DOF  
3. **30Hz control** instead of DROID's 15Hz
4. **LeRobot integration** for robot interface
5. **Simple camera setup** (front + wrist cameras)
6. **Follow DROID's chunking pattern** but adapt dimensions