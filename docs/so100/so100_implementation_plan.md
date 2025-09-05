# SO-100 Implementation Plan v2: Improved Architecture with Module Interactions

Based on comprehensive analysis of OpenPi DROID, ALOHA, LeRobot, and GR00T approaches, this plan integrates the best practices from each reference.

## Architecture Overview

```
┌─────────────────┬─────────────────────┬─────────────────────┐
│  ROBOT CONFIG   │    DATA FORMAT      │ EVALUATION FEATURES │
│  (LeRobot)      │    (OpenPi)         │    (DROID-style)    │
├─────────────────┼─────────────────────┼─────────────────────┤
│ • SO-100 setup  │ • Server interface  │ • Action chunking   │
│ • Camera config │ • observation/      │ • Video recording   │
│ • Safety limits │ • Format validation │ • CSV metrics       │
│ • Connection    │ • Type conversion   │ • Error handling    │
└─────────────────┴─────────────────────┴─────────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ ORCHESTRATION     │
                    │ (Main Loop)       │
                    │ • Module coord.   │
                    │ • Data flow       │
                    │ • Error recovery  │
                    └───────────────────┘
```

## Reference Quality Assessment

| Reference | Robot Interface | Data Format | Eval Features | Overall |
|-----------|----------------|-------------|---------------|---------|
| **LeRobot record.py** | ✅ **Superior** (402 lines, production) | ❌ Own format | ✅ **Superior** (comprehensive) | **Best** |
| **DROID main.py** | ⚠️ Custom (60 lines) | ✅ **OpenPi compatible** | ✅ **Explicit** (video, CSV, chunking) | **Good** |
| **ALOHA main.py** | ⚠️ Framework (hidden) | ✅ OpenPi compatible | ❌ Hidden in framework | Poor |
| **GR00T eval_lerobot.py** | ⚠️ Basic (40 lines) | ❌ GR00T format | ⚠️ Basic evaluation | Poor |
| **Phospho SO-100** | ❌ Transform only | ✅ **OpenPi compatible** | ❌ No evaluation | Transform |

## Module Implementation Strategy

### 1. **Robot Interface Module** (Use LeRobot)
**Winner: LeRobot** - Most comprehensive and production-ready

```python
# Use LeRobot's superior robot interface
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

class SO100RobotInterface:
    def __init__(self, config: SO100EvalConfig):
        # LeRobot's comprehensive configuration
        cameras_config = {}
        for name, cam_config in config.cameras.items():
            cameras_config[name] = OpenCVCameraConfig(**cam_config)
            
        robot_config = SO100FollowerConfig(
            port=config.robot_port,
            id=config.robot_id,
            cameras=cameras_config,
            disable_torque_on_disconnect=True,  # LeRobot safety feature
            max_relative_target=config.max_motion,  # LeRobot safety limits
        )
        
        self.robot = SO100Follower(robot_config)  # LeRobot's production robot
        self.camera_names = list(config.cameras.keys())
        
    def connect(self):
        """Use LeRobot's robust connection handling."""
        self.robot.connect()
        
    def get_raw_observation(self):
        """Get observation in LeRobot's native format."""
        return self.robot.get_observation()  # LeRobot handles all hardware details
        
    def send_action(self, action_dict):
        """Send action using LeRobot's interface."""
        self.robot.send_action(action_dict)  # LeRobot handles robot communication
```

### 2. **Data Format Module** (OpenPi Compatible)
**Use Tested Format with Config-based Selection**

```python
class DataFormatConverter:
    """Convert between LeRobot format and OpenPi format using config-based selection."""
    
    def __init__(self, config: SO100EvalConfig):
        # Use configuration to determine format - no runtime validation
        self.use_format = config.openpi_format  # "phospho", "libero", or "droid"
        
    def lerobot_to_openpi(self, raw_obs: dict, task: str) -> dict:
        """Convert LeRobot observation to OpenPi format based on config."""
        
        # Joint state conversion (LeRobot individual → OpenPi combined)
        state = np.array([
            raw_obs["shoulder_pan.pos"],     # LeRobot format
            raw_obs["shoulder_lift.pos"],
            raw_obs["elbow_flex.pos"],
            raw_obs["wrist_flex.pos"],
            raw_obs["wrist_roll.pos"],
            raw_obs["gripper.pos"],
        ])
        
        openpi_obs = {
            "observation/state": state,  # Works for all formats (same as LIBERO)
            "prompt": task
        }
        
        # Fixed camera mapping (determined by offline format testing)
        # SO-100 typically has: front camera + wrist camera
        
        if self.use_format == "phospho":
            if "front" in raw_obs:
                openpi_obs["observation/images.main.left"] = cv2.resize(raw_obs["front"], (224, 224))
            if "wrist" in raw_obs:
                openpi_obs["observation/images.secondary_0"] = cv2.resize(raw_obs["wrist"], (224, 224))
                
        elif self.use_format == "libero":
            if "front" in raw_obs:
                openpi_obs["observation/image"] = cv2.resize(raw_obs["front"], (224, 224))
            if "wrist" in raw_obs:
                openpi_obs["observation/wrist_image"] = cv2.resize(raw_obs["wrist"], (224, 224))
                
        elif self.use_format == "droid":
            if "front" in raw_obs:
                openpi_obs["observation/exterior_image_1_left"] = cv2.resize(raw_obs["front"], (224, 224))
            if "wrist" in raw_obs:
                openpi_obs["observation/wrist_image_left"] = cv2.resize(raw_obs["wrist"], (224, 224))
                        
        return openpi_obs
        
    def openpi_to_lerobot_action(self, action_array: np.ndarray) -> dict:
        """Convert OpenPi action array to LeRobot action dict."""
        return {
            "shoulder_pan.pos": action_array[0],
            "shoulder_lift.pos": action_array[1],
            "elbow_flex.pos": action_array[2],
            "wrist_flex.pos": action_array[3],
            "wrist_roll.pos": action_array[4],
            "gripper.pos": action_array[5],
        }
```

### **Offline Format Validation Tool**

```python
# scripts/validate_openpi_format.py
"""Offline tool to test which format works with your OpenPi server."""

import argparse
import numpy as np
from openpi_client import WebsocketClientPolicy

def test_format(client, format_name: str, format_obs: dict) -> bool:
    """Test if a specific format works with the server."""
    print(f"\nTesting {format_name} format...")
    try:
        response = client.infer(format_obs)
        if "actions" in response and response["actions"].shape == (16, 6):
            print(f"✅ {format_name} format works!")
            return True
    except Exception as e:
        print(f"❌ {format_name} format failed: {e}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Test OpenPi format compatibility")
    parser.add_argument("--host", default="localhost", help="OpenPi server host")
    parser.add_argument("--port", type=int, default=8000, help="OpenPi server port")
    args = parser.parse_args()
    
    # Connect to server
    client = WebsocketClientPolicy(args.host, args.port)
    print(f"Connected to OpenPi server at {args.host}:{args.port}")
    
    # Prepare test data
    dummy_state = np.random.rand(6).astype(np.float32)
    dummy_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    
    # Test formats
    formats = {
        "phospho": {
            "observation/state": dummy_state,
            "observation/images.main.left": dummy_image,
            "observation/images.secondary_0": dummy_image,
            "prompt": "test task"
        },
        "libero": {
            "observation/state": dummy_state,
            "observation/image": dummy_image,
            "observation/wrist_image": dummy_image,
            "prompt": "test task"
        },
        "droid": {
            "observation/joint_position": dummy_state[:5],
            "observation/gripper_position": dummy_state[5:6],
            "observation/exterior_image_1_left": dummy_image,
            "observation/wrist_image_left": dummy_image,
            "prompt": "test task"
        }
    }
    
    # Test each format
    results = {}
    for name, obs in formats.items():
        results[name] = test_format(client, name, obs)
    
    # Summary
    print("\n" + "="*50)
    print("FORMAT COMPATIBILITY RESULTS:")
    print("="*50)
    for name, success in results.items():
        status = "✅ COMPATIBLE" if success else "❌ INCOMPATIBLE"
        print(f"{name:10s}: {status}")
    
    # Recommendation
    working_formats = [name for name, success in results.items() if success]
    if working_formats:
        print(f"\nRecommended format: {working_formats[0]}")
        print(f"Add to your config: openpi_format = '{working_formats[0]}'")
    else:
        print("\n⚠️ No formats worked! Check your server configuration.")

if __name__ == "__main__":
    main()
```

### 3. **Evaluation Features Module** (DROID-style)
**Winner: DROID's Explicit Implementation** - Full control and rich features

```python
class EvaluationFeatures:
    """DROID-style evaluation features with full control."""
    
    def __init__(self, config: SO100EvalConfig):
        self.config = config
        self.results_df = pd.DataFrame(columns=["success", "duration", "video_filename"])
        self.video_frames = []
        
        # Action chunking tracking (from DROID)
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        
    def manage_action_chunking(self, openpi_obs: dict, policy_client) -> np.ndarray:
        """DROID-style manual action chunking with full visibility."""
        # Check if we need a new action chunk
        if (self.actions_from_chunk_completed == 0 or 
            self.actions_from_chunk_completed >= self.config.open_loop_horizon):
            
            self.actions_from_chunk_completed = 0
            
            # Get new chunk from OpenPi server (with error protection)
            with prevent_keyboard_interrupt():
                response = policy_client.infer(openpi_obs)
                self.pred_action_chunk = response["actions"]
                
                # Validate OpenPi format
                expected_shape = (self.config.action_horizon, 6)
                assert self.pred_action_chunk.shape == expected_shape, \
                    f"Expected {expected_shape}, got {self.pred_action_chunk.shape}"
        
        # Extract next action from chunk
        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1
        
        # Process gripper action (DROID-style binary threshold)
        if action[-1] > 0.5:
            action[-1] = 1.0
        else:
            action[-1] = 0.0
            
        # Safety clipping
        action = np.clip(action, -1.0, 1.0)
        return action
        
    def record_video_frame(self, openpi_obs: dict):
        """Record video frame from OpenPi observation."""
        if self.config.export_video and "observation/images.main.left" in openpi_obs:
            self.video_frames.append(openpi_obs["observation/images.main.left"].copy())
            
    def save_episode_results(self, episode_duration: int) -> str:
        """DROID-style interactive result saving."""
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        video_filename = None
        
        # Save video (DROID approach)
        if self.config.export_video and self.video_frames:
            video_filename = f"video_so100_{timestamp}"
            try:
                video_array = np.stack(self.video_frames)
                ImageSequenceClip(list(video_array), fps=self.config.fps_video).write_videofile(
                    video_filename + ".mp4", codec="libx264", verbose=False, logger=None
                )
                print(f"Video saved: {video_filename}.mp4")
            except Exception as e:
                print(f"Video save failed: {e}")
                video_filename = None
                
        # Interactive success rating (DROID approach)
        success = None
        if self.config.export_csv:
            while success is None:
                user_input = input("Episode success rating (y/n/0-100): ").strip().lower()
                if user_input in ["y", "yes"]:
                    success = 1.0
                elif user_input in ["n", "no"]:
                    success = 0.0
                else:
                    try:
                        success = float(user_input) / 100.0
                        success = np.clip(success, 0.0, 1.0)
                    except ValueError:
                        print("Please enter 'y', 'n', or a number 0-100")
                        
            # Add to results dataframe
            new_row = pd.DataFrame([{
                "success": success,
                "duration": episode_duration,
                "video_filename": video_filename or "",
                "action_chunks_used": self.actions_from_chunk_completed,
            }])
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            
        # Reset for next episode
        self.video_frames = []
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        
        return video_filename or f"episode_{timestamp}"
        
    def export_csv_results(self):
        """DROID-style CSV export with comprehensive metrics."""
        if not self.results_df.empty:
            os.makedirs("results", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
            csv_filename = os.path.join("results", f"so100_eval_{timestamp}.csv")
            self.results_df.to_csv(csv_filename, index=False)
            print(f"Results saved to {csv_filename}")
            
            # Print summary stats
            success_rate = self.results_df["success"].mean()
            avg_duration = self.results_df["duration"].mean()
            print(f"Summary: {success_rate:.1%} success rate, {avg_duration:.1f} average steps")
```

## Main Orchestration Loop (DROID-Style Simple Error Handling)

```python
class SO100Evaluator:
    """Main orchestration using DROID's simple, proven approach."""
    
    def __init__(self, config: SO100EvalConfig):
        # Initialize components (like DROID's simple setup)
        self.robot = SO100Follower(config.robot_config)  # LeRobot robot interface
        self.policy_client = WebsocketClientPolicy(config.policy_host, config.policy_port)  # OpenPi client
        self.format_converter = DataFormatConverter(config)  # Format conversion
        self.results_df = pd.DataFrame(columns=["success", "duration", "timestamp"])  # DROID-style results (no video)
        
        # DROID-style action chunking state
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        
    def run_episode(self) -> dict:
        """DROID-style simple evaluation loop with single try-catch."""
        task = input("Enter task instruction: ")  # DROID-style user input
        
        # DROID-style episode setup (simplified - no video recording)
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        bar = tqdm.tqdm(range(self.config.max_timesteps))
        
        print("Running rollout... press Ctrl+C to stop early.")
        
        for t_step in bar:
            start_time = time.time()
            
            try:
                # 1. Get observation (same as DROID pattern)
                raw_obs = self.robot.get_observation()
                
                # 2. Convert to OpenPi format (our addition)
                openpi_obs = self.format_converter.lerobot_to_openpi(raw_obs, task)
                
                # 3. DROID-style action chunking
                if self.actions_from_chunk_completed == 0 or self.actions_from_chunk_completed >= self.config.open_loop_horizon:
                    self.actions_from_chunk_completed = 0
                    
                    # DROID-style interrupt protection
                    with prevent_keyboard_interrupt():
                        response = self.policy_client.infer(openpi_obs)
                        self.pred_action_chunk = response["actions"]
                    
                    # Validate response (like DROID's assert)
                    expected_shape = (self.config.action_horizon, 6)
                    assert self.pred_action_chunk.shape == expected_shape
                
                # 4. Extract action from chunk (DROID pattern)
                action = self.pred_action_chunk[self.actions_from_chunk_completed]
                self.actions_from_chunk_completed += 1
                
                # 5. DROID-style gripper binarization
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])
                
                # 6. Safety clipping (DROID pattern)
                action = np.clip(action, -1, 1)
                
                # 7. Convert to robot format and send (our addition)
                robot_action = self.format_converter.openpi_to_lerobot_action(action)
                self.robot.send_action(robot_action)
                
                # 8. DROID-style timing control
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / self.config.control_frequency:
                    time.sleep(1 / self.config.control_frequency - elapsed_time)
                    
            except KeyboardInterrupt:
                # DROID-style simple exit on Ctrl+C
                break
        
        # DROID-style interactive success rating
        success = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100: "
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0
            else:
                try:
                    success = float(success) / 100
                    if not (0 <= success <= 1):
                        print(f"Success must be in [0, 100] but got: {success * 100}")
                        success = None
                except ValueError:
                    success = None
        
        # DROID-style results storage (simplified - no video)
        self.results_df = self.results_df.append({
            "success": success,
            "duration": t_step,
            "timestamp": timestamp,
        }, ignore_index=True)
        
        return {
            "duration": t_step,
            "timestamp": timestamp,
            "success": success
        }
        
    def run_evaluation(self):
        """DROID-style evaluation loop with user interaction."""
        while True:
            self.run_episode()
            
            if input("Do one more eval? (enter y or n) ").lower() != "y":
                break
            
            # Reset robot (like DROID)
            self.robot.disconnect()
            self.robot.connect()
        
        # DROID-style CSV export
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
        csv_filename = os.path.join("results", f"so100_eval_{timestamp}.csv")
        self.results_df.to_csv(csv_filename)
        print(f"Results saved to {csv_filename}")
        
        # Print summary stats
        if not self.results_df.empty:
            success_rate = self.results_df["success"].mean()
            avg_duration = self.results_df["duration"].mean()
            print(f"Summary: {success_rate:.1%} success rate, {avg_duration:.1f} average steps")
```

## Module Interaction Summary

### **Data Flow Between Modules**

```
Robot Interface (LeRobot) 
    │ raw_observation (LeRobot format)
    ▼
Data Format Converter
    │ openpi_observation (OpenPi format) 
    ▼
Evaluation Features ←→ Policy Client (OpenPi WebSocket)
    │ action_array (numpy)
    ▼  
Data Format Converter
    │ robot_action (LeRobot format)
    ▼
Robot Interface (LeRobot)
```

### **Error Handling Strategy (DROID-Style Simplicity)**

**DROID's Simple Approach**: Single try-catch for `KeyboardInterrupt` only
- ✅ **Proven to work** in DROID production evaluation
- ✅ **Easy to debug** - clear failure point
- ✅ **Fail fast** - no complex recovery logic
- ✅ **Safe** - user can always Ctrl+C to stop robot

**Applied to SO-100**:
1. **Single try-catch**: Only handle `KeyboardInterrupt` like DROID
2. **Fail fast**: Let other errors bubble up for immediate debugging
3. **User control**: Always allow Ctrl+C to stop robot safely
4. **No complex recovery**: Avoid multi-module error coordination
5. **Restart on failure**: Simple restart approach like DROID's `env.reset()`

### **Configuration Dependencies**

```python
@dataclass
class SO100EvalConfig:
    # Robot Interface Config (LeRobot-based)
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "so100_follower"
    cameras: Dict[str, Dict] = field(default_factory=lambda: {
        "front": {"fps": 25, "width": 640, "height": 480, "index_or_path": 0},
        "wrist": {"fps": 25, "width": 640, "height": 480, "index_or_path": 2},
    })
    
    # Data Format Config (OpenPi requirements)
    openpi_format: str = "libero"  # "phospho", "libero", or "droid" - test offline first!
    image_size: Tuple[int, int] = (224, 224)
    state_dim: int = 6  # SO-100 DOF
    action_horizon: int = 16  # OpenPi default
    
    # Evaluation Config (DROID-style, simplified)
    max_timesteps: int = 200
    open_loop_horizon: int = 8
    control_frequency: int = 30
    export_csv: bool = True
    
    # Policy Config (OpenPi server)
    policy_host: str = "localhost"
    policy_port: int = 8000
    task_instruction: str = "pick up the object"
```

## Implementation Timeline: 4-6 Days

### **Phase 0: Offline Format Testing (0.5 days)**
1. **Run offline format validation tool** against OpenPi server
2. **Test all three formats** (phospho, libero, droid)
3. **Select working format** and add to configuration
4. **Document which format works** for your server version

### **Phase 1: Module Structure (1-2 days)**
1. Create module classes with proper interfaces
2. Implement LeRobot robot interface integration  
3. Setup OpenPi data format conversion with **config-based selection**
4. Basic configuration management

### **Phase 2: Core Integration (1-2 days)**  
1. Implement DROID-style main loop with **pre-tested format**
2. Add DROID-style action chunking (proven to work)
3. Setup OpenPi policy client communication
4. **Simple error handling**: Only `KeyboardInterrupt` like DROID

### **Phase 3: Evaluation Features (1-2 days)**
1. ~~Add DROID-style video recording~~ **Start without video** (reduces risk and complexity)
2. Implement DROID-style CSV export with metrics  
3. DROID-style interactive success rating system
4. **No complex error recovery** - keep it simple like DROID

### **Phase 4: Testing & Optimization (1 day)**
1. Integration testing with real hardware
2. Performance optimization  
3. Documentation and examples

## Format Testing Strategy

### **Step 1: Run Offline Validation**
```bash
# Test which format works with your OpenPi server
python scripts/validate_openpi_format.py --host localhost --port 8000

# Output:
# phospho    : ❌ INCOMPATIBLE
# libero     : ✅ COMPATIBLE  
# droid      : ✅ COMPATIBLE
# 
# Recommended format: libero
```

### **Step 2: Update Configuration**
```python
# config.yaml or in code
openpi_format: "libero"  # Use the format that passed validation
```

### **Step 3: Run Evaluation**
```python
# No runtime validation needed - format already tested
evaluator = SO100Evaluator(config)
evaluator.run_episodes()  # Uses pre-tested format
```

## Benefits of Offline Testing

1. **No runtime failures**: Format validated before robot operation
2. **Simple implementation**: No complex validation logic in production code
3. **Clear configuration**: Explicit format selection in config
4. **Easy debugging**: Test formats independently of robot hardware

## Key Advantages

1. **Best-in-Class Components**: LeRobot robot interface + DROID evaluation features + OpenPi compatibility
2. **DROID-Style Simplicity**: Single try-catch, simple error handling, proven approach
3. **No Video Recording**: Reduces memory usage, eliminates OOM risk, fewer dependencies
4. **Fail Fast**: Immediate debugging, no complex error recovery to mask issues  
5. **Production Ready**: Built on DROID's proven evaluation patterns
6. **Offline Format Testing**: Test formats before deployment, no runtime failures
7. **Simple Configuration**: Explicit format selection, no complex validation logic
8. **User Control**: Always allow Ctrl+C to stop robot safely like DROID

This architecture follows **DROID's core simplicity** - removing even DROID's video recording complexity to focus on reliable robot control. The implementation adds only the necessary components (LeRobot robot interface + format conversion) to bridge to OpenPi servers.

## Optional Enhancement: Add Video Recording Later

If video recording is needed later, add it as **LeRobot-style individual frame saving**:

```python
# Optional addition - save frames individually (no memory accumulation)
if self.config.save_frames:
    frame_dir = f"frames/episode_{timestamp}"
    os.makedirs(frame_dir, exist_ok=True)
    frame_path = f"{frame_dir}/frame_{t_step:04d}.png"
    cv2.imwrite(frame_path, openpi_obs["observation/images.main.left"])
```

This avoids DROID's memory accumulation issues while still providing visual debugging capability.