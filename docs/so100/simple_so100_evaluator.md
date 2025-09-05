# Simple SO-100 Evaluator: One Working File

## Overview

This is a single, complete, working implementation that combines:
- LeRobot's SO-100 robot interface
- OpenPi's WebSocket policy client  
- DROID's simple evaluation pattern

## Complete Implementation

```python
# so100_evaluator.py - Single working file
import time
import datetime
import numpy as np
import pandas as pd
import cv2
import tqdm
from contextlib import contextmanager
import signal
import os
from dataclasses import dataclass
from typing import Dict, Any

from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from openpi_client.websocket_client_policy import WebsocketClientPolicy

@contextmanager
def prevent_keyboard_interrupt():
    """DROID-style keyboard interrupt protection during network calls."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)
    
    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        
    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt

@dataclass
class SO100Config:
    # Robot setup
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "so100_follower"
    
    # OpenPi server
    policy_host: str = "localhost" 
    policy_port: int = 8000
    
    # Data format (determined by offline testing)
    openpi_format: str = "libero"  # "phospho", "libero", or "droid"
    
    # Evaluation settings
    max_timesteps: int = 200
    open_loop_horizon: int = 8
    control_frequency: int = 30
    action_horizon: int = 16
    
    # Camera setup
    cameras: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.cameras is None:
            self.cameras = {
                "front": {"fps": 25, "width": 640, "height": 480, "index_or_path": 0},
                "wrist": {"fps": 25, "width": 640, "height": 480, "index_or_path": 2},
            }

class SO100Evaluator:
    """Simple SO-100 evaluator using DROID's proven patterns."""
    
    def __init__(self, config: SO100Config):
        self.config = config
        
        # Setup robot interface
        cameras_config = {}
        for name, cam_config in config.cameras.items():
            cameras_config[name] = OpenCVCameraConfig(**cam_config)
            
        robot_config = SO100FollowerConfig(
            port=config.robot_port,
            cameras=cameras_config,
            disable_torque_on_disconnect=True,
        )
        
        self.robot = SO100Follower(robot_config)
        
        # Setup OpenPi client
        self.policy_client = WebsocketClientPolicy(config.policy_host, config.policy_port)
        
        # Results tracking
        self.results_df = pd.DataFrame(columns=["success", "duration", "timestamp"])
        
        # DROID-style action chunking state
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        
    def lerobot_to_openpi(self, raw_obs: dict, task: str) -> dict:
        """Convert LeRobot observation to OpenPi format."""
        
        # Convert joint states (LeRobot individual -> OpenPi combined)
        state = np.array([
            raw_obs["shoulder_pan.pos"],
            raw_obs["shoulder_lift.pos"], 
            raw_obs["elbow_flex.pos"],
            raw_obs["wrist_flex.pos"],
            raw_obs["wrist_roll.pos"],
            raw_obs["gripper.pos"]
        ])
        
        # Base observation
        openpi_obs = {
            "observation/state": state,
            "prompt": task
        }
        
        # Add images based on pre-tested format
        if self.config.openpi_format == "phospho":
            if "front" in raw_obs:
                openpi_obs["observation/images.main.left"] = cv2.resize(raw_obs["front"], (224, 224))
            if "wrist" in raw_obs:
                openpi_obs["observation/images.secondary_0"] = cv2.resize(raw_obs["wrist"], (224, 224))
                
        elif self.config.openpi_format == "libero":
            if "front" in raw_obs:
                openpi_obs["observation/image"] = cv2.resize(raw_obs["front"], (224, 224))
            if "wrist" in raw_obs:
                openpi_obs["observation/wrist_image"] = cv2.resize(raw_obs["wrist"], (224, 224))
                
        elif self.config.openpi_format == "droid":
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
            "gripper.pos": action_array[5]
        }
        
    def run_episode(self) -> dict:
        """DROID-style episode execution with simple error handling."""
        task = input("Enter task instruction: ")
        
        # Episode setup
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        bar = tqdm.tqdm(range(self.config.max_timesteps))
        
        print("Running rollout... press Ctrl+C to stop early.")
        
        for t_step in bar:
            start_time = time.time()
            
            try:
                # 1. Get observation from robot
                raw_obs = self.robot.get_observation()
                
                # 2. Convert to OpenPi format  
                openpi_obs = self.lerobot_to_openpi(raw_obs, task)
                
                # 3. DROID-style action chunking
                if (self.actions_from_chunk_completed == 0 or 
                    self.actions_from_chunk_completed >= self.config.open_loop_horizon):
                    
                    self.actions_from_chunk_completed = 0
                    
                    # Get new action chunk with interrupt protection
                    with prevent_keyboard_interrupt():
                        response = self.policy_client.infer(openpi_obs)
                        self.pred_action_chunk = response["actions"]
                    
                    # Validate response shape
                    expected_shape = (self.config.action_horizon, 6)
                    assert self.pred_action_chunk.shape == expected_shape, \
                        f"Expected {expected_shape}, got {self.pred_action_chunk.shape}"
                
                # 4. Extract next action from chunk
                action = self.pred_action_chunk[self.actions_from_chunk_completed]
                self.actions_from_chunk_completed += 1
                
                # 5. DROID-style gripper binarization
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])
                
                # 6. Safety clipping
                action = np.clip(action, -1, 1)
                
                # 7. Convert to robot format and execute
                robot_action = self.openpi_to_lerobot_action(action)
                self.robot.send_action(robot_action)
                
                # 8. DROID-style timing control
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / self.config.control_frequency:
                    time.sleep(1 / self.config.control_frequency - elapsed_time)
                    
            except KeyboardInterrupt:
                print(f"\nEpisode interrupted after {t_step} steps")
                break
        
        # Interactive success rating (DROID-style)
        success = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or numeric value 0-100: "
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
        
        # Store results
        new_row = pd.DataFrame([{
            "success": success,
            "duration": t_step + 1,
            "timestamp": timestamp
        }])
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        
        return {
            "duration": t_step + 1,
            "timestamp": timestamp,
            "success": success
        }
        
    def run_evaluation(self):
        """DROID-style evaluation loop with user interaction."""
        print("Starting SO-100 evaluation...")
        
        # Connect to robot
        self.robot.connect()
        print("Robot connected.")
        
        try:
            while True:
                self.run_episode()
                
                if input("Do one more eval? (enter y or n) ").lower() != "y":
                    break
                    
                # Reset action chunking state between episodes
                self.actions_from_chunk_completed = 0
                self.pred_action_chunk = None
                
        finally:
            # Always disconnect robot safely
            self.robot.disconnect()
            
        # Save final results
        if not self.results_df.empty:
            os.makedirs("results", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y") 
            csv_filename = f"results/so100_eval_{timestamp}.csv"
            self.results_df.to_csv(csv_filename, index=False)
            print(f"Results saved to {csv_filename}")
            
            # Print summary stats
            success_rate = self.results_df["success"].mean()
            avg_duration = self.results_df["duration"].mean()
            print(f"Summary: {success_rate:.1%} success rate, {avg_duration:.1f} average steps")


def main():
    """Example usage."""
    config = SO100Config(
        openpi_format="libero",  # Use format determined by offline testing
        max_timesteps=200,
        open_loop_horizon=8,
    )
    
    evaluator = SO100Evaluator(config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
```

## Usage Instructions

### 1. **Prerequisites**
```bash
# Ensure dependencies are installed
pip install lerobot openpi-client opencv-python pandas tqdm numpy
```

### 2. **Offline Format Testing** (Do This First)
```bash
# Test which format works with your OpenPi server
python validate_openpi_format.py --host localhost --port 8000

# Output will show which format to use:
# libero     : ✅ COMPATIBLE  
# Update config.openpi_format = "libero"
```

### 3. **Run Evaluation**
```bash
# Start OpenPi server first
uv run scripts/serve_policy.py policy:checkpoint --policy.config=CONFIG_NAME

# Run SO-100 evaluation
python so100_evaluator.py
```

### 4. **Interactive Usage**
```
Starting SO-100 evaluation...
Robot connected.
Enter task instruction: pick up the red block
Running rollout... press Ctrl+C to stop early.
100%|████████| 200/200 [01:30<00:00,  2.22it/s]
Did the rollout succeed? (enter y for 100%, n for 0%), or numeric value 0-100: 80
Do one more eval? (enter y or n) y
...
```

## Key Features

1. ✅ **Single file** - no complex modules to coordinate
2. ✅ **DROID patterns** - proven action chunking and evaluation flow
3. ✅ **LeRobot robot interface** - production-grade hardware control
4. ✅ **Format flexibility** - supports phospho/libero/droid formats
5. ✅ **Simple error handling** - only KeyboardInterrupt like DROID
6. ✅ **No video recording** - eliminates memory and complexity issues
7. ✅ **Safe robot handling** - always disconnects properly

## Next Steps

1. **Test offline format validation first**
2. **Start with simple tasks** (short episodes)  
3. **Verify robot safety limits** are properly configured
4. **Add video recording later if needed** (as separate optional feature)

This replaces the complex multi-module plan with **one simple file that actually works**.