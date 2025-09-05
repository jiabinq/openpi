# SO-100 Robot Evaluation Guide for OpenPi

This guide provides detailed instructions for setting up and running evaluation on SO-100 robot arms using OpenPi models, based on NVIDIA GR00T's implementation approach.

## Overview

The SO-100 evaluation setup involves:
1. A GPU server running the OpenPi inference service
2. A robot control computer running the evaluation client
3. Communication via WebSocket protocol

## Architecture Comparison

### NVIDIA GR00T Approach
- **Server**: ZMQ-based server with optional HTTP support
- **Client**: LeRobot integration with custom policy wrapper
- **Data Config**: Specific SO-100 modality configuration
- **Key Features**: 
  - Action chunking (16 steps)
  - Multi-camera support
  - Language instruction support

### OpenPi Adaptation
- **Server**: WebSocket-based server (already implemented)
- **Client**: Minimal client with action chunking
- **Policy**: Custom SO-100 policy class needed

## Step 1: Create SO-100 Policy Class

Create `src/openpi/policies/so100_policy.py`:

```python
import numpy as np
from openpi.policies.base_policy import Policy
from openpi.common.preprocess_transforms import PreprocessConfig
from openpi.policies.action_transforms import (
    SO100Inputs, SO100Outputs  # Need to implement these
)

class SO100Policy(Policy):
    """Policy for SO-100 robot arm."""
    
    # Based on GR00T's SO-100 config
    ROBOT_STATE_KEYS = [
        'shoulder_pan.pos', 
        'shoulder_lift.pos', 
        'elbow_flex.pos', 
        'wrist_flex.pos', 
        'wrist_roll.pos', 
        'gripper.pos'
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocess = PreprocessConfig(
            rgb_scale=255.0,
            rgb_mean=np.array([0.485, 0.456, 0.406]),
            rgb_std=np.array([0.229, 0.224, 0.225]),
        )
    
    def _init_action_transform(self, **kwargs):
        return SO100Outputs(**kwargs)
    
    def _init_observation_transform(self, **kwargs):
        return SO100Inputs(**kwargs)
```

## Step 2: Add SO-100 Training Configuration

Add to `src/openpi/training/config.py`:

```python
@dataclass
class LeRobotSO100DataConfig(LeRobotDatasetConfig):
    """Configuration for SO-100 datasets."""
    
    def __post_init__(self):
        # SO-100 specific settings
        self.image_keys = ["observation/image", "observation/wrist_image"]
        self.state_keys = ["observation/state"]
        self.action_keys = ["action"]
        
        # Normalization asset paths
        self.asset_ids = {
            "observation.state": "so100",
            "action": "so100",
        }
        
        super().__post_init__()

# Add SO-100 configs to CONFIG_TABLE
CONFIG_TABLE.update({
    "pi0_so100": TrainConfig(
        policy="SO100Policy",
        dataset=LeRobotSO100DataConfig(
            dataset_names=["your_so100_dataset"],
        ),
        optimizer=OptimizerConfig(
            lr=1e-4,
            weight_decay=0.0,
            betas=(0.9, 0.999),
        ),
        flow_matching=FlowMatchingConfig(
            num_denoising_steps=50,
            sigma_min=0.001,
        ),
        num_train_steps=50000,
        batch_size=64,
    ),
    
    "pi0_fast_so100": TrainConfig(
        policy="SO100Policy",
        model="Pi0Fast",
        dataset=LeRobotSO100DataConfig(
            dataset_names=["your_so100_dataset"],
        ),
        # ... rest of config
    ),
})
```

## Step 3: Create SO-100 Evaluation Client

Create `examples/so100/main.py`:

```python
import dataclasses
import logging
import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro

# Import SO-100 robot control (assuming LeRobot integration)
from lerobot.robots import make_robot_from_config, RobotConfig

@dataclasses.dataclass
class Args:
    # Server configuration
    host: str = "localhost"
    port: int = 8000
    
    # Robot configuration
    robot_type: str = "so100_follower"
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "so100_robot"
    
    # Camera configuration (format matches GR00T)
    camera_config: dict = dataclasses.field(default_factory=lambda: {
        "wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
        "front": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
    })
    
    # Policy parameters
    action_horizon: int = 16  # Match GR00T's action chunk size
    replan_steps: int = 8     # Execute 8 steps before replanning
    
    # Task configuration
    lang_instruction: str = "Pick up the object and place it in the bin"
    max_episode_steps: int = 600
    control_frequency: int = 30  # Hz

def main(args: Args):
    # Initialize robot
    robot_config = RobotConfig(
        type=args.robot_type,
        port=args.robot_port,
        id=args.robot_id,
        cameras=args.camera_config,
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()
    
    # Get robot state keys (should match SO-100 policy)
    robot_state_keys = list(robot._motors_ft.keys())
    logging.info(f"Robot state keys: {robot_state_keys}")
    
    # Initialize policy client
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Connected to server: {client.get_server_metadata()}")
    
    # Main control loop
    action_chunk = []
    actions_executed = 0
    
    for step in range(args.max_episode_steps):
        start_time = time.time()
        
        # Get observation
        obs = robot.get_observation()
        
        # Check if we need a new action chunk
        if actions_executed >= args.replan_steps or not action_chunk:
            # Prepare observation for policy
            policy_obs = {
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs["front"], 224, 224)
                ),
                "observation/wrist_image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs["wrist"], 224, 224)
                ),
                "observation/state": np.array([
                    obs[key] for key in robot_state_keys[:5]  # First 5 are joint positions
                ] + [obs[robot_state_keys[5]]]),  # Last is gripper
                "prompt": args.lang_instruction,
            }
            
            # Get new action chunk
            action_chunk = client.infer(policy_obs)["actions"]
            actions_executed = 0
            
            logging.info(f"Got new action chunk with shape: {action_chunk.shape}")
        
        # Execute current action
        current_action = action_chunk[actions_executed]
        actions_executed += 1
        
        # Convert action to robot command format
        action_dict = {
            robot_state_keys[i]: float(current_action[i])
            for i in range(len(robot_state_keys))
        }
        
        # Send action to robot
        robot.send_action(action_dict)
        
        # Maintain control frequency
        elapsed = time.time() - start_time
        if elapsed < 1.0 / args.control_frequency:
            time.sleep(1.0 / args.control_frequency - elapsed)
    
    # Cleanup
    robot.disconnect()
    logging.info("Evaluation complete")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
```

## Step 4: Create Action/Observation Transforms

Create transforms in `src/openpi/policies/action_transforms.py`:

```python
class SO100Inputs(Inputs):
    """Input transforms for SO-100 robot."""
    
    def forward(self, observation):
        # Process images
        images = []
        for key in ["observation/image", "observation/wrist_image"]:
            if key in observation:
                img = observation[key]
                # Ensure correct shape and type
                if img.ndim == 3:
                    img = img[np.newaxis, ...]  # Add batch dimension
                images.append(img)
        
        # Stack images if multiple cameras
        if len(images) > 1:
            observation["observation/image"] = np.concatenate(images, axis=-1)
        
        # Process state (5 joint positions + 1 gripper)
        state = observation.get("observation/state", np.zeros(6))
        
        return observation

class SO100Outputs(Outputs):
    """Output transforms for SO-100 robot."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SO-100 has 5 joints + 1 gripper
        self.action_dim = 6
        self.action_horizon = kwargs.get("action_horizon", 16)
    
    def forward(self, actions):
        # Ensure actions are in correct format
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {actions.shape[-1]}")
        
        # Clip actions to valid range
        actions = np.clip(actions, -1.0, 1.0)
        
        return {"actions": actions}
```

## Step 5: Running Evaluation

### Start the Inference Server (GPU Machine)

```bash
# For a fine-tuned SO-100 model
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_so100 \
  --policy.dir=path/to/your/so100/checkpoint

# Or use default prompts
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_so100 \
  --policy.dir=path/to/your/so100/checkpoint \
  --default_prompt="Pick up the object"
```

### Run the Evaluation Client (Robot Computer)

```bash
# Install dependencies
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .

# Run evaluation
python examples/so100/main.py \
  --host=192.168.1.100 \
  --port=8000 \
  --robot_port=/dev/ttyACM0 \
  --lang_instruction="Pick up the red block and place it in the bin" \
  --action_horizon=16 \
  --replan_steps=8
```

## Key Differences from GR00T

1. **Communication Protocol**: OpenPi uses WebSocket instead of ZMQ
2. **Action Format**: OpenPi returns action chunks directly, GR00T uses modality-based format
3. **Normalization**: OpenPi handles normalization on server side
4. **Camera Processing**: Both resize to 224x224, but preprocessing differs slightly

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Robot not found | Check USB connection and device permissions |
| Camera not detected | Verify camera indices with `v4l2-ctl --list-devices` |
| Action dimension mismatch | Ensure policy and robot configs match |
| Network timeout | Check firewall settings and network connectivity |

## Best Practices

1. **Camera Setup**: Ensure both wrist and external cameras have clear views
2. **Action Chunking**: Use 16-step chunks with 8-step replanning for smooth motion
3. **Control Frequency**: 30Hz is recommended for SO-100
4. **Safety**: Always have emergency stop ready during evaluation

## Example Dataset Structure

For training, SO-100 datasets should follow LeRobot format:
```
dataset/
├── meta.json
├── episode_0/
│   ├── observation.image.front_*.png
│   ├── observation.image.wrist_*.png
│   ├── observation.state.json
│   └── action.json
└── episode_1/
    └── ...
```

## References

- [NVIDIA GR00T SO-100 Implementation](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot SO-100 Support](https://github.com/huggingface/lerobot)
- [OpenPi Remote Inference Guide](./remote_inference.md)