# Complete Evaluation Workflow for Physical Robots

This document describes the complete workflow for evaluating fine-tuned models on physical robots using the OpenPi framework.

## Overview

After fine-tuning a model and setting up an inference server, the evaluation process involves:
1. Starting an inference server on a GPU machine
2. Installing the client on the robot control computer
3. Running robot-specific evaluation scripts
4. Collecting metrics and videos

## 1. Start the Inference Server (GPU Machine)

The inference server hosts your trained model and serves predictions via WebSocket.

### For a fine-tuned model:
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=YOUR_CONFIG_NAME \
  --policy.dir=PATH_TO_YOUR_CHECKPOINT
```

### For pre-trained models:
```bash
uv run scripts/serve_policy.py --env=[ALOHA|DROID|LIBERO]
```

The server will start on port 8000 by default and display model metadata when ready.

## 2. Install Client on Robot (Robot Control Computer)

The OpenPi client package provides minimal dependencies for robot integration:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

## 3. Run Robot-Specific Evaluation Script

### For ALOHA Robot

**Files:**
- Main script: `examples/aloha_real/main.py`
- Environment: `examples/aloha_real/env.py`

**Commands:**
```bash
# Terminal 1: Run ROS nodes
roslaunch aloha ros_nodes.launch

# Terminal 2: Run evaluation
python -m examples.aloha_real.main --host=SERVER_IP --port=8000
```

**Features:**
- Uses Runtime framework for control loop orchestration
- 50Hz control frequency
- Automatic episode management

### For DROID Robot

**File:** `examples/droid/main.py`

**Command:**
```bash
python scripts/main.py \
  --remote_host=SERVER_IP \
  --remote_port=8000 \
  --external_camera=left
```

**Features:**
- Interactive task prompts
- Video recording of rollouts
- Success rate tracking with CSV export
- 15Hz control frequency (matching DROID dataset)

### For Custom Robots (e.g., SO-100)

To add support for a new robot:

1. **Create a policy class** in `src/openpi/policies/`:
   - Define input/output transforms
   - Handle robot-specific action spaces
   - Configure normalization

2. **Add training configs** to `src/openpi/training/config.py`:
   - Create dataset configuration
   - Define model variants (π₀ and π₀-FAST)

3. **Create evaluation script** based on existing examples:
   - WebSocket client connection
   - Observation preprocessing
   - Action execution loop
   - Metrics collection

## 4. Evaluation Loop Structure

The typical evaluation loop follows this pattern:

```python
# Initialize client
client = WebsocketClientPolicy(host=SERVER_IP, port=8000)

# Episode loop
for episode in range(num_episodes):
    observation = reset_environment()
    action_chunk = []
    
    for step in range(max_steps):
        # Prepare observation
        obs_dict = {
            "observation/image": resize_image(img, 224, 224),
            "observation/wrist_image": resize_image(wrist_img, 224, 224),
            "observation/state": proprioceptive_state,
            "prompt": task_instruction,
        }
        
        # Get new action chunk if needed
        if need_new_chunk():
            action_chunk = client.infer(obs_dict)["actions"]
        
        # Execute action
        action = action_chunk[current_index]
        robot.step(action)
        
    # Record success
    success = evaluate_success()
    save_metrics(success)
```

## 5. Key Configuration Parameters

### Action Execution
- **`action_horizon`**: Length of predicted action chunks (typically 10-25)
- **`open_loop_horizon`**: Steps to execute before replanning (typically 5-8)
- **`replan_steps`**: When to query for new action chunk

### Episode Control
- **`max_episode_steps`**: Maximum steps per episode
- **`num_episodes`**: Number of evaluation rollouts
- **`control_frequency`**: Robot control rate in Hz (15-50)

### Image Processing
- **`resize_size`**: Image size for model input (typically 224)
- **Image format**: uint8, RGB order
- **Preprocessing**: `resize_with_pad` to maintain aspect ratio

## 6. Output Files and Metrics

### Video Recording
- Format: MP4 using libx264 codec
- Naming: `video_YYYY_MM_DD_HH:MM:SS.mp4`
- Content: External camera view of rollout

### Metrics CSV
- Columns: `success`, `duration`, `video_filename`
- Location: `results/eval_TIMESTAMP.csv`
- Success values: 0.0-1.0 (allows partial success)

### Timing Statistics
- Client inference time
- Server processing time
- Policy forward pass time
- Network latency

## 7. Best Practices

### Network Setup
- Use wired connection for minimal latency
- Expect 0.5-1s latency per action chunk
- Ensure firewall allows WebSocket connections

### Camera Configuration
- Verify camera IDs before running
- Ensure all task-relevant objects are visible
- Check lighting conditions

### Error Handling
- The client automatically reconnects on connection loss
- Action chunks provide temporal consistency
- Keyboard interrupts are handled gracefully

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| Cannot reach server | Check IP/port, firewall settings, use `ping` |
| Slow inference | Use wired connection, check GPU utilization |
| Camera not found | Verify camera IDs, replug USB connections |
| Policy fails | Adjust scene setup, check camera views |

## Example: Complete DROID Evaluation

```bash
# On GPU server
uv run scripts/serve_policy.py --env=DROID

# On DROID laptop
cd $DROID_ROOT
python scripts/main.py \
  --remote_host=192.168.1.100 \
  --remote_port=8000 \
  --external_camera=left \
  --max_timesteps=600 \
  --open_loop_horizon=8
```

This will start an interactive evaluation session where you can enter language instructions and observe the robot's performance.