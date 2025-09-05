# SO-100 Evaluation with OpenPi

Simple SO-100 robot evaluation using OpenPi remote inference, combining:
- **LeRobot's production-grade SO-100 robot interface**
- **OpenPi's WebSocket policy server** 
- **DROID's proven evaluation patterns**

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install lerobot openpi-client opencv-python pandas tqdm numpy

# Hardware requirements:
# - SO-100 robot arm connected via USB (/dev/ttyACM0)  
# - 2 cameras (front and wrist views)
# - OpenPi policy server running
```

### 2. Test Format Compatibility (IMPORTANT - Do This First)

```bash
# Start your OpenPi server first
uv run scripts/serve_policy.py policy:checkpoint --policy.config=CONFIG_NAME

# Test which data format works with your server
python validate_openpi_format.py --host localhost --port 8000

# Expected output:
# phospho    : ❌ INCOMPATIBLE
# libero     : ✅ COMPATIBLE  
# droid      : ✅ COMPATIBLE
# 
# Recommended format: libero
```

### 3. Update Configuration

Edit the format in `so100_evaluator.py` based on test results:

```python
config = SO100Config(
    openpi_format="libero",  # Use the format that passed testing
    max_timesteps=200,
    open_loop_horizon=8,
)
```

### 4. Run Evaluation

```bash
# Make sure OpenPi server is still running
# Run SO-100 evaluation
python so100_evaluator.py
```

### 5. Interactive Usage

```
Starting SO-100 evaluation...
Robot connected.
Enter task instruction: pick up the red block
Running rollout... press Ctrl+C to stop early.
100%|████████| 200/200 [01:30<00:00,  2.22it/s]
Did the rollout succeed? (enter y for 100%, n for 0%), or numeric value 0-100: 80
Do one more eval? (enter y or n) y
Enter task instruction: place it in the box
...
Do one more eval? (enter y or n) n
Results saved to results/so100_eval_02_15PM_January_15_2025.csv
Summary: 75.0% success rate, 180.5 average steps
```

## Configuration Options

```python
@dataclass
class SO100Config:
    # Robot hardware
    robot_port: str = "/dev/ttyACM0"        # SO-100 USB port
    robot_id: str = "so100_follower"         # Robot identifier
    
    # OpenPi server  
    policy_host: str = "localhost"           # Server IP address
    policy_port: int = 8000                  # Server port
    
    # Data format (test with validate_openpi_format.py)
    openpi_format: str = "libero"            # "phospho", "libero", or "droid"
    
    # Evaluation settings
    max_timesteps: int = 200                 # Max steps per episode
    open_loop_horizon: int = 8               # Action chunk size (DROID default)
    control_frequency: int = 30              # Control loop frequency (Hz)
    action_horizon: int = 16                 # OpenPi action sequence length
    
    # Camera setup
    cameras: Dict[str, Dict] = {
        "front": {"fps": 25, "width": 640, "height": 480, "index_or_path": 0},
        "wrist": {"fps": 25, "width": 640, "height": 480, "index_or_path": 2},
    }
```

## Key Features

- ✅ **Single file implementation** - no complex modules to coordinate
- ✅ **DROID-style patterns** - proven action chunking and evaluation flow  
- ✅ **LeRobot robot interface** - production-grade hardware control
- ✅ **Format flexibility** - supports phospho/libero/droid formats
- ✅ **Simple error handling** - only KeyboardInterrupt like DROID
- ✅ **No video recording** - eliminates memory and complexity issues
- ✅ **Safe robot handling** - always disconnects properly
- ✅ **CSV results export** - detailed evaluation metrics

## Architecture

```
SO-100 Hardware ←→ LeRobot Interface ←→ Format Converter ←→ OpenPi WebSocket Server
     │                    │                    │                      │
   Robot Control     Observation/Action    Data Format         Policy Inference
   (servo motors,      (joint states,      Conversion         (π₀, π₀-FAST, etc)
    cameras)           camera images)      (LeRobot ↔ OpenPi)
```

## Safety Features

1. **Hardware safety**: LeRobot's built-in torque limits and disconnect protection
2. **Motion safety**: Action clipping to [-1, 1] range  
3. **User control**: Always allow Ctrl+C to stop robot safely
4. **Safe shutdown**: Automatic robot disconnection on exit

## Troubleshooting

### Format Testing Fails
```bash
❌ All formats failed: Connection refused
```
**Solution**: Make sure OpenPi server is running first

### Robot Connection Fails  
```bash
ConnectionError: Could not connect to /dev/ttyACM0
```
**Solution**: Check USB connection and permissions:
```bash
ls -la /dev/ttyACM*
sudo chmod 666 /dev/ttyACM0  # If needed
```

### Camera Not Found
```bash
OpenCV Error: Camera index 0 not found
```
**Solution**: Check camera connections and update camera indices in config

### Network Timeout
```bash
TimeoutError: Policy server did not respond
```
**Solution**: 
- Check server is running and responsive
- Verify host/port configuration
- Try reducing action_horizon or increasing timeout

## Comparison with Other Approaches

| Feature | SO-100 (This) | DROID | ALOHA | LeRobot Direct |
|---------|---------------|-------|--------|----------------|
| **Robot Interface** | LeRobot (superior) | Custom | Framework | LeRobot (same) |
| **Policy Server** | OpenPi (remote) | OpenPi (remote) | OpenPi (remote) | Local inference |
| **Error Handling** | Simple (DROID-style) | Simple | Hidden | Simple |
| **Memory Usage** | Low (no video) | High (video) | Unknown | Low |
| **Setup Complexity** | Medium | Medium | High | Low |
| **Scalability** | High (remote) | High (remote) | High (remote) | Low (local) |

## Future Enhancements

See `docs/so100/network_timeout_improvements.md` for planned improvements:
- Network timeout handling during critical movements
- Action buffering for better reliability  
- Advanced error recovery mechanisms
- Optional video recording (LeRobot-style frame saving)

## Files

- `so100_evaluator.py` - Main evaluation implementation
- `validate_openpi_format.py` - Offline format testing tool
- `README.md` - This documentation