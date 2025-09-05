# DROID vs ALOHA: Action Chunking, Video/CSV Export, Error Handling

## Completely Different Architectures

| Feature | DROID main.py | ALOHA main.py | Similarity |
|---------|---------------|---------------|------------|
| **Code Lines** | 247 lines | 52 lines | ❌ **Completely Different** |
| **Architecture** | Manual implementation | Framework-based | ❌ **Completely Different** |
| **Action Chunking** | Manual tracking | Framework handles | ❌ **Completely Different** |
| **Video Export** | Explicit implementation | Framework handles | ❌ **Completely Different** |
| **CSV Export** | Explicit implementation | Framework handles | ❌ **Completely Different** |
| **Error Handling** | Custom context managers | Framework handles | ❌ **Completely Different** |

## Detailed Comparison

### **1. Action Chunking**

#### **DROID (Manual Implementation - 20+ lines)**
```python
# Manual action chunk tracking
actions_from_chunk_completed = 0
pred_action_chunk = None

# Manual chunking logic
if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
    actions_from_chunk_completed = 0
    
    # Get new chunk from server
    pred_action_chunk = policy_client.infer(request_data)["actions"]
    assert pred_action_chunk.shape == (10, 8)

# Manual action selection
action = pred_action_chunk[actions_from_chunk_completed]
actions_from_chunk_completed += 1

# Manual action processing
if action[-1].item() > 0.5:
    action = np.concatenate([action[:-1], np.ones((1,))])
else:
    action = np.concatenate([action[:-1], np.zeros((1,))])

action = np.clip(action, -1, 1)
```

#### **ALOHA (Framework-Based - 0 lines visible)**
```python
# All action chunking hidden in framework
agent=_policy_agent.PolicyAgent(
    policy=action_chunk_broker.ActionChunkBroker(  # Framework handles chunking
        policy=ws_client_policy,
        action_horizon=args.action_horizon,  # Just specify horizon
    )
)

# Runtime framework handles everything automatically
runtime.run()  # All chunking logic inside here
```

### **2. Video Export**

#### **DROID (Explicit Implementation - 15+ lines)**
```python
from moviepy.editor import ImageSequenceClip

# Manual video recording
video = []
for t_step in bar:
    video.append(curr_obs[f"{args.external_camera}_image"])  # Collect frames

# Manual video saving
video = np.stack(video)
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
save_filename = "video_" + timestamp
ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")
```

#### **ALOHA (Framework-Based - 0 lines visible)**
```python
# No explicit video code visible
# Framework likely handles video recording internally
runtime = _runtime.Runtime(...)  # Video recording handled inside
runtime.run()
```

### **3. CSV Export**

#### **DROID (Explicit Implementation - 25+ lines)**
```python
import pandas as pd

# Manual CSV tracking
df = pd.DataFrame(columns=["success", "duration", "video_filename"])

# Manual success collection
success = input("Did the rollout succeed? (enter y for 100%, n for 0%)")
if success == "y":
    success = 1.0
elif success == "n":
    success = 0.0

# Manual CSV saving
df = df.append({
    "success": success,
    "duration": t_step,
    "video_filename": save_filename,
}, ignore_index=True)

# Manual file export
os.makedirs("results", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
df.to_csv(csv_filename)
print(f"Results saved to {csv_filename}")
```

#### **ALOHA (Framework-Based - 0 lines visible)**
```python
# No explicit CSV export code
# Framework might handle metrics internally or not at all
runtime.run()  # All evaluation handling inside
```

### **4. Error Handling**

#### **DROID (Sophisticated Implementation - 25+ lines)**
```python
import contextlib
import signal
import faulthandler

faulthandler.enable()

# Complex interrupt handling
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
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

# Protected server calls
with prevent_keyboard_interrupt():
    pred_action_chunk = policy_client.infer(request_data)["actions"]

# Try-catch in main loop
try:
    # Main execution
    ...
except KeyboardInterrupt:
    break
```

#### **ALOHA (Framework-Based - 0 lines visible)**
```python
# No explicit error handling code
# Framework handles errors internally
runtime.run()
```

## Key Insights

### **DROID = Manual Control (Explicit Everything)**
- ✅ **Full visibility** - You see every step of the process
- ✅ **Full control** - Can modify any part of the logic
- ✅ **Rich features** - Video, CSV, interactive evaluation
- ✅ **Debugging friendly** - Easy to add prints, modify logic
- ❌ **More code** - 247 lines vs 52 lines

### **ALOHA = Framework Abstraction (Hidden Everything)**
- ✅ **Concise** - Only 52 lines of user code
- ✅ **Less error-prone** - Framework handles complexity
- ❌ **Black box** - Can't see or modify internal logic
- ❌ **Limited features** - No obvious video/CSV export
- ❌ **Hard to debug** - Issues hidden in framework

## Recommendation for SO-100

**Definitely use DROID as the base**, because:

### **1. Feature Completeness**
DROID has **explicit implementations** of all the features you want:
- ✅ Manual action chunking (easy to adapt for SO-100's 6 DOF)
- ✅ Video recording (essential for evaluation debugging)
- ✅ CSV export (crucial for metrics tracking)  
- ✅ Interactive evaluation (practical for real robot testing)

### **2. Transparency & Control**
- You can **see exactly** how chunking works and modify it
- You can **customize** video recording, CSV format, etc.
- You can **debug** any issues by adding prints/logs

### **3. Proven Implementation**
- DROID's chunking logic is **battle-tested** for real robots
- Error handling is **sophisticated** (prevents server connection deaths)
- Timing control is **precise** for robot control

### **4. Easy Adaptation**
Since everything is explicit, adapting for SO-100 is straightforward:
- Change action dimensions: `(10, 8)` → `(16, 6)`
- Change robot interface: `RobotEnv` → LeRobot SO-100
- Keep all the evaluation features as-is

## Conclusion

**DROID and ALOHA are completely different architectures**:

- **DROID**: Manual, explicit, feature-rich (247 lines)
- **ALOHA**: Framework-based, hidden, minimal (52 lines)

**For SO-100 implementation**, DROID is the clear choice because:
1. ✅ You get all evaluation features explicitly
2. ✅ You can modify/debug every component
3. ✅ The manual chunking logic is perfect for adaptation
4. ✅ It's designed for real robot evaluation scenarios

**ALOHA's Runtime framework** would require:
- Understanding internal framework code
- Potentially missing video/CSV features  
- Less control over the evaluation process
- More complex adaptation for SO-100 specifics