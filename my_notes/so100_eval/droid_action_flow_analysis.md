# DROID Action Processing Analysis (`examples/droid/main.py`)

This document details the action processing flow in the `droid/main.py` example and identifies the key supporting files and modules involved in controlling the DROID arm.

---

### 1. Action Processing Flow

The script receives a chunk of future actions from the policy server and executes them sequentially. Here is the step-by-step flow after an action chunk is received:

#### Step 1: Receive Action Chunk

The `policy_client.infer()` call returns a dictionary containing an `"actions"` key. This value is a NumPy array representing a sequence of future actions, for example, an array of shape `(10, 8)` for 10 timesteps of an 8D action.

```python
# this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
pred_action_chunk = policy_client.infer(request_data)["actions"]
```

#### Step 2: Select a Single Action

In each iteration of the main control loop, the script selects the next single action from the chunk to be processed and executed.

```python
# Select current action to execute from chunk
action = pred_action_chunk[actions_from_chunk_completed]
actions_from_chunk_completed += 1
```

#### Step 3: Binarize Gripper Command

The policy outputs a continuous value for the gripper. The script converts this to a binary command (0.0 for open, 1.0 for close) using a 0.5 threshold. This adapts the policy's output to a two-state gripper.

```python
# Binarize gripper action
if action[-1].item() > 0.5:
    action = np.concatenate([action[:-1], np.ones((1,))])
else:
    action = np.concatenate([action[:-1], np.zeros((1,))])
```

#### Step 4: Clip Action Values

For safety and stability, the entire action vector is clipped to ensure all values fall within the range `[-1, 1]`. This prevents the policy from sending dangerously large commands to the robot.

```python
# clip all dimensions of action to [-1, 1]
action = np.clip(action, -1, 1)
```

#### Step 5: Send to Robot Environment

The final, processed `action` vector is sent to the physical robot by calling the `.step()` method of the `env` object.

```python
env.step(action)
```

---

### 2. Supporting Files and Modules

The `main.py` script collaborates with several key modules:

1.  **`droid.robot_env` (The `RobotEnv` class)**
    *   **Role:** The Hardware Abstraction Layer (HAL) for the DROID arm. It provides a high-level Python API to control the physical robot.
    *   **Interaction:** It is used to get observations from the robot (`env.get_observation()`) and to send motor commands to the robot (`env.step(action)`).

2.  **`openpi_client.websocket_client_policy`**
    *   **Role:** Provides the `WebsocketClientPolicy` class that manages all network communication with the remote policy server.
    *   **Interaction:** Used via `policy_client.infer()` to send observations and receive action chunks, hiding the complexity of WebSocket communication.

3.  **`openpi_client.image_tools`**
    *   **Role:** A utility module for image processing.
    *   **Interaction:** The script uses `image_tools.resize_with_pad()` to resize camera images to the policy's expected input size (224x224) before sending them over the network, which helps reduce latency.

4.  **`numpy`**
    *   **Role:** The core library for numerical computation.
    *   **Interaction:** Used for all manipulation of observation and action data, which are represented as NumPy arrays.
