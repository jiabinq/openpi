# OpenPI Policy Server API Analysis

This document summarizes the API interface for the policy server script (`scripts/serve_policy.py`) and outlines how to create a client for it.

---

### 1. API Interface Analysis

The server's communication is handled by the `WebsocketPolicyServer` class.

*   **Protocol:** The server uses the **WebSocket** protocol for real-time, bidirectional communication. It does not use HTTP.
*   **Endpoint:** The server listens on all network interfaces (`0.0.0.0`) at a configurable port (default `8000`). A client connects to a WebSocket URL like `ws://<server_ip>:8000`.
*   **Functionality:** The server's core function is to wait for a client to connect and send an "observation" message. It passes this observation to the loaded policy model, and sends the resulting "action" back to the client.

---

### 2. Input/Output Schemas

The interaction is message-based, with data typically serialized as JSON. The schemas are determined by the `policy.infer()` method.

#### Input Schema (Client → Server)

The client sends a single message containing a dictionary that represents the robot's observation. For a single-arm SO100 policy, the expected format is:

```json
{
  "observation.state": [
    <float>, <float>, <float>, <float>, <float>, <float>
  ],
  "observation.images.main.left": [
    [ ["<int>", "<int>", "<int>"], ... ],
    ...
  ],
  "observation.images.secondary_0": [
    [ ["<int>", "<int>", "<int>"], ... ],
    ...
  ],
  "prompt": "<string>"
}
```

*   `observation.state`: A list of 6 floats (5 joint angles + 1 gripper state).
*   `observation.images.main.left`: The main camera image (e.g., top-down view) as a HxWx3 nested list.
*   `observation.images.secondary_0`: The wrist camera image, in the same format.
*   `prompt`: A string instruction for the policy.

#### Output Schema (Server → Client)

The server responds with a dictionary containing the predicted action. A key feature is **action chunking**, where the server returns a sequence of future actions, not just a single one.

```json
{
  "actions": [
    [<float>, ...],  // Action for timestep 1
    [<float>, ...],  // Action for timestep 2
    ...
  ],
  "server_timing": { ... },
  "policy_timing": { ... }
}
```

*   `actions`: A list of lists, where each inner list is a full action (e.g., 6 floats for joint/gripper targets) for a future timestep. This allows the client to operate more efficiently without constant network requests.
*   `server_timing` / `policy_timing`: Optional dictionaries with performance metrics.

---

### 3. How to Create a Python Client (Using `openpi-client`)

The recommended method is to use the provided `openpi-client` package, which abstracts away all the low-level networking and data handling.

**Step 1: Install the client package**

```bash
uv pip install -e ./packages/openpi-client
```

**Step 2: Write the Python client script**

The following script demonstrates how to connect to the server, send a sample observation, and receive an action chunk.

```python
import numpy as np
# Import the high-level client from the library
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# 1. Instantiate the client, pointing to your server's IP and port
server_ip = "localhost"  # Or the remote IP of your server
server_port = 8000
policy = WebsocketClientPolicy(host=server_ip, port=server_port)

print(f"Connected to server. Metadata: {policy.get_server_metadata()}")

# 2. Prepare the observation data in the correct dictionary format
# (Using random data for this example)
observation = {
    "observation.state": np.random.rand(6).astype(np.float32),
    "observation.images.main.left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation.images.secondary_0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "prompt": "Pick up the object"
}

# 3. Call the infer() method
# The client handles connecting, serializing the data, sending it,
# and deserializing the response.
print("Sending observation to the server...")
action_dict = policy.infer(observation)
print("Received action chunk from the server.")

# 4. Use the result
# The 'actions' key contains a sequence of actions
action_chunk = action_dict.get("actions")
if action_chunk is not None:
    print(f"Received an action chunk of shape: {np.asarray(action_chunk).shape}")
    # In a real robot loop, you would execute action_chunk[0], then action_chunk[1], etc.
    print(f"First action in chunk: {action_chunk[0]}")

```
