# SO100 Single-Arm Client

A minimal, runnable client that connects to a remote Pi0 policy server and controls a single-arm SO100/ SO101 robot. It captures frames from your top and wrist cameras, reads the arm state, sends a correctly-shaped observation to the server, and streams returned actions to the robot at a fixed rate.

## Overview
- Protocol: WebSocket via `openpi_client.WebsocketClientPolicy`.
- Observation schema (single-arm):
  - `observation/state`: length 6 float array (5 joints + 1 gripper), unnormalized.
  - `observation/images.main.left`: HxWx3 uint8 (context/top cam).
  - `observation/images.secondary_0`: HxWx3 uint8 (wrist cam).
  - `prompt`: string.
- File: `examples/so100_single_client/main.py`.

## Prerequisites
- Install the client package on the robot machine:
  ```bash
  uv pip install -e packages/openpi-client
  ```
- Additional deps (depending on IO mode):
  - Cameras: `pip install opencv-python`
  - HTTP robot API: `pip install httpx`
  - LeRobot SO101 follower: make sure your LeRobot env is installed and importable.

## Start the Policy Server
Serve a single-arm SO100/101 checkpoint (adjust config/dir/port as needed):
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_so100_single \
  --policy.dir=<YOUR_CHECKPOINT_DIR> \
  --port 8000
```

If you only have a dual-arm checkpoint, switch to `pi0_so100` and adapt the client accordingly (not covered here).

## Run the Client
The client defaults to your camera mapping: side=0 (unused), top=1 → context, wrist=2 → wrist.

- HTTP robot I/O (phosphobot-style):
  ```bash
  uv run examples/so100_single_client/main.py \
    --host <SERVER_IP> --port 8000 \
    --top-cam 1 --wrist-cam 2 \
    --io-mode http --http-url http://localhost:80 \
    --freq-hz 30 \
    --prompt "Pick up the orange brick"
  ```
  Expected endpoints:
  - `POST /joints/read` → `{ "angles_rad": [a1, ..., a6] }`
  - `POST /joints/write` with `{ "angles": [a1, ..., a6] }`

- LeRobot SO101 follower I/O (direct control):
  ```bash
  uv run examples/so100_single_client/main.py \
    --host <SERVER_IP> --port 8000 \
    --top-cam 1 --wrist-cam 2 \
    --io-mode lerobot --use-degrees True \
    --freq-hz 30
  ```
  Notes:
  - The example initializes the SO101 follower with `port=/dev/ttyUSB0` and no camera configs — adapt inside `_LeRobotSO101IO` if needed.
  - With `--use-degrees True`, the client converts the first 5 joints deg↔rad; gripper is passed through (0..100).

- No robot I/O (debug only):
  ```bash
  uv run examples/so100_single_client/main.py --host <SERVER_IP> --port 8000 --io-mode none
  ```

## CLI Options (common)
- `--host`, `--port`: address of the policy server.
- `--top-cam`, `--wrist-cam`: OpenCV device indices (default: `1`, `2`).
- `--height`, `--width`: image size (default: `224x224`).
- `--freq-hz`: control loop rate (default: `30`).
- `--action-horizon`: action chunk length (default: `10`).
- `--prompt`: default instruction string.

## How It Works
- Captures top and wrist frames with OpenCV, converts to RGB, resizes/pads to 224, and casts to uint8.
- Reads state (6 floats) from your robot interface.
- Builds an SO100-single observation dict and calls the server via `ActionChunkBroker`.
- Applies the first action from each returned chunk at the requested loop rate.

## Camera Mapping
- `top` (device `--top-cam`) → `observation/images.main.left`.
- `wrist` (device `--wrist-cam`) → `observation/images.secondary_0`.
- `side` (device `--side-cam`) is unused for single arm.

## Safety & Notes
- Start with `--io-mode none` to verify connectivity and shapes.
- Verify joint units/order for your robot; adjust conversions if needed.
- Consider adding joint bounds and e-stop hooks before unattended runs.

## Troubleshooting
- Connection retries: the client waits for the server to accept connections; ensure ports and IPs are reachable.
- Shape errors: confirm observation keys and image shapes match the single-arm policy.
- Camera issues: check device indices with `v4l2-ctl --list-devices` or a quick OpenCV script.
- LeRobot mode: confirm serial `port`, calibration, and permissions for your user.

## Files
- `main.py`: client implementation with camera adapters, HTTP/LeRobot robot I/O, action chunking, and rate control.
