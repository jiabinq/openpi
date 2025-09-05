import dataclasses
import logging
import time
from typing import Optional

import numpy as np
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client import action_chunk_broker as _action_chunk_broker


def _read_rgb_bgr(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read from camera")
    # OpenCV returns BGR; convert to RGB
    return frame[:, :, ::-1].copy()


def _resize_uint8(img: np.ndarray, h: int, w: int) -> np.ndarray:
    # Expects HxWxC, returns HxWxC uint8
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, h, w))


@dataclasses.dataclass
class Args:
    # Server location
    host: str = "0.0.0.0"
    port: int = 8000

    # Camera device indices (as provided): side=0 (unused), top=1 (context), wrist=2 (wrist)
    side_cam: int = 0
    top_cam: int = 1
    wrist_cam: int = 2

    # Image size expected by SO100 policies
    height: int = 224
    width: int = 224

    # Control
    freq_hz: int = 30
    action_horizon: int = 10
    prompt: str = "Pick up the object"

    # Robot I/O mode
    # - "http": POST /joints/read -> {"angles_rad": [...]}, POST /joints/write {"angles": [...]}
    # - "lerobot": use SO101 follower from lerobot if available
    # - "none": no robot I/O, just print actions
    io_mode: str = "none"
    http_url: str = "http://localhost:80"
    use_degrees: bool = False  # applies to lerobot mode; if True, convert deg->rad for policy state


def _init_cameras(args: Args):
    import cv2

    caps = {}
    for name, idx in {"top": args.top_cam, "wrist": args.wrist_cam}.items():
        cap = cv2.VideoCapture(int(idx))
        if not cap.isOpened():
            raise RuntimeError(f"Camera {name} (index {idx}) failed to open")
        # Prefer MJPG if supported for lower latency/bandwidth.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        caps[name] = cap
    return caps


def _close_cameras(caps):
    for cap in caps.values():
        try:
            cap.release()
        except Exception:
            pass


class _RobotIOBase:
    def read_state_rad(self) -> np.ndarray:
        raise NotImplementedError

    def send_action(self, action_rad: np.ndarray) -> None:
        raise NotImplementedError


class _HTTPRobotIO(_RobotIOBase):
    def __init__(self, base_url: str) -> None:
        import httpx

        self._client = httpx.Client(timeout=5.0)
        self._base = base_url.rstrip("/")

    def read_state_rad(self) -> np.ndarray:
        r = self._client.post(f"{self._base}/joints/read")
        r.raise_for_status()
        data = r.json()
        return np.asarray(data["angles_rad"], dtype=np.float32)

    def send_action(self, action_rad: np.ndarray) -> None:
        # Send the entire 6-dof target (including gripper if used as 6th entry)
        self._client.post(f"{self._base}/joints/write", json={"angles": action_rad.tolist()}).raise_for_status()


class _LeRobotSO101IO(_RobotIOBase):
    def __init__(self, use_degrees: bool) -> None:
        # Lazy import to avoid hard dependency when not used
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        # Minimal config: adjust port/cameras via your own config file if needed.
        # Here we rely on default camera configs defined externally.
        # You may need to adapt this to your environment.
        # This class assumes the robot object is already configured to access cameras elsewhere, we only use joints here.
        self._cfg = SO101FollowerConfig(
            id="so101",
            port="/dev/ttyUSB0",
            cameras={},
            use_degrees=use_degrees,
        )
        self._robot = SO101Follower(self._cfg)
        self._robot.connect()
        self._deg = use_degrees

    def read_state_rad(self) -> np.ndarray:
        obs = self._robot.get_observation()
        # Order: bus.motors = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        vals = np.asarray([obs[k] for k in keys], dtype=np.float32)
        if self._deg:
            # Convert first 5 joints from degrees to radians; gripper kept as-is (0..100)
            vals[:5] = np.deg2rad(vals[:5])
        return vals

    def send_action(self, action_rad: np.ndarray) -> None:
        # Convert back to robot units
        act = np.asarray(action_rad, dtype=np.float32).copy()
        if self._deg:
            act[:5] = np.rad2deg(act[:5])
        # Map to robot action dictionary
        cmd = {
            "shoulder_pan.pos": float(act[0]),
            "shoulder_lift.pos": float(act[1]),
            "elbow_flex.pos": float(act[2]),
            "wrist_flex.pos": float(act[3]),
            "wrist_roll.pos": float(act[4]),
            "gripper.pos": float(act[5]),
        }
        self._robot.send_action(cmd)


def _make_robot_io(args: Args) -> Optional[_RobotIOBase]:
    mode = args.io_mode.lower()
    if mode == "none":
        return None
    if mode == "http":
        return _HTTPRobotIO(args.http_url)
    if mode == "lerobot":
        return _LeRobotSO101IO(use_degrees=args.use_degrees)
    raise ValueError(f"Unsupported io_mode: {args.io_mode}")


def main(args: Args) -> None:
    logging.info("Connecting to policy server %s:%d", args.host, args.port)
    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", policy.get_server_metadata())

    # Optional: action chunk broker, so we call the server less frequently
    broker = _action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=args.action_horizon)

    # Cameras
    caps = _init_cameras(args)

    # Robot I/O
    robot = _make_robot_io(args)

    # Warmup: read once to ensure server loads
    obs = {
        "observation/state": np.zeros(6, dtype=np.float32),
        "observation/images.main.left": _resize_uint8(_read_rgb_bgr(caps["top"]), args.height, args.width),
        "observation/images.secondary_0": _resize_uint8(_read_rgb_bgr(caps["wrist"]), args.height, args.width),
        "prompt": args.prompt,
    }
    _ = broker.infer(obs)

    period = 1.0 / float(args.freq_hz)
    try:
        next_t = time.perf_counter()
        while True:
            start = time.perf_counter()

            # Read cameras
            img_top = _resize_uint8(_read_rgb_bgr(caps["top"]), args.height, args.width)
            img_wrist = _resize_uint8(_read_rgb_bgr(caps["wrist"]), args.height, args.width)

            # Read state
            if robot is not None:
                state = robot.read_state_rad()
            else:
                state = np.zeros(6, dtype=np.float32)

            obs = {
                "observation/state": state,
                "observation/images.main.left": img_top,
                "observation/images.secondary_0": img_wrist,
                "prompt": args.prompt,
            }

            # Get (or reuse) next action from broker
            action_dict = broker.infer(obs)
            action = np.asarray(action_dict["actions"], dtype=np.float32)
            # If broker returns a chunk, take the first step
            step_action = action[0] if action.ndim == 2 else action

            if robot is not None:
                robot.send_action(step_action)
            else:
                logging.info("Action (first 6): %s", np.array2string(step_action, precision=3))

            # Rate control
            next_t += period
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                # If we fell behind, reset next_t
                next_t = time.perf_counter()

            end = time.perf_counter()
            dt_ms = (end - start) * 1e3
            logging.debug("Loop dt: %.1f ms", dt_ms)

    finally:
        _close_cameras(caps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)

