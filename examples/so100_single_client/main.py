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
    # Match the model's action_horizon (pi05 default typically 50)
    action_horizon: int = 50
    prompt: str = "Pick up the object"

    # Robot I/O mode
    # - "http": POST /joints/read -> {"angles_rad": [...]}, POST /joints/write {"angles": [...]}
    # - "lerobot": use SO101 follower from lerobot if available
    # - "none": no robot I/O, just print actions
    io_mode: str = "none"
    http_url: str = "http://localhost:80"
    robot_port: str = "/dev/ttyACM0"  # Robot serial port for lerobot mode
    # Logical robot identifier used by LeRobot to locate calibration and configs.
    # This maps to SO101FollowerConfig(id=...). Defaults to "so101".
    robot_id: str = "so101"
    use_degrees: bool = False  # applies to lerobot mode; if True, convert deg->rad for policy state

    # State validation / mapping (client-side)
    # Units used for the incoming robot state; if "deg", convert first 5 joints to radians.
    state_units: str = "rad"  # one of {"rad", "deg"}
    # Mapping from robot joint order -> model joint order, as comma-separated indices (e.g., "0,1,2,3,4,5").
    state_order: str = "0,1,2,3,4,5"
    # Per-joint limits (radians) for first 5 joints: semicolon-separated pairs, e.g.,
    # "(-2.5,2.5);(-2.5,2.5);(-2.5,2.5);(-2.5,2.5);(-3.14,3.14)". If empty, no clamp applied.
    state_joint_limits: str | None = None
    # Gripper limits as a single pair, e.g., "(0,1)" or "(0,100)". If empty, no clamp applied.
    state_gripper_limits: str | None = None


def _init_cameras(args: Args):
    import cv2

    caps = {}
    for name, idx in {"top": args.top_cam, "wrist": args.wrist_cam, "side": args.side_cam}.items():
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


def _build_lerobot_robot(*, robot_id: str, port: str, use_degrees: bool):
    """
    Construct a LeRobot SO101 follower using the new public API when available,
    falling back to the legacy API if needed.
    """
    # Try new factory-style API first: typed config + factory
    try:
        from lerobot.robots import make_robot_from_config  # type: ignore
        from lerobot.robots.so101_follower import SO101FollowerConfig as _SO101Cfg  # type: ignore

        cfg = _SO101Cfg(
            id=robot_id,
            port=port,
            cameras={},
            use_degrees=use_degrees,
        )
        robot = make_robot_from_config(cfg)
        logging.info("LeRobot: using new robots API (typed SO101FollowerConfig + factory)")
        return robot
    except Exception as e_new:
        # Fallback to legacy direct classes
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig  # type: ignore

            cfg = SO101FollowerConfig(
                id=robot_id,
                port=port,
                cameras={},
                use_degrees=use_degrees,
            )
            robot = SO101Follower(cfg)
            logging.info("LeRobot: using legacy robots API (SO101Follower)")
            return robot
        except Exception as e_old:
            raise ModuleNotFoundError(
                "LeRobot APIs not available for --io-mode lerobot.\n"
                "- Install LeRobot in this environment (e.g., 'GIT_LFS_SKIP_SMUDGE=1 uv sync' or 'uv pip install lerobot').\n"
                "- Newer LeRobot moved modules; this client supports both new and old APIs.\n"
                f"New API error: {e_new}\nOld API error: {e_old}"
            )


class _LeRobotSO101IO(_RobotIOBase):
    def __init__(self, *, robot_id: str, use_degrees: bool, robot_port: str) -> None:
        # Build robot via compatibility builder
        self._robot = _build_lerobot_robot(robot_id=robot_id, port=robot_port, use_degrees=use_degrees)
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
        return _LeRobotSO101IO(robot_id=args.robot_id, use_degrees=args.use_degrees, robot_port=args.robot_port)
    raise ValueError(f"Unsupported io_mode: {args.io_mode}")


def _parse_order(order_str: str) -> list[int]:
    parts = [p.strip() for p in order_str.split(',') if p.strip() != ""]
    try:
        order = [int(p) for p in parts]
    except Exception as e:
        raise ValueError(f"Invalid state_order: {order_str!r}") from e
    if len(order) != 6 or sorted(order) != [0, 1, 2, 3, 4, 5]:
        raise ValueError(f"Invalid state_order length/permutation: {order_str!r}")
    return order


def _parse_limits_list(lims: str, expected_len: int) -> list[tuple[float, float]]:
    entries = [e.strip() for e in lims.split(';') if e.strip()]
    pairs: list[tuple[float, float]] = []
    for e in entries:
        if not (e.startswith('(') and e.endswith(')')):
            raise ValueError(f"Invalid limits entry: {e!r}")
        lo, hi = e[1:-1].split(',')
        pairs.append((float(lo), float(hi)))
    if len(pairs) != expected_len:
        raise ValueError(f"Expected {expected_len} joint limit entries, got {len(pairs)}")
    return pairs


def _parse_limits_pair(lims: str) -> tuple[float, float]:
    s = lims.strip()
    if not (s.startswith('(') and s.endswith(')')):
        raise ValueError(f"Invalid gripper limits: {lims!r}")
    lo, hi = s[1:-1].split(',')
    return float(lo), float(hi)


def _validate_and_convert_state(
    state: np.ndarray,
    *,
    units: str,
    order: list[int],
    joint_limits: list[tuple[float, float]] | None,
    gripper_limits: tuple[float, float] | None,
) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 6:
        raise ValueError(f"Expected state of length 6, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("State contains non-finite values")

    out = arr.copy()
    if units.lower() == "deg":
        out[:5] = np.deg2rad(out[:5])
    elif units.lower() != "rad":
        raise ValueError(f"Unsupported state_units: {units!r}")

    if order != [0, 1, 2, 3, 4, 5]:
        out = out[np.asarray(order, dtype=np.int32)]

    if joint_limits is not None:
        for i in range(5):
            lo, hi = joint_limits[i]
            out[i] = float(np.clip(out[i], lo, hi))
    if gripper_limits is not None:
        lo, hi = gripper_limits
        out[5] = float(np.clip(out[5], lo, hi))

    return out.astype(np.float32)


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
    # Validate a zero-state using the same pipeline for consistency
    zero_state = _validate_and_convert_state(
        np.zeros(6, dtype=np.float32),
        units=args.state_units,
        order=_parse_order(args.state_order),
        joint_limits=_parse_limits_list(args.state_joint_limits, 5) if args.state_joint_limits else None,
        gripper_limits=_parse_limits_pair(args.state_gripper_limits) if args.state_gripper_limits else None,
    )
    obs = {
        "observation.state": zero_state,
        "observation.images.top": _resize_uint8(_read_rgb_bgr(caps["top"]), args.height, args.width),
        "observation.images.wrist": _resize_uint8(_read_rgb_bgr(caps["wrist"]), args.height, args.width),
        "observation.images.side": _resize_uint8(_read_rgb_bgr(caps["side"]), args.height, args.width),
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
            img_side = _resize_uint8(_read_rgb_bgr(caps["side"]), args.height, args.width)

            # Read state
            if robot is not None:
                raw_state = robot.read_state_rad()
            else:
                raw_state = np.zeros(6, dtype=np.float32)

            try:
                state = _validate_and_convert_state(
                    raw_state,
                    units=args.state_units,
                    order=_parse_order(args.state_order),
                    joint_limits=_parse_limits_list(args.state_joint_limits, 5) if args.state_joint_limits else None,
                    gripper_limits=_parse_limits_pair(args.state_gripper_limits) if args.state_gripper_limits else None,
                )
            except Exception as e:
                logging.warning("Invalid state %s; using zeros. Error: %s", raw_state, e)
                state = np.zeros(6, dtype=np.float32)

            obs = {
                "observation.state": state,
                "observation.images.top": img_top,
                "observation.images.wrist": img_wrist,
                "observation.images.side": img_side,
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
    args = tyro.cli(Args)
    main(args)
