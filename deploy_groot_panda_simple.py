#!/usr/bin/env python3
"""
deploy_groot_panda_simple.py
============================

Single-file GR00T deploy for the Franka Panda. Patterned on the G1 deploy
(creo-g1-teleop/deploy_groot.py): keyboard control loop, per-chunk approval
gate, per-step safety clamping. No docker, no DROID wrapper, no policy
server — just gr00t + polymetis client + opencv cameras.

Required env (Python 3.10):
  pip install -e /path/to/Isaac-GR00T[base]
  pip install polymetis      # client side only — no C++/libfranka here
  pip install opencv-python numpy

The Polymetis SERVER (and the C++ Franka client) keeps running in the
polymetis-local FAIRO env, exactly as today. This script connects to it
over gRPC.

Controls
--------
  S      Start: read obs, run inference, show chunk preview.
  A      Approve & execute the pending chunk.
  R      Reject — run a fresh inference.
  P      Pause: hold last pose, re-arm approval gate.
  I      Return to initial (home) pose.
  Q / Esc Quit.

Safety
------
- Per-step EE translation clamp (default 2 cm), gripper-width cap.
- Joint torque outputs from the model are NOT executed (auxiliary head).
- The FIRST chunk of every (re-)start requires approval before any motion.
- Chunks auto-execute after approval until R, P, or Q.
"""

import argparse
import datetime
import os
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

# Defer heavy imports (cv2, torch, gr00t, polymetis) until after argparse.


# ─────────────────────────────────────────────────────────────────────
# Defaults / safety limits
# ─────────────────────────────────────────────────────────────────────
MAX_STEP_TRANSLATION = 0.02      # m per micro-step
MAX_GRIPPER_WIDTH    = 0.08      # Franka hand max opening (m)
DEFAULT_FPS          = 15
INIT_HOME_Q          = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                                dtype=np.float32)  # canonical Franka home


# ─────────────────────────────────────────────────────────────────────
# Non-blocking single-char keyboard reader
# ─────────────────────────────────────────────────────────────────────
class KeyboardListener:
    """tty raw-mode single-char reader. ASCII only — no pynput dep."""
    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self):
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self):
        """Return one char if pressed, else None. Non-blocking."""
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            ch = sys.stdin.read(1)
            return ch
        return None


# ─────────────────────────────────────────────────────────────────────
# Policy loading
# ─────────────────────────────────────────────────────────────────────
def load_modality_config(modality_path: str):
    """Import the modality config module so register_modality_config runs."""
    import importlib.util
    p = Path(modality_path)
    if not p.exists():
        raise FileNotFoundError(f"modality config not found: {p}")
    spec = importlib.util.spec_from_file_location("modality_config_panda", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"[deploy] Registered modality from {p}")


def load_groot_policy(checkpoint_dir: str, embodiment_tag: str, device: str):
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    if isinstance(embodiment_tag, str):
        try:
            tag = EmbodimentTag(embodiment_tag)
        except ValueError:
            tag = EmbodimentTag[embodiment_tag.upper()]
    else:
        tag = embodiment_tag

    print(f"[deploy] Loading GR00T policy from {checkpoint_dir} (tag={tag.name})")
    policy = Gr00tPolicy(
        embodiment_tag=tag,
        model_path=checkpoint_dir,
        device=device,
        strict=True,
    )
    print("[deploy] Policy loaded.")
    return policy


# ─────────────────────────────────────────────────────────────────────
# Robot (Polymetis client)
# ─────────────────────────────────────────────────────────────────────
class PandaClient:
    """Minimal Polymetis client. Reads state, drives EE via Cartesian impedance."""
    def __init__(self, server_ip: str, server_port: int,
                 gripper_ip: str, gripper_port: int, use_gripper: bool):
        from polymetis import RobotInterface
        self._RobotInterface = RobotInterface
        self.robot = RobotInterface(ip_address=server_ip, port=server_port,
                                    enforce_version=False)
        self.use_gripper = use_gripper
        self.gripper = None
        if use_gripper:
            from polymetis import GripperInterface
            self.gripper = GripperInterface(ip_address=gripper_ip, port=gripper_port)
        self._impedance_started = False
        self._last_gripper_width = MAX_GRIPPER_WIDTH

    def read_state(self) -> dict:
        """Return the 25 panda_laas scalars + ee pose."""
        import torch
        s = self.robot.get_robot_state()
        ee_pos, ee_quat = self.robot.get_ee_pose()   # quat is xyzw

        joint_pos = np.asarray(s.joint_positions, dtype=np.float32)   # (7,)
        joint_vel = np.asarray(s.joint_velocities, dtype=np.float32)  # (7,)
        ee_pos    = np.asarray(ee_pos, dtype=np.float32)              # (3,)
        ee_quat   = np.asarray(ee_quat, dtype=np.float32)             # (4,) xyzw

        if self.gripper is not None:
            g = self.gripper.get_state()
            g_width = float(g.width)
            g_vel   = float(g.velocity if hasattr(g, "velocity") else 0.0)
        else:
            g_width, g_vel = self._last_gripper_width, 0.0

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "ee_pos":    ee_pos,
            "ee_quat":   ee_quat,
            "gripper_width": g_width,
            "gripper_vel":   g_vel,
        }

    def start_impedance(self):
        if not self._impedance_started:
            self.robot.start_cartesian_impedance()
            self._impedance_started = True
            print("[robot] Cartesian impedance started.")

    def stop_impedance(self):
        if self._impedance_started:
            try:
                self.robot.terminate_current_policy()
            except Exception as e:
                print(f"[robot] terminate failed: {e}")
            self._impedance_started = False
            print("[robot] Cartesian impedance stopped.")

    def send_ee(self, pos_xyz: np.ndarray, quat_xyzw: np.ndarray):
        import torch
        self.robot.update_desired_ee_pose(
            position=torch.from_numpy(pos_xyz.astype(np.float32)),
            orientation=torch.from_numpy(quat_xyzw.astype(np.float32)),
        )

    def send_gripper(self, target_width: float, speed: float, force: float):
        if self.gripper is None:
            return
        target_width = float(np.clip(target_width, 0.0, MAX_GRIPPER_WIDTH))
        if abs(target_width - self._last_gripper_width) < 0.005:
            return
        try:
            self.gripper.goto(width=target_width, speed=speed, force=force,
                              blocking=False)
            self._last_gripper_width = target_width
        except Exception as e:
            print(f"[robot] gripper goto failed: {e}")

    def go_home(self, blocking: bool = True):
        import torch
        # Stop any impedance policy first.
        self.stop_impedance()
        print("[robot] Moving to home joint pose...")
        self.robot.move_to_joint_positions(
            torch.from_numpy(INIT_HOME_Q), time_to_go=4.0
        )
        print("[robot] At home.")


# ─────────────────────────────────────────────────────────────────────
# Cameras (cv2.VideoCapture)
# ─────────────────────────────────────────────────────────────────────
class Cameras:
    """Three feeds named exterior_1 / exterior_2 / wrist. cv2 captures."""
    def __init__(self, ext1_dev: int, ext2_dev: int, wrist_dev: int,
                 width: int, height: int):
        import cv2
        self.cv2 = cv2
        self.width, self.height = width, height
        self.caps = {}
        for name, dev in (("exterior_1", ext1_dev),
                          ("exterior_2", ext2_dev),
                          ("wrist",      wrist_dev)):
            if dev < 0:
                continue
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                print(f"[cam] WARNING: /dev/video{dev} ({name}) didn't open")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            self.caps[name] = cap
            print(f"[cam] {name} → /dev/video{dev}")
        if not self.caps:
            raise RuntimeError("No cameras opened; pass --no-cameras to bypass.")

    def warmup(self, n: int = 30):
        for _ in range(n):
            for cap in self.caps.values():
                cap.read()

    def read(self) -> dict:
        out = {}
        for name, cap in self.caps.items():
            ok, frame = cap.read()
            if not ok:
                print(f"[cam] {name} read failed; reusing last frame.")
                continue
            # cv2 returns BGR; gr00t expects RGB.
            frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = self.cv2.resize(frame, (self.width, self.height))
            out[name] = frame
        return out

    def close(self):
        for cap in self.caps.values():
            cap.release()


# ─────────────────────────────────────────────────────────────────────
# Observation assembly (panda_laas modality)
# ─────────────────────────────────────────────────────────────────────
def build_video_dict(frames: dict) -> dict:
    """frames: {exterior_1, exterior_2, wrist} → gr00t obs format (B=1, T=1, H, W, 3)."""
    out = {}
    for name in ("exterior_1", "exterior_2", "wrist"):
        if name not in frames:
            raise KeyError(f"Missing camera feed: {name}")
        out[name] = frames[name][None, None, ...].astype(np.uint8)
    return out


def build_state_dict(state: dict) -> dict:
    """Convert PandaClient.read_state() output to the 25-scalar panda_laas dict.

    Each value is (B=1, T=1, 1) float32. Single gripper width is mirrored
    onto both L and R (modality has the L/R split from training data).
    """
    def s(v: float) -> np.ndarray:
        return np.array([[[float(v)]]], dtype=np.float32)

    jp, jv = state["joint_pos"], state["joint_vel"]
    ep, eq = state["ee_pos"],    state["ee_quat"]
    gw, gV = state["gripper_width"], state["gripper_vel"]
    half_w = gw * 0.5

    d = {}
    for i in range(7):
        d[f"joint_pos_{i+1}"] = s(jp[i])
        d[f"joint_vel_{i+1}"] = s(jv[i])
    d["ee_pos_x"] = s(ep[0]); d["ee_pos_y"] = s(ep[1]); d["ee_pos_z"] = s(ep[2])
    d["ee_quat_x"] = s(eq[0]); d["ee_quat_y"] = s(eq[1])
    d["ee_quat_z"] = s(eq[2]); d["ee_quat_w"] = s(eq[3])
    d["gripper_pos_l"] = s(half_w); d["gripper_pos_r"] = s(half_w)
    d["gripper_vel_l"] = s(gV);     d["gripper_vel_r"] = s(gV)
    return d


# ─────────────────────────────────────────────────────────────────────
# Action chunking, clamping, preview
# ─────────────────────────────────────────────────────────────────────
PANDA_ACTION_KEYS = [
    "ee_pos_x", "ee_pos_y", "ee_pos_z",
    "ee_quat_x", "ee_quat_y", "ee_quat_z", "ee_quat_w",
    "gripper_cmd",
    "torque_j1", "torque_j2", "torque_j3", "torque_j4",
    "torque_j5", "torque_j6", "torque_j7",
]


def chunk_to_actions(action_dict: dict, T: int = 16) -> np.ndarray:
    """Stack gr00t action dict into (T, 15). Order: PANDA_ACTION_KEYS."""
    arr = np.zeros((T, 15), dtype=np.float32)
    for j, k in enumerate(PANDA_ACTION_KEYS):
        v = action_dict[k]
        v = np.asarray(v).reshape(T, -1)[:, 0]
        arr[:, j] = v
    return arr


def clamp_step(target_xyz: np.ndarray, current_xyz: np.ndarray,
               max_translation: float) -> tuple[np.ndarray, bool]:
    delta = target_xyz - current_xyz
    n = float(np.linalg.norm(delta))
    if n > max_translation:
        target_xyz = current_xyz + delta * (max_translation / n)
        return target_xyz.astype(np.float32), True
    return target_xyz.astype(np.float32), False


def summarize_chunk(chunk: np.ndarray, current_xyz: np.ndarray) -> str:
    pos = chunk[:, 0:3]
    grip = chunk[:, 7]
    first_jump = float(np.linalg.norm(pos[0] - current_xyz))
    max_step = float(np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    total_path = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    lines = [
        f"  EE pos start  : {pos[0]}",
        f"  EE pos end    : {pos[-1]}",
        f"  first-step jump from current: {first_jump:.3f} m",
        f"  max intra-chunk step      : {max_step:.3f} m",
        f"  total path length         : {total_path:.3f} m",
        f"  gripper cmd range         : [{grip.min():.2f}, {grip.max():.2f}]",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--modality-config",
                   default=str(Path(__file__).resolve().parent / "modality_config_panda.py"))

    # Polymetis (gRPC)
    p.add_argument("--polymetis-ip",   default="localhost")
    p.add_argument("--polymetis-port", type=int, default=50051)
    p.add_argument("--gripper-ip",     default="localhost")
    p.add_argument("--gripper-port",   type=int, default=50052)
    p.add_argument("--no-gripper", action="store_true")

    # Cameras (cv2.VideoCapture indices). Pass -1 to skip a feed.
    p.add_argument("--cam-ext1",   type=int, default=0)
    p.add_argument("--cam-ext2",   type=int, default=2)
    p.add_argument("--cam-wrist",  type=int, default=4)
    p.add_argument("--image-width",  type=int, default=224)
    p.add_argument("--image-height", type=int, default=224)

    # Control loop
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--max-step-translation", type=float, default=MAX_STEP_TRANSLATION,
                   help="Hard cap on |target_EE - current_EE| per micro-step (m).")
    p.add_argument("--max-gripper-width", type=float, default=MAX_GRIPPER_WIDTH)
    p.add_argument("--gripper-every", type=int, default=4)
    p.add_argument("--gripper-speed", type=float, default=0.1)
    p.add_argument("--gripper-force", type=float, default=10.0)

    # Safety gates
    p.add_argument("--confirm-real", action="store_true",
                   help="Required to enable robot motion. Without this, runs read-only.")

    p.add_argument("--no-cameras", action="store_true",
                   help="Bypass cameras — feed black frames (for wiring tests).")

    p.add_argument("--record-dir", default="./runs")
    args = p.parse_args()

    # ── Load modality + policy ──────────────────────────────────────
    load_modality_config(args.modality_config)
    policy = load_groot_policy(args.checkpoint, args.embodiment_tag, args.device)

    # ── Connect robot ───────────────────────────────────────────────
    print(f"[deploy] Connecting to Polymetis {args.polymetis_ip}:{args.polymetis_port}")
    robot = PandaClient(args.polymetis_ip, args.polymetis_port,
                        args.gripper_ip, args.gripper_port,
                        use_gripper=not args.no_gripper)

    # ── Cameras ─────────────────────────────────────────────────────
    if args.no_cameras:
        cams = None
        blank = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)
    else:
        cams = Cameras(args.cam_ext1, args.cam_ext2, args.cam_wrist,
                       args.image_width, args.image_height)
        cams.warmup(30)

    # ── Output dir ──────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.record_dir) / f"groot_panda_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[deploy] Run dir: {run_dir}")

    if not args.confirm_real:
        print("\n[deploy] READ-ONLY MODE (no motion). Re-launch with --confirm-real "
              "to enable robot motion.\n")

    # ── Main control loop ──────────────────────────────────────────
    print("\nControls: S start  A approve  R reject  P pause  I home  Q quit\n")

    pending_chunk = None       # np.ndarray (T,15) or None
    auto_run = False           # after first approval, subsequent chunks run without re-approval
    target_dt = 1.0 / args.fps
    target_dt_step = target_dt / args.chunk_size  # micro-step dt inside a chunk

    def do_inference(current_xyz: np.ndarray) -> np.ndarray:
        # Read state + cameras
        state = robot.read_state()
        if cams is not None:
            frames = cams.read()
        else:
            frames = {n: blank for n in ("exterior_1", "exterior_2", "wrist")}
        obs = {
            "video":    build_video_dict(frames),
            "state":    build_state_dict(state),
            "language": {"annotation.human.action.task_description": [[args.task]]},
        }
        t0 = time.time()
        action, _info = policy.get_action(obs)
        dt = time.time() - t0
        chunk = chunk_to_actions(action, T=args.chunk_size)
        print(f"[infer] {dt*1000:.0f} ms, chunk shape={chunk.shape}")
        print("[infer] chunk preview:")
        print(summarize_chunk(chunk, current_xyz))
        return chunk

    def execute_chunk(chunk: np.ndarray, kb: KeyboardListener):
        """Stream chunk to the robot with per-step safety clamp."""
        if not args.confirm_real:
            print("[exec] (dry) would execute chunk now.")
            time.sleep(target_dt)
            return True   # pretend success
        robot.start_impedance()
        for i in range(args.chunk_size):
            row = chunk[i]
            target_xyz  = row[0:3].astype(np.float32)
            target_quat = row[3:7].astype(np.float32)
            grip_cmd    = float(np.clip(row[7], 0.0, 1.0))

            state = robot.read_state()
            cur = state["ee_pos"]
            target_xyz, clipped = clamp_step(target_xyz, cur,
                                             args.max_step_translation)
            if clipped:
                print(f"[clamp] step {i}: EE delta capped to "
                      f"{args.max_step_translation:.3f} m")

            robot.send_ee(target_xyz, target_quat)

            if i % args.gripper_every == 0:
                robot.send_gripper(grip_cmd * args.max_gripper_width,
                                   args.gripper_speed, args.gripper_force)

            # Allow interrupt mid-chunk
            ch = kb.poll()
            if ch in ("q", "\x1b"):
                print("[exec] Quit requested mid-chunk.")
                return False
            if ch == "p":
                print("[exec] Paused mid-chunk.")
                return False

            time.sleep(target_dt_step)
        return True

    started = False
    try:
        with KeyboardListener() as kb:
            while True:
                ch = kb.poll()
                if ch in ("q", "\x1b"):
                    print("[deploy] Quit.")
                    break

                if ch == "i":
                    print("[deploy] Returning home.")
                    robot.go_home(blocking=True)
                    pending_chunk = None
                    auto_run = False
                    started = False
                    continue

                if ch == "p":
                    print("[deploy] Paused. Press S to resume.")
                    robot.stop_impedance()
                    pending_chunk = None
                    auto_run = False
                    started = False
                    continue

                if not started:
                    if ch == "s":
                        print("\n[deploy] Starting — running inference for first chunk.")
                        started = True
                        state = robot.read_state()
                        pending_chunk = do_inference(state["ee_pos"])
                        print("\n[deploy] Press A to approve, R to reject.\n")
                    time.sleep(0.05)
                    continue

                # Running. Either we have a pending chunk awaiting approval,
                # or we're in auto-run mode.
                if pending_chunk is not None:
                    if auto_run:
                        ok = execute_chunk(pending_chunk, kb)
                        pending_chunk = None
                        if not ok:
                            auto_run = False
                            started = False
                            continue
                        # Pipeline next chunk
                        state = robot.read_state()
                        pending_chunk = do_inference(state["ee_pos"])
                        continue

                    if ch == "a" or ch == "\r" or ch == " ":
                        print("[deploy] APPROVED — executing chunk.")
                        ok = execute_chunk(pending_chunk, kb)
                        pending_chunk = None
                        if ok:
                            auto_run = True
                            state = robot.read_state()
                            pending_chunk = do_inference(state["ee_pos"])
                        else:
                            auto_run = False
                            started = False
                    elif ch == "r":
                        print("[deploy] REJECTED — re-running inference.")
                        state = robot.read_state()
                        pending_chunk = do_inference(state["ee_pos"])
                    else:
                        time.sleep(0.05)
                else:
                    time.sleep(0.05)
    finally:
        robot.stop_impedance()
        if cams is not None:
            cams.close()
        print("[deploy] Clean exit.")


if __name__ == "__main__":
    main()
