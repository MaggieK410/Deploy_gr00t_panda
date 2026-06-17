#!/usr/bin/env python3
"""
deploy_groot_panda_simple.py
============================

Single-file GR00T deploy for the Franka Panda. STANDALONE — no docker,
no Polymetis, no DROID. Talks directly to the Franka FCI via panda-py
(libfranka python wrapper) and reads cameras with OpenCV.

Install (one Python 3.10+ env):

    pip install -e /path/to/Isaac-GR00T[base]
    pip install panda-python opencv-python numpy pandas pyarrow

Run (dry; no robot motion):

    python deploy_groot_panda_simple.py \\
        --checkpoint /path/to/checkpoint \\
        --task "pick up the red block" \\
        --robot-ip 172.16.0.2

Run (live):

    python deploy_groot_panda_simple.py ... --robot-ip 172.16.0.2 --confirm-real

Run with dataset-derived init pose + debug + per-chunk approval:

    python deploy_groot_panda_simple.py ... --init-dataset /path/to/dataset \\
        --debug --safe --confirm-real

Controls
--------
  S       Start: run inference, show chunk preview.
  A/Enter Approve & execute the pending chunk; later chunks auto-execute
          unless --safe is set.
  R       Reject — fresh inference.
  P       Pause (mid-chunk too): stop controller, re-arm approval.
  I       Move to init pose (dataset init pose if --init-dataset, else Franka home).
  Q/Esc   Quit.

Safety
------
- Per-step EE translation clamp (default 2 cm).
- Joint torque outputs from the model are NOT executed (auxiliary head).
- Without --confirm-real, robot motion is fully suppressed (dry mode).
- FIRST chunk after S/I/P always requires A approval.
- --safe makes every chunk require approval.
"""

import argparse
import datetime
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────
MAX_STEP_TRANSLATION = 0.02      # m per micro-step
MAX_GRIPPER_WIDTH    = 0.08      # Franka hand max opening (m)
DEFAULT_FPS          = 15


# ─────────────────────────────────────────────────────────────────────
# Keyboard
# ─────────────────────────────────────────────────────────────────────
class KeyboardListener:
    """Non-blocking single-char reader (tty raw mode). ASCII only."""
    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self):
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None


# ─────────────────────────────────────────────────────────────────────
# GR00T policy
# ─────────────────────────────────────────────────────────────────────
def load_modality_config(modality_path: str):
    import importlib.util
    p = Path(modality_path)
    if not p.exists():
        raise FileNotFoundError(f"modality config not found: {p}")
    spec = importlib.util.spec_from_file_location("modality_config_panda", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"[deploy] Modality registered from {p}")


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
# Robot (panda-py)
# ─────────────────────────────────────────────────────────────────────
class PandaClient:
    """Thin wrapper around panda-py Panda + libfranka.Gripper."""

    def __init__(self, robot_ip: str, use_gripper: bool):
        import panda_py
        import panda_py.controllers as controllers
        self.panda_py = panda_py
        self.controllers = controllers

        print(f"[robot] Connecting to Franka at {robot_ip}")
        self.panda = panda_py.Panda(robot_ip)
        self.controller = None
        self.controller_running = False

        self.gripper = None
        self._last_gripper_width = MAX_GRIPPER_WIDTH
        if use_gripper:
            from panda_py import libfranka
            print(f"[robot] Connecting to Franka hand at {robot_ip}")
            self.gripper = libfranka.Gripper(robot_ip)
            try:
                state = self.gripper.read_once()
                self._last_gripper_width = float(state.width)
                print(f"[robot] Gripper width = {self._last_gripper_width:.3f} m")
            except Exception as e:
                print(f"[robot] gripper read failed: {e}")

    # ── State ────────────────────────────────────────────────────
    def read_state(self) -> dict:
        s = self.panda.get_state()
        joint_pos = np.asarray(s.q, dtype=np.float32)            # (7,)
        joint_vel = np.asarray(s.dq, dtype=np.float32)           # (7,)
        ee_pos    = np.asarray(self.panda.get_position(), dtype=np.float32)   # (3,)
        # panda-py returns quaternion in [w, x, y, z]. gr00t expects [x,y,z,w].
        q_wxyz = np.asarray(self.panda.get_orientation(), dtype=np.float32)
        ee_quat_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]],
                                dtype=np.float32)

        if self.gripper is not None:
            try:
                g = self.gripper.read_once()
                g_width = float(g.width)
                g_vel = 0.0
                self._last_gripper_width = g_width
            except Exception:
                g_width, g_vel = self._last_gripper_width, 0.0
        else:
            g_width, g_vel = self._last_gripper_width, 0.0

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "ee_pos":    ee_pos,
            "ee_quat":   ee_quat_xyzw,
            "gripper_width": g_width,
            "gripper_vel":   g_vel,
        }

    # ── Cartesian impedance ──────────────────────────────────────
    def start_impedance(self):
        if self.controller_running:
            return
        self.controller = self.controllers.CartesianImpedance()
        self.panda.start_controller(self.controller)
        self.controller_running = True
        print("[robot] Cartesian impedance controller started.")

    def stop_impedance(self):
        if not self.controller_running:
            return
        try:
            self.panda.stop_controller()
        except Exception as e:
            print(f"[robot] stop_controller failed: {e}")
        self.controller_running = False
        self.controller = None
        print("[robot] Controller stopped.")

    def send_ee(self, pos_xyz: np.ndarray, quat_xyzw: np.ndarray):
        """panda-py controllers want quaternion [w, x, y, z]."""
        q_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                           quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        self.controller.set_control(pos_xyz.astype(np.float64), q_wxyz)

    # ── Gripper ──────────────────────────────────────────────────
    def send_gripper(self, target_width: float, speed: float):
        if self.gripper is None:
            return
        target_width = float(np.clip(target_width, 0.0, MAX_GRIPPER_WIDTH))
        if abs(target_width - self._last_gripper_width) < 0.005:
            return
        try:
            self.gripper.move(target_width, speed)
            self._last_gripper_width = target_width
        except Exception as e:
            print(f"[robot] gripper move failed: {e}")

    # ── Home ─────────────────────────────────────────────────────
    def go_home(self, joint_target: np.ndarray | None = None,
                gripper_width: float | None = None):
        """If joint_target is given, move there. Otherwise canonical Franka home."""
        self.stop_impedance()
        if joint_target is not None:
            print(f"[robot] Moving to dataset init joint pose {joint_target}")
            tgt = joint_target.astype(np.float64)
            if hasattr(self.panda, "move_to_joint_position"):
                self.panda.move_to_joint_position(tgt, speed_factor=0.2)
            elif hasattr(self.panda, "move_to_joint_positions"):
                self.panda.move_to_joint_positions(tgt, speed_factor=0.2)
            else:
                print("[robot] WARNING: no move_to_joint_position{,s} method; "
                      "falling back to move_to_start()")
                self.panda.move_to_start()
        else:
            print("[robot] Moving to Franka home pose...")
            self.panda.move_to_start()
        if self.gripper is not None and gripper_width is not None:
            try:
                target = float(np.clip(gripper_width, 0.0, MAX_GRIPPER_WIDTH))
                self.gripper.move(target, 0.1)
                self._last_gripper_width = target
            except Exception as e:
                print(f"[robot] gripper init move failed: {e}")
        print("[robot] At init pose.")


# ─────────────────────────────────────────────────────────────────────
# Cameras
# ─────────────────────────────────────────────────────────────────────
class Cameras:
    """Three feeds: exterior_1 / exterior_2 / wrist via cv2.VideoCapture."""
    def __init__(self, ext1: int, ext2: int, wrist: int,
                 width: int, height: int):
        import cv2
        self.cv2 = cv2
        self.width, self.height = width, height
        self.caps = {}
        for name, dev in (("exterior_1", ext1),
                          ("exterior_2", ext2),
                          ("wrist",      wrist)):
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
            raise RuntimeError("No cameras opened. Pass --no-cameras to bypass.")

    def warmup(self, n: int = 30):
        for _ in range(n):
            for cap in self.caps.values():
                cap.read()

    def read(self) -> dict:
        out = {}
        for name, cap in self.caps.items():
            ok, frame = cap.read()
            if not ok:
                print(f"[cam] {name} read failed.")
                continue
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
    out = {}
    for name in ("exterior_1", "exterior_2", "wrist"):
        if name not in frames:
            raise KeyError(f"Missing camera feed: {name}")
        out[name] = frames[name][None, None, ...].astype(np.uint8)  # (B=1,T=1,H,W,3)
    return out


def build_state_dict(state: dict) -> dict:
    """25-scalar dict matching panda_laas. Single gripper width mirrored L/R."""
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
# Dataset init pose loader (gr00t LeRobot v2 parquet layout)
# ─────────────────────────────────────────────────────────────────────
def load_init_pose_from_dataset(dataset_dir: str, episode_idx: int = 0) -> dict:
    """Return the first-frame state of `episode_idx` in a LeRobot dataset.

    Expected layout:
        <dataset_dir>/data/chunk-XXX/episode_NNNNNN.parquet
    Column 'observation.state' must contain the 25 panda_laas scalars in
    the order specified by meta/info.json (joint_pos_1..7, joint_vel_1..7,
    ee_pos_xyz, ee_quat_xyzw, gripper_pos_l/r, gripper_vel_l/r).
    """
    import pandas as pd
    import json
    info_path = Path(dataset_dir) / "meta" / "info.json"
    if info_path.exists():
        info = json.load(open(info_path))
        chunks_size = info.get("chunks_size", 1000)
        data_path_tmpl = info.get("data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    else:
        chunks_size = 1000
        data_path_tmpl = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    chunk = episode_idx // chunks_size
    rel = data_path_tmpl.format(episode_chunk=chunk, episode_index=episode_idx)
    parquet_path = Path(dataset_dir) / rel
    if not parquet_path.exists():
        candidates = sorted((Path(dataset_dir) / "data").rglob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No parquet under {dataset_dir}/data")
        parquet_path = candidates[0]
        print(f"[init] Episode {episode_idx} not found; using {parquet_path.name}")
    df = pd.read_parquet(parquet_path)
    state0 = np.asarray(df.iloc[0]["observation.state"], dtype=np.float32)
    out = {
        "joint_pos":      state0[0:7],
        "joint_vel":      state0[7:14],
        "ee_pos":         state0[14:17],
        "ee_quat_xyzw":   state0[17:21],
        "gripper_pos_l":  float(state0[21]),
        "gripper_pos_r":  float(state0[22]),
        "source_parquet": str(parquet_path),
    }
    print(f"[init] Loaded init pose from {parquet_path.name}:")
    print(f"  joint_pos = {out['joint_pos']}")
    print(f"  ee_pos    = {out['ee_pos']}")
    print(f"  ee_quat   = {out['ee_quat_xyzw']}")
    print(f"  gripper   = {out['gripper_pos_l']:.3f} (mirrored L/R)")
    return out


# ─────────────────────────────────────────────────────────────────────
# Action chunk handling
# ─────────────────────────────────────────────────────────────────────
PANDA_ACTION_KEYS = [
    "ee_pos_x", "ee_pos_y", "ee_pos_z",
    "ee_quat_x", "ee_quat_y", "ee_quat_z", "ee_quat_w",
    "gripper_cmd",
    "torque_j1", "torque_j2", "torque_j3", "torque_j4",
    "torque_j5", "torque_j6", "torque_j7",
]


def chunk_to_array(action_dict: dict, T: int = 16) -> np.ndarray:
    """Stack gr00t action dict into (T, 15) in PANDA_ACTION_KEYS order."""
    arr = np.zeros((T, 15), dtype=np.float32)
    for j, k in enumerate(PANDA_ACTION_KEYS):
        v = np.asarray(action_dict[k]).reshape(T, -1)[:, 0]
        arr[:, j] = v
    return arr


def clamp_step(target_xyz: np.ndarray, current_xyz: np.ndarray,
               max_translation: float):
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
    if len(pos) > 1:
        max_step = float(np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        total_path = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    else:
        max_step = total_path = 0.0
    return "\n".join([
        f"  EE pos start  : {pos[0]}",
        f"  EE pos end    : {pos[-1]}",
        f"  first-step jump from current: {first_jump:.3f} m",
        f"  max intra-chunk step       : {max_step:.3f} m",
        f"  total path length          : {total_path:.3f} m",
        f"  gripper cmd range          : [{grip.min():.2f}, {grip.max():.2f}]",
    ])


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    # Policy
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--modality-config",
                   default=str(Path(__file__).resolve().parent / "modality_config_panda.py"))

    # Robot
    p.add_argument("--robot-ip", default="172.16.0.2",
                   help="Franka FCI IP.")
    p.add_argument("--no-gripper", action="store_true")

    # Cameras
    p.add_argument("--cam-ext1",  type=int, default=0)
    p.add_argument("--cam-ext2",  type=int, default=2)
    p.add_argument("--cam-wrist", type=int, default=4)
    p.add_argument("--image-width",  type=int, default=224)
    p.add_argument("--image-height", type=int, default=224)
    p.add_argument("--no-cameras", action="store_true",
                   help="Black frames instead of cv2 (wiring tests only).")

    # Control + safety
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--max-step-translation", type=float, default=MAX_STEP_TRANSLATION)
    p.add_argument("--max-gripper-width", type=float, default=MAX_GRIPPER_WIDTH)
    p.add_argument("--gripper-every", type=int, default=4)
    p.add_argument("--gripper-speed", type=float, default=0.1)
    p.add_argument("--confirm-real", action="store_true",
                   help="Enable real robot motion. Without this, dry mode only.")
    p.add_argument("--safe", action="store_true",
                   help="Require A-key approval before EVERY chunk.")
    p.add_argument("--debug", action="store_true",
                   help="Print proprio, first action, quat sanity, dump cam frames.")

    # Init pose from dataset
    p.add_argument("--init-dataset", default=None,
                   help="Path to a LeRobot v2 dataset dir. On I press the robot "
                        "moves to the joint pose of the first frame of episode "
                        "--init-episode, instead of Franka's canonical home.")
    p.add_argument("--init-episode", type=int, default=0)

    p.add_argument("--record-dir", default="./runs")
    args = p.parse_args()

    # ── Load policy ─────────────────────────────────────────────
    load_modality_config(args.modality_config)
    policy = load_groot_policy(args.checkpoint, args.embodiment_tag, args.device)

    # ── Connect robot ───────────────────────────────────────────
    robot = PandaClient(args.robot_ip, use_gripper=not args.no_gripper)

    # ── Dataset init pose (optional) ────────────────────────────
    init_pose = None
    if args.init_dataset:
        init_pose = load_init_pose_from_dataset(args.init_dataset, args.init_episode)

    # ── Cameras ─────────────────────────────────────────────────
    if args.no_cameras:
        cams = None
        blank = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)
    else:
        cams = Cameras(args.cam_ext1, args.cam_ext2, args.cam_wrist,
                       args.image_width, args.image_height)
        cams.warmup(30)

    # ── Output dir ──────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.record_dir) / f"groot_panda_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[deploy] Run dir: {run_dir}")

    if not args.confirm_real:
        print("\n[deploy] DRY MODE (--confirm-real not set). No robot motion.\n")

    print("\nControls: S start  A approve  R reject  P pause  I home  Q quit\n")

    target_dt_chunk = 1.0 / args.fps
    target_dt_step = target_dt_chunk / args.chunk_size

    def do_inference(current_xyz: np.ndarray) -> np.ndarray:
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
        if args.debug:
            print("\n[debug] STATE sent to model:")
            print(f"  joint_pos = {state['joint_pos']}")
            print(f"  joint_vel = {state['joint_vel']}")
            print(f"  ee_pos    = {state['ee_pos']}")
            print(f"  ee_quat (xyzw) = {state['ee_quat']}  |q|={np.linalg.norm(state['ee_quat']):.4f}")
            print(f"  gripper_w = {state['gripper_width']:.3f}")
            if init_pose is not None:
                jp_drift = state['joint_pos'] - init_pose['joint_pos']
                ee_drift = state['ee_pos']    - init_pose['ee_pos']
                print(f"  drift vs dataset init: joint max |delta| = "
                      f"{np.abs(jp_drift).max():.3f} rad, ee delta = "
                      f"{ee_drift} (|delta|={np.linalg.norm(ee_drift):.3f} m)")
        t0 = time.time()
        action, _ = policy.get_action(obs)
        dt = time.time() - t0
        chunk = chunk_to_array(action, T=args.chunk_size)
        print(f"\n[infer] {dt*1000:.0f} ms  chunk={chunk.shape}")
        print("[infer] preview:")
        print(summarize_chunk(chunk, current_xyz))
        if args.debug:
            q0 = chunk[0, 3:7]
            cur_q = state["ee_quat"]
            print("[debug] FIRST ACTION:")
            print(f"  ee_pos     = {chunk[0, 0:3]}")
            print(f"  ee_quat    = {q0}  |q|={np.linalg.norm(q0):.4f}")
            print(f"  gripper    = {chunk[0, 7]:.3f}")
            print(f"  torque[7]  = {chunk[0, 8:15]}")
            print(f"  quat_dot(cur, action[0]) = "
                  f"{abs(float(np.dot(cur_q, q0))):.4f} "
                  f"(near 1.0 = orientations agree, near 0 = swapped/garbage)")
            if cams is not None:
                import cv2
                for name, img in frames.items():
                    cv2.imwrite(str(run_dir / f"debug_{name}.jpg"),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"[debug] camera frames dumped to {run_dir}/debug_*.jpg")
        return chunk

    def execute_chunk(chunk: np.ndarray, kb: KeyboardListener) -> bool:
        if not args.confirm_real:
            print("[exec] (dry) would execute chunk now.")
            time.sleep(target_dt_chunk)
            return True
        robot.start_impedance()
        for i in range(args.chunk_size):
            row = chunk[i]
            target_xyz  = row[0:3].astype(np.float32)
            target_quat = row[3:7].astype(np.float32)   # xyzw
            grip_cmd    = float(np.clip(row[7], 0.0, 1.0))

            cur = robot.read_state()["ee_pos"]
            target_xyz, clipped = clamp_step(target_xyz, cur,
                                             args.max_step_translation)
            if clipped:
                print(f"[clamp] step {i}: EE delta capped to "
                      f"{args.max_step_translation:.3f} m")

            robot.send_ee(target_xyz, target_quat)

            if i % args.gripper_every == 0:
                robot.send_gripper(grip_cmd * args.max_gripper_width,
                                   args.gripper_speed)

            ch = kb.poll()
            if ch in ("q", "\x1b"):
                print("[exec] Quit mid-chunk.")
                return False
            if ch == "p":
                print("[exec] Paused mid-chunk.")
                return False

            time.sleep(target_dt_step)
        return True

    pending_chunk = None
    auto_run = False
    started = False

    try:
        with KeyboardListener() as kb:
            while True:
                ch = kb.poll()
                if ch in ("q", "\x1b"):
                    print("[deploy] Quit.")
                    break

                if ch == "i":
                    if init_pose is not None:
                        robot.go_home(joint_target=init_pose["joint_pos"],
                                      gripper_width=init_pose["gripper_pos_l"])
                    else:
                        robot.go_home()
                    pending_chunk = None
                    auto_run = False
                    started = False
                    continue

                if ch == "p":
                    print("[deploy] Paused.")
                    robot.stop_impedance()
                    pending_chunk = None
                    auto_run = False
                    started = False
                    continue

                if not started:
                    if ch == "s":
                        print("[deploy] Start.")
                        started = True
                        cur = robot.read_state()["ee_pos"]
                        pending_chunk = do_inference(cur)
                        print("\n[deploy] Press A to approve, R to reject.\n")
                    time.sleep(0.05)
                    continue

                if pending_chunk is not None:
                    if auto_run:
                        ok = execute_chunk(pending_chunk, kb)
                        pending_chunk = None
                        if not ok:
                            auto_run = False
                            started = False
                            continue
                        cur = robot.read_state()["ee_pos"]
                        pending_chunk = do_inference(cur)
                        continue

                    if ch in ("a", "\r", " "):
                        print("[deploy] APPROVED — executing.")
                        ok = execute_chunk(pending_chunk, kb)
                        pending_chunk = None
                        if ok:
                            auto_run = not args.safe
                            cur = robot.read_state()["ee_pos"]
                            pending_chunk = do_inference(cur)
                            if args.safe:
                                print("\n[deploy] [SAFE] Press A to approve next chunk, R to reject.\n")
                        else:
                            auto_run = False
                            started = False
                    elif ch == "r":
                        print("[deploy] REJECTED — re-running inference.")
                        cur = robot.read_state()["ee_pos"]
                        pending_chunk = do_inference(cur)
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
