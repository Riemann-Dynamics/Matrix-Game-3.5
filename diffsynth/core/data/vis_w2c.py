import argparse
import numpy as np
from tqdm import tqdm

import open3d as o3d
import open3d.visualization.rendering as rendering
import cv2

import numpy as np


def make_w2c_trajectory(direction: str, num_frames: int) -> np.ndarray:
    """
    根据指令生成 w2c 轨迹。

    参数:
        direction: 只能是 {"left", "right", "forward", "backward"}
            - left/right: 原地绕相机自身y轴左右转 30 度
            - forward/backward: 沿相机自身前向方向前进/后退 1 米
        num_frames: 轨迹帧数

    返回:
        w2c_traj: np.ndarray, shape [num_frames, 4, 4], dtype float32
    """
    if direction not in {"left", "right", "forward", "backward"}:
        raise ValueError(f"Unsupported direction: {direction}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    def rot_y(deg: float) -> np.ndarray:
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float32,
        )

    w2c_traj = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))

    if direction in {"left", "right"}:
        sign = 1.0 if direction == "left" else -1.0
        angles = np.linspace(0.0, sign * 30.0, num_frames, dtype=np.float32)
        for i, a in enumerate(angles):
            w2c_traj[i, :3, :3] = rot_y(a)

    else:
        # 约定相机前向为自身 +Z；w2c 满足 X_cam = R X_world + t
        # 若相机中心 C 在世界坐标下移动，则 t = -R @ C
        sign = 1.0 if direction == "forward" else -1.0
        zs = np.linspace(0.0, sign * 1.0, num_frames, dtype=np.float32)
        for i, z in enumerate(zs):
            C = np.array([0.0, 0.0, z], dtype=np.float32)  # 相机中心 c2w 平移
            R = np.eye(3, dtype=np.float32)
            t = -R @ C
            w2c_traj[i, :3, :3] = R
            w2c_traj[i, :3, 3] = t

    return w2c_traj


def parse_input_extrinsics_to_w2c(
    input_extrinsics,
    degrees=True,
    ypr_order="ypr",
    dtype=np.float64,
):
    """将输入的 xyz+ypr 外参描述解析为 w2c 矩阵序列。

    参数:
        input_extrinsics: 形如 ['x,y,z,yaw,pitch,roll', ...] 的列表，
            也支持 [[x,y,z,yaw,pitch,roll], ...]。
        degrees: ypr 是否按角度输入。
        ypr_order: 旋转叠加顺序，默认 "ypr"。
        dtype: 输出矩阵的数据类型。

    返回:
        np.ndarray: shape=(N,4,4) 的 w2c 外参矩阵。
    """

    def _parse_six_values(item, idx):
        if isinstance(item, str):
            parts = [p.strip() for p in item.split(",")]
        elif isinstance(item, (list, tuple, np.ndarray)):
            parts = list(item)
        else:
            raise ValueError(
                f"input_extrinsics[{idx}] 类型不支持: {type(item).__name__}"
            )

        if len(parts) != 6:
            raise ValueError(
                f"input_extrinsics[{idx}] 必须包含 6 个值: xyz+ypr，当前为 {len(parts)}"
            )

        values = np.asarray([float(v) for v in parts], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"input_extrinsics[{idx}] 存在 NaN/Inf")
        return values

    def _rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    def _rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    def _rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _ypr_to_R(yaw, pitch, roll, order="ypr"):
        mats = {"y": _rot_y(yaw), "p": _rot_x(pitch), "r": _rot_z(roll)}
        R = np.eye(3, dtype=np.float64)
        for k in order:
            if k not in mats:
                raise ValueError(f"ypr_order 中存在非法字符: {k}")
            R = R @ mats[k]
        return R

    if input_extrinsics is None:
        return np.zeros((0, 4, 4), dtype=dtype)
    if not isinstance(input_extrinsics, (list, tuple)):
        raise ValueError("input_extrinsics 必须是 list/tuple")

    w2c_list = []
    for idx, item in enumerate(input_extrinsics):
        x, y, z, yaw, pitch, roll = _parse_six_values(item, idx)
        if degrees:
            yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])

        # 输入 xyz 视为相机中心的世界坐标，先构造 c2w，再求逆得到 w2c。
        R_c2w = _ypr_to_R(yaw, pitch, roll, order=ypr_order)
        C = np.array([x, y, z], dtype=np.float64)

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_c2w.T
        w2c[:3, 3] = -(R_c2w.T @ C)
        w2c_list.append(w2c)

    return np.stack(w2c_list, axis=0).astype(dtype, copy=False)


import numpy as np


import numpy as np


import numpy as np


def build_smooth_w2c_trajectory(
    P_w2c,
    delta_xyz,
    delta_ypr,
    num_frames,
    degrees=True,
    ypr_order="ypr",
    enable_sin_speed=False,
    enable_camera_shake=True,
    shake_xyz_std=(0.01 * 2.5, 0.01 * 2.5, 0.005 * 2.5),
    shake_ypr_std=(0.5 * 2.5, 0.35 * 2.5, 0.2 * 2.5),
    shake_freq=(1.5, 2.3, 3.7),
    shake_seed=0,
    shake_decay_to_zero=True,
    static_shake_ratio=0.1,
):
    """
    Build a smooth W2C trajectory from current pose to target pose defined by a local-camera delta pose.

    Args:
        P_w2c:      (4, 4) current world-to-camera matrix.
        delta_xyz:  (3,) translation delta in the CURRENT CAMERA LOCAL frame.
                    Convention: x=right, y=up, z=forward.
        delta_ypr:  (3,) rotation delta in the CURRENT CAMERA LOCAL frame.
                    Convention: [yaw, pitch, roll].
                    yaw   : rotate around local +y
                    pitch : rotate around local +x
                    roll  : rotate around local +z
        num_frames: int, number of output frames, including start and end.
        degrees:    whether delta_ypr is in degrees.
        ypr_order:  rotation composition order, default "ypr" = yaw -> pitch -> roll.
        enable_sin_speed:
                    If True, use sinusoidal ease-in/ease-out timing so that motion
                    starts slow, becomes faster in the middle, then slows down again.
                    This changes speed profile only, not start/end poses.

        enable_camera_shake:
                    If True, add smooth handheld-like camera shake in LOCAL CAMERA frame.
        shake_xyz_std:
                    3-tuple, local translation shake amplitude (x, y, z), in meters.
        shake_ypr_std:
                    3-tuple, local rotation shake amplitude (yaw, pitch, roll).
                    Unit follows `degrees`: if degrees=True, these are in degrees.
        shake_freq:
                    3-tuple base temporal frequencies for the smooth shake signal.
        shake_seed:
                    Random seed for reproducible shake phase/amplitude sampling.
        shake_decay_to_zero:
                    If True, shake strength smoothly decays to zero at both start and end,
                    so the first and last frames remain very close to the original plan.
        static_shake_ratio:
                    When the camera is effectively static, shake strength is scaled by this ratio.
                    For example 0.1 means static shake is 1/10 of moving shake.

    Returns:
        traj_w2c: (num_frames, 4, 4) smooth W2C trajectory.
    """
    P_w2c = np.asarray(P_w2c, dtype=np.float64)
    delta_xyz = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    delta_ypr = np.asarray(delta_ypr, dtype=np.float64).reshape(3)
    shake_xyz_std = np.asarray(shake_xyz_std, dtype=np.float64).reshape(3)
    shake_ypr_std = np.asarray(shake_ypr_std, dtype=np.float64).reshape(3)
    shake_freq = np.asarray(shake_freq, dtype=np.float64).reshape(3)

    if P_w2c.shape != (4, 4):
        raise ValueError(f"P_w2c must have shape (4,4), got {P_w2c.shape}")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if static_shake_ratio < 0:
        raise ValueError(f"static_shake_ratio must be >= 0, got {static_shake_ratio}")

    delta_ypr_for_motion = delta_ypr.copy()
    if degrees:
        delta_ypr_for_motion = np.deg2rad(delta_ypr_for_motion)

    shake_ypr_std_rad = shake_ypr_std.copy()
    if degrees:
        shake_ypr_std_rad = np.deg2rad(shake_ypr_std_rad)

    def _make_T(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _inv_T(T):
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    def _rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array(
            [
                [1, 0, 0],
                [0, c, -s],
                [0, s, c],
            ],
            dtype=np.float64,
        )

    def _rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array(
            [
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c],
            ],
            dtype=np.float64,
        )

    def _rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def _ypr_to_R(yaw, pitch, roll, order="ypr"):
        mats = {
            "y": _rot_y(yaw),
            "p": _rot_x(pitch),
            "r": _rot_z(roll),
        }
        R = np.eye(3, dtype=np.float64)
        for k in order:
            R = R @ mats[k]
        return R

    def _R_to_quat(R):
        m = R
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        else:
            if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        q /= np.linalg.norm(q) + 1e-12
        return q

    def _quat_to_R(q):
        q = np.asarray(q, dtype=np.float64)
        q = q / (np.linalg.norm(q) + 1e-12)
        w, x, y, z = q
        return np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ],
            dtype=np.float64,
        )

    def _slerp(q0, q1, t):
        q0 = q0 / (np.linalg.norm(q0) + 1e-12)
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)

        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return q / (np.linalg.norm(q) + 1e-12)

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s0 = np.sin(theta_0 - theta) / (sin_theta_0 + 1e-12)
        s1 = sin_theta / (sin_theta_0 + 1e-12)
        q = s0 * q0 + s1 * q1
        return q / (np.linalg.norm(q) + 1e-12)

    def _smoothstep01(x):
        return x * x * (3.0 - 2.0 * x)

    def _sinusoidal_progress(x):
        return 0.5 - 0.5 * np.cos(np.pi * x)

    def _shake_envelope(x):
        return np.sin(np.pi * x) ** 2

    def _make_smooth_noise_signal(ts, amp, freq, rng):
        phase1 = rng.uniform(0.0, 2.0 * np.pi)
        phase2 = rng.uniform(0.0, 2.0 * np.pi)
        phase3 = rng.uniform(0.0, 2.0 * np.pi)

        a1 = rng.uniform(0.6, 1.0)
        a2 = rng.uniform(0.2, 0.5)
        a3 = rng.uniform(0.1, 0.3)

        s = (
            a1 * np.sin(2.0 * np.pi * freq * ts + phase1)
            + a2 * np.sin(2.0 * np.pi * (freq * 1.9) * ts + phase2)
            + a3 * np.sin(2.0 * np.pi * (freq * 2.7) * ts + phase3)
        )

        s = s - s.mean()
        s_std = s.std()
        if s_std < 1e-12:
            return np.zeros_like(ts)
        s = s / s_std
        return amp * s

    P_c2w_start = _inv_T(P_w2c)
    R_start = P_c2w_start[:3, :3]
    t_start = P_c2w_start[:3, 3]

    yaw, pitch, roll = delta_ypr_for_motion
    R_delta = _ypr_to_R(yaw, pitch, roll, order=ypr_order)

    P_delta_local = _make_T(R_delta, delta_xyz)
    P_c2w_end = P_c2w_start @ P_delta_local

    R_end = P_c2w_end[:3, :3]
    t_end = P_c2w_end[:3, 3]

    if num_frames == 1:
        return P_w2c[None].copy()

    q_start = _R_to_quat(R_start)
    q_end = _R_to_quat(R_end)

    motion_translation = np.linalg.norm(delta_xyz)
    motion_rotation = np.linalg.norm(delta_ypr_for_motion)
    has_motion = (motion_translation > 1e-10) or (motion_rotation > 1e-10)

    do_shake = enable_camera_shake
    shake_scale = 1.0 if has_motion else static_shake_ratio

    ts = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
    if do_shake:
        rng = np.random.default_rng(shake_seed)

        shake_tx = _make_smooth_noise_signal(
            ts, shake_xyz_std[0] * shake_scale, shake_freq[0], rng
        )
        shake_ty = _make_smooth_noise_signal(
            ts, shake_xyz_std[1] * shake_scale, shake_freq[1], rng
        )
        shake_tz = _make_smooth_noise_signal(
            ts, shake_xyz_std[2] * shake_scale, shake_freq[2], rng
        )

        shake_yaw = _make_smooth_noise_signal(
            ts, shake_ypr_std_rad[0] * shake_scale, shake_freq[0], rng
        )
        shake_pitch = _make_smooth_noise_signal(
            ts, shake_ypr_std_rad[1] * shake_scale, shake_freq[1], rng
        )
        shake_roll = _make_smooth_noise_signal(
            ts, shake_ypr_std_rad[2] * shake_scale, shake_freq[2], rng
        )

        if shake_decay_to_zero:
            env = _shake_envelope(ts)
            shake_tx *= env
            shake_ty *= env
            shake_tz *= env
            shake_yaw *= env
            shake_pitch *= env
            shake_roll *= env
    else:
        shake_tx = np.zeros(num_frames, dtype=np.float64)
        shake_ty = np.zeros(num_frames, dtype=np.float64)
        shake_tz = np.zeros(num_frames, dtype=np.float64)
        shake_yaw = np.zeros(num_frames, dtype=np.float64)
        shake_pitch = np.zeros(num_frames, dtype=np.float64)
        shake_roll = np.zeros(num_frames, dtype=np.float64)

    traj = np.zeros((num_frames, 4, 4), dtype=np.float64)
    for i in range(num_frames):
        alpha = i / (num_frames - 1)

        if enable_sin_speed:
            alpha = _sinusoidal_progress(alpha)
        else:
            alpha = _smoothstep01(alpha)

        q_i = _slerp(q_start, q_end, alpha)
        R_i = _quat_to_R(q_i)
        t_i = (1.0 - alpha) * t_start + alpha * t_end

        if do_shake:
            t_shake_local = np.array(
                [shake_tx[i], shake_ty[i], shake_tz[i]], dtype=np.float64
            )
            t_i = t_i + R_i @ t_shake_local

            R_shake_local = _ypr_to_R(
                shake_yaw[i], shake_pitch[i], shake_roll[i], order=ypr_order
            )
            R_i = R_i @ R_shake_local

        P_c2w_i = _make_T(R_i, t_i)
        P_w2c_i = _inv_T(P_c2w_i)
        traj[i] = P_w2c_i

    return traj


def prompt_xyz_ypr(
    default_xyz=(0.0, 0.0, 0.0),
    default_ypr=(0.0, 0.0, 0.0),
    input_fn=input,
    print_fn=print,
    require_explicit_input=False,
):
    """交互式读取 xyz/ypr，并返回确认后的输入值。

    参数:
        default_xyz/default_ypr: 回车时使用的默认值。
        require_explicit_input: 为 True 时，不允许直接回车，必须显式输入。

    返回:
        xyz: np.ndarray, shape=(3,)
        ypr: np.ndarray, shape=(3,)
        build_ring_camera: bool，是否构建环视轨迹。
    """

    def _parse_three_floats_strict(raw_text, field_name):
        text = (raw_text or "").strip()
        parts = [p.strip() for p in text.split(",")]
        # 严格要求逗号分隔且恰好 3 项。
        if len(parts) != 3 or any(p == "" for p in parts):
            raise ValueError(
                f"{field_name} 必须是严格的逗号分隔格式，例如 0.0,1.0,-2.5"
            )
        values = np.asarray([float(p) for p in parts], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{field_name} 中包含非有限数值（NaN/Inf）")
        return values

    def _parse_yes_no(raw_text, default=False):
        text = (raw_text or "").strip().lower()
        if text == "":
            return default
        if text in {"y", "yes", "ok", "1", "true"}:
            return True
        if text in {"n", "no", "0", "false"}:
            return False
        raise ValueError("请输入 y 或 n（直接回车使用默认值）")

    default_xyz = np.asarray(default_xyz, dtype=np.float64).reshape(3)
    default_ypr = np.asarray(default_ypr, dtype=np.float64).reshape(3)

    while True:
        try:
            build_ring_camera = _parse_yes_no(
                input_fn("\n\n是否构建环视camera? [y/N], 默认n:").strip(),
                default=False,
            )
            break
        except ValueError as e:
            print_fn(f"输入无效: {e}。请重新输入。")

    while True:
        # xyz 输入校验循环：只有格式合法才会继续。
        while True:
            raw_xyz = input_fn(
                f"\n\n xyz, comma separated [{default_xyz[0]},{default_xyz[1]},{default_xyz[2]}]:"
            ).strip()
            if raw_xyz == "":
                if require_explicit_input:
                    print_fn("首次输入必须显式填写 xyz，不能直接回车。")
                    continue
                xyz = default_xyz.copy()
                break
            try:
                xyz = _parse_three_floats_strict(raw_xyz, "xyz")
                break
            except ValueError as e:
                print_fn(f"输入无效: {e}。请重新输入。")

        # ypr 输入校验循环：只有格式合法才会继续。
        while True:
            raw_ypr = input_fn(
                f"\n\n ypr, comma separated [{default_ypr[0]},{default_ypr[1]},{default_ypr[2]}]:"
            ).strip()
            if raw_ypr == "":
                if require_explicit_input:
                    print_fn("首次输入必须显式填写 ypr，不能直接回车。")
                    continue
                ypr = default_ypr.copy()
                break
            try:
                ypr = _parse_three_floats_strict(raw_ypr, "ypr")
                break
            except ValueError as e:
                print_fn(f"输入无效: {e}。请重新输入。")

        confirm = (
            input_fn(
                f"\n\n确认使用 xyz={xyz.tolist()}, ypr={ypr.tolist()} 构建轨迹吗? [y/N]:"
            )
            .strip()
            .lower()
        )
        if confirm in {"y", "yes", "ok", "1", "true"}:
            return xyz, ypr, build_ring_camera

        print_fn("已取消本次输入，请重新输入 xyz/ypr。")


def build_w2c_trajectory_from_xyz_ypr(
    num_frames,
    P_w2c,
    xyz,
    ypr,
    degrees=True,
    ypr_order="ypr",
    return_normal_indices=True,
    num_fierce=0.0,
    build_ring_camera=False,
    print_fn=print,
):
    """使用给定 xyz/ypr 构建平滑 w2c 轨迹（可选环视模式）。"""

    def _inv_T(T):
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    ypr = np.asarray(ypr, dtype=np.float64).reshape(3)

    target_num_frames = int(num_frames)
    if build_ring_camera:
        # 环视模式固定到 81 帧，便于按 1+20*4 的结构做分组。
        if target_num_frames != 81:
            print_fn(
                f"环视模式固定使用 81 帧，当前输入 {target_num_frames} 将被覆盖为 81。"
            )
        target_num_frames = 81

    traj_w2c = build_smooth_w2c_trajectory(
        delta_xyz=xyz,
        delta_ypr=ypr,
        num_frames=target_num_frames,
        P_w2c=P_w2c,
        degrees=degrees,
        ypr_order=ypr_order,
        shake_xyz_std=(
            0.01 * num_fierce,
            0.01 * num_fierce,
            0.005 * num_fierce,
        ),
        shake_ypr_std=(0.5 * num_fierce, 0.35 * num_fierce, 0.2 * num_fierce),
    )

    normal_indices = list(range(target_num_frames))

    if build_ring_camera:
        ring_traj = traj_w2c.copy()
        normal_indices = []

        # 20 个大帧（每个大帧 4 帧），按“2 个大帧一组”构建双相机：
        # - 第一个大帧：保持输入轨迹不变
        # - 第二个大帧：与第一个大帧朝向一致，仅在世界坐标 x 方向平移 +2
        total_big_frames = max((target_num_frames - 1 + 3) // 4, 0)
        for group_idx in range((total_big_frames + 1) // 2):
            base_big_idx = group_idx * 2
            base_start = 1 + base_big_idx * 4
            if base_start >= target_num_frames:
                continue

            # 以该组第一个大帧的 pose 作为组内参考 pose。
            base_w2c = traj_w2c[base_start]
            base_c2w = _inv_T(base_w2c)
            base_t = base_c2w[:3, 3].copy()
            base_R = base_c2w[:3, :3].copy()

            for slot, x_offset in enumerate([0.0, 2.0]):
                big_idx = base_big_idx + slot
                if big_idx >= total_big_frames:
                    continue
                frame_start = 1 + big_idx * 4
                frame_end = min(frame_start + 4, target_num_frames)
                if frame_start >= target_num_frames:
                    continue

                c2w_new = np.eye(4, dtype=np.float64)
                c2w_new[:3, :3] = base_R
                c2w_new[:3, 3] = base_t.copy()
                c2w_new[0, 3] += float(x_offset)
                w2c_new = _inv_T(c2w_new)

                ring_traj[frame_start:frame_end] = w2c_new

                if slot == 0:
                    normal_indices.extend(list(range(frame_start, frame_end)))

        traj_w2c = ring_traj

    if return_normal_indices:
        return traj_w2c, normal_indices, build_ring_camera
    return traj_w2c, None, None


def prompt_and_build_smooth_w2c_trajectory(
    num_frames,
    P_w2c,
    default_xyz=(0.0, 0.0, 0.0),
    default_ypr=(0.0, 0.0, 0.0),
    degrees=True,
    ypr_order="ypr",
    return_normal_indices=True,
    input_fn=input,
    print_fn=print,
    num_fierce=0.0,
    require_explicit_input=False,
):
    """兼容封装：先 prompt xyz/ypr，再构建 w2c 轨迹。"""
    xyz, ypr, build_ring_camera = prompt_xyz_ypr(
        default_xyz=default_xyz,
        default_ypr=default_ypr,
        input_fn=input_fn,
        print_fn=print_fn,
        require_explicit_input=require_explicit_input,
    )

    traj_w2c, normal_indices, ring_flag = build_w2c_trajectory_from_xyz_ypr(
        num_frames=num_frames,
        P_w2c=P_w2c,
        xyz=xyz,
        ypr=ypr,
        degrees=degrees,
        ypr_order=ypr_order,
        return_normal_indices=return_normal_indices,
        num_fierce=num_fierce,
        build_ring_camera=build_ring_camera,
        print_fn=print_fn,
    )
    return traj_w2c, xyz, ypr, normal_indices, ring_flag


def prompt_num_inference_steps(
    default_steps=5,
    min_steps=1,
    max_steps=100,
    input_fn=input,
    print_fn=print,
):
    """交互式读取推理步数，支持默认值并严格校验为整数。

    参数:
        default_steps: 直接回车时使用的默认步数。
        min_steps: 允许的最小步数（含）。
        max_steps: 允许的最大步数（含）。
        input_fn/print_fn: 便于测试替换输入输出函数。

    返回:
        int: 合法的推理步数。
    """
    default_steps = int(default_steps)
    min_steps = int(min_steps)
    max_steps = int(max_steps)
    if min_steps > max_steps:
        raise ValueError(f"min_steps 不能大于 max_steps: {min_steps} > {max_steps}")
    if not (min_steps <= default_steps <= max_steps):
        raise ValueError(
            f"default_steps={default_steps} 超出范围 [{min_steps}, {max_steps}]"
        )

    while True:
        raw = input_fn(f"\n\n请输入推理步数，默认{default_steps}:").strip()
        if raw == "":
            return default_steps
        if not raw.isdigit():
            print_fn("输入无效：请只输入正整数，例如 5。")
            continue

        steps = int(raw)
        if steps < min_steps or steps > max_steps:
            print_fn(f"输入无效：步数必须在 [{min_steps}, {max_steps}] 之间，请重试。")
            continue
        return steps


def prompt_cfg_mode(
    default_mode="same",
    valid_modes=("same", "cfg_map", "fix_noised", "time_noised", "zero"),
    input_fn=input,
    print_fn=print,
):
    """交互式读取 cfg_mode，回车使用默认值。

    参数:
        default_mode: 默认模式。
        valid_modes: 允许的模式集合。
        input_fn/print_fn: 便于测试替换输入输出函数。

    返回:
        str: 合法的 cfg_mode。
    """
    valid_modes = tuple(valid_modes)
    if default_mode not in valid_modes:
        raise ValueError(f"default_mode={default_mode} 不在允许集合中: {valid_modes}")

    hint = ", ".join(valid_modes)
    while True:
        raw = input_fn(f"\n\n请输入 cfg_mode ({hint})，默认{default_mode}: ").strip()
        if raw == "":
            return default_mode
        if raw in valid_modes:
            return raw
        print_fn(f"输入无效：cfg_mode 只能是 {valid_modes} 之一。")


def prompt_cfg_scale(
    default_scale=6.0,
    min_scale=1.0,
    max_scale=None,
    input_fn=input,
    print_fn=print,
):
    """交互式读取 cfg_scale，回车使用默认值，并校验为有限浮点数。

    参数:
        default_scale: 默认值。
        min_scale: 最小值（含）。
        max_scale: 最大值（含），为 None 时不设上限。
        input_fn/print_fn: 便于测试替换输入输出函数。

    返回:
        float: 合法的 cfg_scale。
    """
    default_scale = float(default_scale)
    min_scale = float(min_scale)
    if max_scale is not None:
        max_scale = float(max_scale)
        if min_scale > max_scale:
            raise ValueError(f"min_scale 不能大于 max_scale: {min_scale} > {max_scale}")

    if not np.isfinite(default_scale):
        raise ValueError("default_scale 必须是有限浮点数")
    if default_scale < min_scale:
        raise ValueError(f"default_scale={default_scale} 小于最小值 {min_scale}")
    if max_scale is not None and default_scale > max_scale:
        raise ValueError(f"default_scale={default_scale} 大于最大值 {max_scale}")

    while True:
        if max_scale is None:
            raw = input_fn(
                f"\n\n请输入 cfg_scale(float, >= {min_scale})，默认{default_scale}: "
            ).strip()
        else:
            raw = input_fn(
                f"\n\n请输入 cfg_scale(float, [{min_scale}, {max_scale}])，默认{default_scale}: "
            ).strip()

        if raw == "":
            return default_scale

        try:
            value = float(raw)
        except ValueError:
            print_fn("输入无效：cfg_scale 必须是浮点数，例如 6.0。")
            continue

        if not np.isfinite(value):
            print_fn("输入无效：cfg_scale 不能是 NaN 或 Inf。")
            continue
        if value < min_scale:
            print_fn(f"输入无效：cfg_scale 不能小于 {min_scale}。")
            continue
        if max_scale is not None and value > max_scale:
            print_fn(f"输入无效：cfg_scale 不能大于 {max_scale}。")
            continue
        return value


def prompt_special_query_frame(
    default=(0,),
    min_frame=0,
    max_frame=None,
    input_fn=input,
    print_fn=print,
    msg="请输入 special_query_frame",
):
    """交互式读取 special_query_frame，支持逗号分隔，始终返回 int 列表。

    参数:
        default: 默认值，允许为整数、整数序列或 None。
        min_frame: 最小允许帧号（含）。
        max_frame: 最大允许帧号（含），为 None 时不设上限。
        input_fn/print_fn: 便于测试替换输入输出函数。

    返回:
        List[int]: 合法帧号列表。单值也会返回单元素列表，例如 [0]。
    """
    min_frame = int(min_frame)
    if max_frame is not None:
        max_frame = int(max_frame)
        if min_frame > max_frame:
            raise ValueError(f"min_frame 不能大于 max_frame: {min_frame} > {max_frame}")

    def _normalize_to_int_list(value, field_name):
        # 统一把默认值和用户输入解析成 int 列表，保证返回类型稳定。
        if value is None:
            return []
        if isinstance(value, int):
            values = [value]
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return []
            if not all(isinstance(v, int) for v in value):
                raise ValueError(f"{field_name} 必须只包含整数")
            values = [int(v) for v in value]
        else:
            raise ValueError(f"{field_name} 必须是 None、整数或整数序列")

        for v in values:
            if v < min_frame:
                raise ValueError(f"{field_name} 中存在小于 {min_frame} 的值: {v}")
            if max_frame is not None and v > max_frame:
                raise ValueError(f"{field_name} 中存在大于 {max_frame} 的值: {v}")
        return values

    default_values = _normalize_to_int_list(default, "default")
    # 若 default=None 或 []，回车时按交互偏好回退到 [0]。
    fallback_values = [0]
    if fallback_values[0] < min_frame or (
        max_frame is not None and fallback_values[0] > max_frame
    ):
        fallback_values = [min_frame]

    default_text = str(default_values if len(default_values) > 0 else fallback_values)
    while True:
        if max_frame is None:
            raw = input_fn(
                f"\n\n{msg}, >= {min_frame})，默认{default_text}(回车): "
            ).strip()
        else:
            raw = input_fn(
                f"\n\n{msg}, [{min_frame}, {max_frame}])，默认{default_text}(回车): "
            ).strip()

        if raw == "":
            return None

        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 0 or any(p == "" for p in parts):
            print_fn("输入无效：请使用逗号分隔整数，例如 2,3,4。")
            continue

        try:
            values = [int(p) for p in parts]
        except ValueError:
            print_fn("输入无效：只能输入整数，多个值请用逗号分隔。")
            continue

        if any(v < min_frame for v in values):
            print_fn(f"输入无效：special_query_frame 不能小于 {min_frame}。")
            continue
        if max_frame is not None and any(v > max_frame for v in values):
            print_fn(f"输入无效：special_query_frame 不能大于 {max_frame}。")
            continue

        # 去重并保留输入顺序，避免重复查询同一帧。
        values = list(dict.fromkeys(values))
        return values


def prompt_input_frame_fn(
    default="",
    input_fn=input,
    print_fn=print,
    msg="请输入插入帧文件名: ",
):
    """交互式读取输入文件名(是图片用cv2.imread)无效则重试,输入空则跳过。"""
    # 延迟导入，避免在不需要该功能时强依赖 cv2。
    try:
        import cv2
    except Exception as e:
        raise ImportError(
            "prompt_input_frame_fn 需要安装 opencv-python 才能读取图片。"
        ) from e

    default = str(default or "").strip()
    for _ in range(3):
        print("\n")
    while True:
        prompt_msg = msg if default == "" else f"{msg}（默认: {default}）"
        raw = input_fn(prompt_msg).strip()

        # 空输入按需求直接跳过。
        if raw == "":
            return None

        file_name = raw
        try:
            img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print_fn(f"读取失败: {e}，请重新输入。")
            continue

        if img is None:
            print_fn("读取失败：不是有效图片路径或图片文件损坏，请重试。")
            continue

        return file_name


def prompt_accept_and_truncate_length(
    default_accept="n",
    min_raw_value=4,
    max_raw_value=320,
    align_divisor=4,
    input_fn=input,
    print_fn=print,
):
    """交互式读取 accept(y/N) 与截断长度整数，并返回按对齐规则截断后的值。

    参数:
        default_accept: 默认接受值，仅允许 "y" 或 "n"。
        min_raw_value: 截断输入的最小整数（含）。
        max_raw_value: 截断输入的最大整数（含）。
        align_divisor: 截断时对齐除数，返回值为 raw_value // align_divisor。
        input_fn/print_fn: 便于测试替换输入输出函数。

    返回:
        accept: bool，是否接受当前结果。
        truncated_value: int，按 align_divisor 截断后的整数。
        raw_value: int，用户原始输入整数。
    """
    default_accept = str(default_accept).strip().lower()
    if default_accept not in {"y", "n"}:
        raise ValueError("default_accept 只能是 'y' 或 'n'")
    min_raw_value = int(min_raw_value)
    max_raw_value = int(max_raw_value)
    align_divisor = int(align_divisor)
    if min_raw_value > max_raw_value:
        raise ValueError(
            f"min_raw_value 不能大于 max_raw_value: {min_raw_value} > {max_raw_value}"
        )
    if align_divisor <= 0:
        raise ValueError("align_divisor 必须是正整数")

    while True:
        # 安全起见，必须显式输入 y/n；回车或其他内容都不接受，防止误触回车。
        raw_accept = input_fn(
            f"\n\n是否接受当前结果? [y/n]（必须显式输入，不可回车默认）: "
        ).strip()
        if raw_accept == "":
            print_fn("输入无效：不能为空，请明确输入 y 或 n。")
            continue

        accept_text = raw_accept.lower()
        if accept_text not in {"y", "n"}:
            print_fn("输入无效：accept 只能输入 y 或 n。")
            continue
        accept = accept_text == "y"
        break
    if accept:
        while True:
            raw = input_fn(
                f"\n\n请输入截断整数 [{min_raw_value}, {max_raw_value}]，将按 //{align_divisor} 对齐:"
            ).strip()
            if not raw.isdigit():
                print_fn("输入无效：请只输入正整数。")
                continue
            raw_value = int(raw)
            if raw_value < min_raw_value or raw_value > max_raw_value:
                print_fn(
                    f"输入无效：截断整数必须在 [{min_raw_value}, {max_raw_value}] 之间。"
                )
                continue
            truncated_value = raw_value // align_divisor
            if truncated_value <= 0:
                print_fn(
                    f"输入无效：当前值经过 //{align_divisor} 后为 {truncated_value}，请增大输入。"
                )
                continue
            return accept, truncated_value, raw_value
    else:
        return accept, 1e9, 1e9


def render_third_person_w2c_frames(
    w2c: np.ndarray,
    width: int = 832,
    height: int = 448,
    follow_dist: float = 2.5,
    follow_up: float = -1.2,
    look_ahead: float = 2.0,
    # Bigger default extent so trajectories that wander several tens of
    # metres (typical for in-the-wild data like DL3DV / Sekai) still sit
    # inside the floor grid. ``grid_step`` is scaled with the size to
    # avoid grid lines becoming too dense to read. ``grid_size`` is also
    # auto-expanded below to fit the actual trajectory bbox.
    grid_size: float = 50.0,
    grid_step: float = 5.0,
    grid_plane: str = "xz",  # "xz" | "xy" | "yz"
    auto_fit_grid: bool = True,
    grid_fit_margin: float = 1.5,
    traj_width: float = 4.0,
    frustum_scale: float = 0.25,
    fov: float = 60.0,
    near: float = 0.05,
    far: float = 200.0,
    background_rgba=(1.0, 1.0, 1.0, 1.0),
    sun_dir=(0.3, -1.0, -0.2),
    sun_color=(1.0, 1.0, 1.0),
    sun_intensity: float = 90000.0,
    annotate_frame_index: bool = True,
    annotation_font_scale: float = 1.0,
):
    """
    Open3D offscreen render:
      - third-person follow camera (behind+above)
      - world grid (default half-extent grid_size=50 m, step grid_step=5 m;
        auto-expanded to cover the trajectory when ``auto_fit_grid`` is set)
      - past trajectory polyline
      - current camera frustum

    Each output frame has ``Frame i/N`` stamped in the top-left corner,
    drawn **after** the geometry flips so the label is upright at the
    visible top-left. ASCII-only on purpose: OpenCV's HERSHEY fonts do
    not render CJK glyphs and would otherwise produce empty boxes.

    Input:
      w2c: (N,4,4)
    Return:
      frames: (N,H,W,3) uint8
    """
    try:
        import open3d as o3d
        import open3d.visualization.rendering as rendering
    except Exception as e:
        raise ImportError("Open3D is required. `pip install open3d`") from e

    w2c = np.asarray(w2c)
    assert w2c.ndim == 3 and w2c.shape[1:] == (
        4,
        4,
    ), f"expected (N,4,4), got {w2c.shape}"
    N = int(w2c.shape[0])
    if N == 0:
        return np.zeros((0, height, width, 3), dtype=np.uint8)

    # invert to c2w (guard numerical issues)
    try:
        c2w = np.linalg.inv(w2c)
        # cv2o3d = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV (y down, z forward) -> Open3D/GL-ish (y up, z backward)
        # c2w = c2w @ cv2o3d
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "w2c contains non-invertible matrices; cannot invert to c2w."
        ) from e
    c2w[:, 1, 3] -= 1
    centers = c2w[:, :3, 3].astype(np.float64)  # (N,3)
    # centers[:, 1] += 1
    R = c2w[:, :3, :3].astype(np.float64)  # (N,3,3)
    forward = R[:, :, 2]  # camera +Z in world
    up = R[:, :, 1]  # camera +Y in world

    # C = np.diag([1.0, -1.0, -1.0]).astype(
    #     np.float64
    # )  # OpenCV cam axes -> OpenGL-ish cam axes
    # R_vis = R @ C  # convert camera basis for visualization

    # # In OpenGL-style camera, "forward viewing direction" is -Z
    # forward = -R_vis[:, :, 2]  # (N,3)
    # up = R_vis[:, :, 1]  # (N,3)

    # Auto-fit the floor grid to the trajectory bbox so wide-area
    # captures (e.g. DL3DV scenes whose camera travels 20+ metres) still
    # have a grid underneath them. We only ever *grow* the grid, never
    # shrink, so callers that pass a deliberately larger size keep it.
    if auto_fit_grid:
        finite = centers[np.isfinite(centers).all(axis=1)]
        if finite.shape[0] > 0:
            if grid_plane == "xz":
                axes = finite[:, [0, 2]]
            elif grid_plane == "xy":
                axes = finite[:, [0, 1]]
            elif grid_plane == "yz":
                axes = finite[:, [1, 2]]
            else:
                axes = finite[:, [0, 2]]
            max_abs = float(np.max(np.abs(axes))) if axes.size > 0 else 0.0
            needed = max_abs * float(grid_fit_margin)
            if needed > grid_size:
                # Snap up to a multiple of grid_step so the lines remain
                # symmetric around the origin.
                grid_size = float(np.ceil(needed / grid_step) * grid_step)

    # eps: ensure AABB is non-empty even if camera is static/repeated
    # use something meaningful relative to your scene scale
    eps = float(max(1e-3, 1e-3 * grid_step, 1e-4 * grid_size))

    def _finite3(x):
        return np.isfinite(x).all(axis=-1)

    def make_grid_lines(size, step, plane="xz"):
        s = float(size)
        st = float(step)
        coords = np.arange(-s, s + 1e-9, st, dtype=np.float64)
        pts, lines = [], []

        if plane == "xz":
            for z in coords:
                pts.append([-s, 0.0, z])
                pts.append([s, 0.0, z])
                lines.append([len(pts) - 2, len(pts) - 1])
            for x in coords:
                pts.append([x, 0.0, -s])
                pts.append([x, 0.0, s])
                lines.append([len(pts) - 2, len(pts) - 1])
        elif plane == "xy":
            for y in coords:
                pts.append([-s, y, 0.0])
                pts.append([s, y, 0.0])
                lines.append([len(pts) - 2, len(pts) - 1])
            for x in coords:
                pts.append([x, -s, 0.0])
                pts.append([x, s, 0.0])
                lines.append([len(pts) - 2, len(pts) - 1])
        elif plane == "yz":
            for z in coords:
                pts.append([0.0, -s, z])
                pts.append([0.0, s, z])
                lines.append([len(pts) - 2, len(pts) - 1])
            for y in coords:
                pts.append([0.0, y, -s])
                pts.append([0.0, y, s])
                lines.append([len(pts) - 2, len(pts) - 1])
        else:
            raise ValueError("grid_plane must be one of: 'xz','xy','yz'")

        pts = np.asarray(pts, dtype=np.float64)
        lines = np.asarray(lines, dtype=np.int32)

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.55, 0.55, 0.55]], dtype=np.float64), (len(lines), 1))
        )
        return ls

    def make_polyline(points, color=(1.0, 0.2, 0.2)):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            pts = pts.reshape(-1, 3)

        # drop non-finite
        if len(pts) > 0:
            pts = pts[_finite3(pts)]
        if len(pts) == 0:
            # dummy non-empty AABB line
            pts = np.array([[0.0, 0.0, 0.0], [eps, 0.0, 0.0]], dtype=np.float64)

        if len(pts) == 1:
            pts = np.vstack([pts, pts + np.array([eps, 0.0, 0.0], dtype=np.float64)])

        # if all points identical / extent too small -> force a tiny extent
        extent = np.max(pts, axis=0) - np.min(pts, axis=0)
        if float(np.max(extent)) < eps:
            pts = pts.copy()
            pts[-1] = pts[-1] + np.array([eps, 0.0, 0.0], dtype=np.float64)

        lines = np.stack(
            [np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1
        ).astype(np.int32)

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([color], dtype=np.float64), (len(lines), 1))
        )
        return ls

    def make_frustum(c2w_4x4, scale=0.25, color=(0.8, 0.8, 1.0)):
        s = float(scale)
        s = max(s, eps)  # avoid degenerate frustum
        o = np.array([0, 0, 0, 1], dtype=np.float64)
        z = s
        x = 0.6 * s
        y = 0.45 * s
        corners = np.array(
            [[-x, -y, z, 1], [x, -y, z, 1], [x, y, z, 1], [-x, y, z, 1]],
            dtype=np.float64,
        )
        pts_cam = np.vstack([o[None, :], corners])  # (5,4)

        T = np.asarray(c2w_4x4, dtype=np.float64)
        pts_w = (T @ pts_cam.T).T[:, :3]

        # drop non-finite (if w2c was bad)
        if not np.isfinite(pts_w).all():
            pts_w = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [eps, 0.0, 0.0],
                    [eps, eps, 0.0],
                    [0.0, eps, 0.0],
                    [0.0, 0.0, eps],
                ],
                dtype=np.float64,
            )

        extent = np.max(pts_w, axis=0) - np.min(pts_w, axis=0)
        if float(np.max(extent)) < eps:
            pts_w = pts_w.copy()
            pts_w[1] = pts_w[1] + np.array([eps, 0.0, 0.0], dtype=np.float64)

        lines = np.array(
            [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]],
            dtype=np.int32,
        )
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts_w)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([color], dtype=np.float64), (len(lines), 1))
        )
        return ls

    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background(list(background_rgba))

    scene.scene.set_sun_light(list(sun_dir), list(sun_color), float(sun_intensity))
    scene.scene.enable_sun_light(True)
    scene.scene.enable_indirect_light(True)

    mat_line = rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = float(traj_width)

    grid = make_grid_lines(grid_size, grid_step, grid_plane)
    scene.add_geometry("grid", grid, mat_line)

    cam = scene.camera
    cam.set_projection(
        float(fov),
        float(width) / float(height),
        float(near),
        float(far),
        rendering.Camera.FovType.Vertical,
    )

    frames = np.empty((N, height, width, 3), dtype=np.uint8)

    traj_name, frus_name = "traj", "frus"

    for i in range(N):
        # trajectory (0..i)
        traj_ls = make_polyline(centers[: i + 1], color=(1.0, 0.2, 0.2))
        if scene.has_geometry(traj_name):
            scene.remove_geometry(traj_name)
        scene.add_geometry(traj_name, traj_ls, mat_line)

        # frustum at i
        frus_ls = make_frustum(c2w[i], scale=frustum_scale, color=(0.2, 0.8, 1.0))
        if scene.has_geometry(frus_name):
            scene.remove_geometry(frus_name)
        scene.add_geometry(frus_name, frus_ls, mat_line)

        c = centers[i]
        f = forward[i]
        u = up[i]
        if not (np.isfinite(c).all() and np.isfinite(f).all() and np.isfinite(u).all()):
            # fallback to something safe
            c = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            f = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            u = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # third-person follow
        eye = c - float(follow_dist) * f + float(follow_up) * u
        lookat = c + float(look_ahead) * f
        upv = (
            u
            if np.linalg.norm(u) >= 1e-9
            else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        )
        cam.look_at(lookat, eye, upv)

        img_o3d = renderer.render_to_image()
        img = np.asarray(img_o3d)  # (H,W,4) RGBA uint8
        frames[i] = img[:, :, :3]

    # renderer.release_resources()
    # 两次 np.flip 把渲染缓冲的轴序矫正到 (上→下, 左→右)；返回的是视图
    # 且非连续，cv2 在某些版本上往视图写入会静默失败，所以下面 putText
    # 之前显式 ascontiguousarray 一次，确保字一定被画上去。
    frames = np.flip(frames, axis=1)  # vertical flip for correct orientation
    frames = np.flip(frames, axis=2)  # horizontal flip for correct handedness
    frames = np.ascontiguousarray(frames)

    if annotate_frame_index and N > 0:
        # 标注必须在几何翻转之后绘制，否则会被一起翻到右下角且倒置。
        fh = float(height)
        base_scale = max(1.0, fh / 360.0)
        font_scale = base_scale * float(annotation_font_scale)
        text_thick = max(2, int(round(fh / 200.0 * annotation_font_scale)))
        outline_thick = text_thick + max(2, int(round(text_thick * 0.8)))
        text_y = int(round(28.0 * fh / 448.0 + 6.0 * font_scale))
        for j in range(N):
            # ASCII only — OpenCV's HERSHEY fonts cannot render CJK and
            # would otherwise produce empty boxes / question marks.
            label = f"Frame {j}/{N - 1}"
            # White outline then black fill: keeps the label readable on
            # both the white background and over darker grid lines.
            cv2.putText(
                frames[j],
                label,
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                outline_thick,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frames[j],
                label,
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                text_thick,
                lineType=cv2.LINE_AA,
            )
    return frames


if __name__ == "__main__":
    # 输入是w2c，如果输入c2w要inv
    import glob
    from tqdm import tqdm

    for idx, vid in tqdm(
        enumerate(glob.glob("runjia.qian/full_camparams_100_c2w_npz/*.npz"))
    ):
        d = np.load(vid)["data"]
        # d = bo
        d = np.linalg.inv(d)
        frames = render_third_person_w2c_frames(d)
        from skvideo.io import vwrite

        fn = vid.replace(".npz", ".mp4")
        vwrite(fn, frames, outputdict={"-r": "30"})
