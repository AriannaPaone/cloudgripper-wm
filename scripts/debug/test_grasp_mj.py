"""Scripted pick-and-lift test for the MuJoCo CloudGripper sim (cube).

Drives cloudgripper_mujoco/Tracking-v0 purely through env.step() delta
actions (same control path the policies use): approach above the cube,
descend, close the gripper, lift, hold — then reports whether the cube
came up with the gripper, how much it slipped, and how fast it spun.

--baseline switches the friction fix off at runtime (pyramidal cone,
impratio=1, condim=3, MuJoCo default friction) so the two can be compared
on the same seeds without touching the scene file. --impratio / --solref
override those solver/contact values the same way (otherwise the scene
file's values are used).

Usage:
    MUJOCO_GL=egl uv run python scripts/debug/test_grasp_mj.py
    MUJOCO_GL=egl uv run python scripts/debug/test_grasp_mj.py --baseline
    MUJOCO_GL=egl uv run python scripts/debug/test_grasp_mj.py --impratio 10
    MUJOCO_GL=egl uv run python scripts/debug/test_grasp_mj.py --grasp-yaw 0 90   # both orientations
    MUJOCO_GL=egl uv run python scripts/debug/test_grasp_mj.py --trials 20 --video-dir /tmp/grasp

    # live MuJoCo viewer (needs a display; leave MUJOCO_GL unset), real-time,
    # optionally also recording the main camera
    uv run python scripts/debug/test_grasp_mj.py --viewer --trials 3 --video-dir /tmp/grasp
"""

import argparse
import os
import sys
import time
from pathlib import Path

# triton (pulled in lazily by torch/torchvision) must load before
# dm_control's EGL backend: on machines where EGL falls back to Mesa,
# Mesa's libLLVM and triton's bundled LLVM clash and importing triton
# second segfaults
try:
    import triton  # noqa: F401
except ImportError:
    pass

import gymnasium as gym
import mujoco
import numpy as np

import environments.mj_cloudgripper  # noqa: F401  (registers the env ids)

# fingertip points in each finger body's frame — same offsets as the
# right/left_finger_site in the upstream cloudgripper_mj scene (finger
# bodies are identical between the two scenes)
FINGERTIP_LOCAL = {
    "Arm_grip_finger_right": np.array([0.02785738743841648, -0.003831740003079176, -0.0022484660148620605]),
    "Arm_grip_finger_left": np.array([0.027868159115314484, 0.0038031437434256077, -0.0022480040788650513]),
}
FINGER_GEOMS = ("Arm_grip_finger_right geom", "Arm_grip_finger_left geom")
CUBE_SIZE = 0.025
CUBE_REST_Z = 0.0139


class GraspTester:
    def __init__(self, env, baseline: bool, video_dir: Path | None,
                 impratio: float | None = None, solref: list[float] | None = None,
                 viewer: bool = False, speed: float = 1.0, camera: str | None = "Camera_main"):
        self.env = env
        self.u = env.unwrapped
        self.baseline = baseline
        self.impratio = impratio
        self.solref = solref
        self.video_dir = video_dir
        self.frames: list[np.ndarray] = []
        self.use_viewer = viewer
        self.speed = speed
        self.camera = camera
        self.viewer = None
        self._t_last = None

    # --- model helpers -------------------------------------------------

    @property
    def m(self):
        return self.u.model

    @property
    def d(self):
        return self.u.data

    def setup_model(self):
        m = self.m
        self.cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == self.cube_body)
        self.cube_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")]
        self.cube_vadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")]
        self.finger_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in FINGERTIP_LOCAL]
        self.finger_geoms = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FINGER_GEOMS}
        # the model is only recompiled when the env is dirty, so set both
        # states explicitly rather than only overriding for the baseline
        if not hasattr(self, "scene_impratio"):
            self.scene_impratio = float(m.opt.impratio)  # value from the scene file
        if self.baseline:
            m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            m.opt.impratio = 1.0
            m.geom_condim[self.cube_geom] = 3
            m.geom_friction[self.cube_geom] = [1.0, 0.005, 0.0001]
        else:
            m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
            m.opt.impratio = self.scene_impratio
            m.geom_condim[self.cube_geom] = 6
            m.geom_friction[self.cube_geom] = [1.0, 0.05, 0.01]
        if self.impratio is not None:
            m.opt.impratio = self.impratio
        if self.solref is not None:
            m.geom_solref[self.cube_geom] = self.solref

    def fingertips(self) -> np.ndarray:
        return np.array([
            self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ local
            for b, local in zip(self.finger_bodies, FINGERTIP_LOCAL.values())
        ])

    def tcp(self) -> np.ndarray:
        return self.fingertips().mean(axis=0)

    def cube_pos(self) -> np.ndarray:
        return self.d.qpos[self.cube_qadr:self.cube_qadr + 3].copy()

    def cube_angvel(self) -> float:
        return float(np.linalg.norm(self.d.qvel[self.cube_vadr + 3:self.cube_vadr + 6]))

    def fingers_touch_cube(self) -> int:
        n = 0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            pair = {c.geom1, c.geom2}
            if self.cube_geom in pair and pair & self.finger_geoms:
                n += 1
        return n

    def _fk(self, pose5) -> tuple[np.ndarray, np.ndarray]:
        """Fingertips at a normalized pose, simulation state restored after."""
        backup = self.d.qpos.copy()
        try:
            self.u.set_active_joints(np.asarray(pose5, dtype=float))
            return self.fingertips()
        finally:
            self.d.qpos[:] = backup
            mujoco.mj_forward(self.m, self.d)

    def solve_tcp_ik(self, target_xyz, rot, grip, x0, max_iter=15, tol=1e-5) -> np.ndarray:
        """Gauss-Newton on normalized x-y-z (same approach as upstream
        CloudgripperMuJoCoEnv.solve_tcp_ik)."""
        target = np.asarray(target_xyz, dtype=float)
        guess = np.asarray(x0, dtype=float).copy()
        fk = lambda xyz: self._fk([*xyz, rot, grip]).mean(axis=0)
        pos = fk(guess)
        eps = 1e-3
        for _ in range(max_iter):
            err = target - pos
            if np.linalg.norm(err) < tol:
                break
            jac = np.zeros((3, 3))
            for i in range(3):
                probe = guess.copy()
                probe[i] = np.clip(probe[i] + eps, 0.0, 1.0)
                step = probe[i] - guess[i]
                if step == 0.0:
                    probe[i] = np.clip(guess[i] - eps, 0.0, 1.0)
                    step = probe[i] - guess[i]
                jac[:, i] = (fk(probe) - pos) / step
            delta, *_ = np.linalg.lstsq(jac, err, rcond=None)
            guess = np.clip(guess + delta, 0.0, 1.0)
            pos = fk(guess)
        return guess

    def finger_sep(self, grip, rot=0.5) -> float:
        tips = self._fk([0.5, 0.5, 0.5, rot, grip])
        return float(np.linalg.norm(tips[0] - tips[1]))

    def grip_for_sep(self, target_sep: float) -> float:
        """Bisect for the grip value whose fingertip separation is target_sep."""
        lo, hi = self.grip_open, self.grip_closed
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            if self.finger_sep(mid) > target_sep:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def calibrate(self):
        """Figure out which grip end is open and, for each grasp yaw (0 and
        90 deg), the rotation that squares the fingers up with the
        (axis-aligned) cube.

        yaw 0 is the cube-face-aligned rotation nearest the joint's centre;
        yaw 90 squeezes across the other pair of faces."""
        s0, s1 = self.finger_sep(0.0), self.finger_sep(1.0)
        self.grip_open, self.grip_closed = (0.0, 1.0) if s0 > s1 else (1.0, 0.0)
        self.max_sep = max(s0, s1)

        rots = np.linspace(0.0, 1.0, 401)
        angs = []
        for rot in rots:
            tips = self._fk([0.5, 0.5, 0.5, rot, self.grip_open])
            v = tips[0] - tips[1]
            angs.append(np.arctan2(v[1], v[0]))
        angs = np.array(angs)

        def wrap(a, period):  # distance to nearest multiple of period
            return np.abs((a + period / 2) % period - period / 2)

        # squeeze axis only matters mod 180deg; face-aligned means mod 90 -> 0
        face_err = wrap(angs, np.pi / 2)
        i0 = int(np.argmin(face_err + 1e-3 * np.abs(rots - 0.5)))
        axis0 = angs[i0]
        # yaw 90: face-aligned AND squeeze axis perpendicular to yaw 0's
        err90 = face_err + wrap(angs - axis0 - np.pi / 2, np.pi)
        i90 = int(np.argmin(err90))
        self.rot_for_yaw = {0: float(rots[i0]), 90: float(rots[i90])}
        self.yaw_err_deg = {0: float(np.degrees(face_err[i0])), 90: float(np.degrees(err90[i90]))}
        self.rot = self.rot_for_yaw[0]

    # --- motion ----------------------------------------------------------

    def open_viewer(self):
        """(Re)attach the passive viewer — reset recompiles the model when
        the env is dirty, which leaves an old viewer pointing at a stale model."""
        if not self.use_viewer:
            return
        if self.viewer is not None and self.viewer.is_running() and self._viewer_model is self.m:
            return
        import mujoco.viewer
        # an old viewer is left open rather than closed: GLFW teardown
        # segfaults on Wayland sessions (see close_viewer)
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        if self.camera:
            # start on a fixed scene camera; Esc in the viewer switches to
            # the free camera, [ / ] cycle through the fixed ones
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        self._viewer_model = self.m
        self._t_last = None

    def close_viewer(self):
        """Wait for the user to close the window, then hard-exit: GLFW's
        own teardown (viewer.close() or interpreter shutdown) segfaults on
        Wayland sessions, so skip it entirely."""
        if self.viewer is None:
            return
        print("close the viewer window to exit")
        while self.viewer.is_running():
            time.sleep(0.1)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def step(self, action):
        obs, *_ = self.env.step(np.asarray(action, dtype=np.float32))
        if self.video_dir is not None:
            self.frames.append(self.u.render())
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt("viewer closed")
            self.viewer.sync()
            # pace to real time (one env step = control_timestep)
            dt = self.u._control_timestep / self.speed
            now = time.perf_counter()
            if self._t_last is not None and now - self._t_last < dt:
                time.sleep(dt - (now - self._t_last))
            self._t_last = time.perf_counter()
        return obs

    def move_to(self, target5, settle=15, max_steps=300):
        """Walk _target_pos to target5 with bounded delta actions, then hold
        until the joints stop moving."""
        target5 = np.clip(np.asarray(target5, dtype=np.float32), 0.0, 1.0)
        max_delta = self.u._max_delta
        for _ in range(max_steps):
            delta = np.clip(target5 - self.u._target_pos, -max_delta, max_delta)
            if np.all(np.abs(delta) < 1e-6):
                break
            self.step(delta)
        still = 0
        prev = self.d.qpos[self.u._active_joint_adrs].copy()
        for _ in range(max_steps):
            self.step(np.zeros(5))
            cur = self.d.qpos[self.u._active_joint_adrs].copy()
            still = still + 1 if np.max(np.abs(cur - prev)) < 1e-5 else 0
            prev = cur
            if still >= settle:
                break

    # --- trial -----------------------------------------------------------

    def run_trial(self, seed, cube_xy, z_hover, grasp_dz, lift, squeeze, hold_steps, grasp_yaw=0) -> dict:
        self.frames = []
        self.env.reset(seed=seed)
        self.setup_model()
        # place cube (mirrors CloudgripperMuJoCoTracking.initialize_episode)
        self.d.qpos[self.cube_qadr:self.cube_qadr + 2] = cube_xy
        self.d.qpos[self.cube_qadr + 2] = CUBE_REST_Z
        self.d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [1, 0, 0, 0]
        self.d.qvel[self.cube_vadr:self.cube_vadr + 6] = 0
        mujoco.mj_forward(self.m, self.d)
        self.open_viewer()
        if not hasattr(self, "rot_for_yaw"):
            self.calibrate()
        self.rot = self.rot_for_yaw[grasp_yaw]

        start_contact = self.fingers_touch_cube()
        cube0 = self.cube_pos()
        grip_close = self.grip_for_sep(CUBE_SIZE - squeeze)

        def pose(xyz_world, grip, rot=None):
            rot = self.rot if rot is None else rot
            xyz = self.solve_tcp_ik(xyz_world, rot, grip, x0=self.u._target_pos[:3])
            return np.array([*xyz, rot, grip], dtype=np.float32)

        cx, cy, cz = cube0
        # 1) open above cube  2) descend  3) close  4) lift  5) hold
        # rise to hover height first with the current rotation, so turning
        # to the grasp yaw can't sweep the fingers through the cube
        self.move_to(pose([cx, cy, cz + z_hover], self.u._target_pos[4], rot=self.u._target_pos[3]))
        self.move_to([*self.u._target_pos[:3], self.rot, self.grip_open])
        self.move_to(pose([cx, cy, cz + z_hover], self.grip_open))
        self.move_to(pose([cx, cy, cz + grasp_dz], self.grip_open))
        tcp_err = float(np.linalg.norm(self.tcp()[:2] - cube0[:2]))
        cube_before_close = self.cube_pos()
        # ramp the grip in a few increments instead of slamming shut
        for g in np.linspace(self.grip_open, grip_close, 6)[1:]:
            self.move_to([*self.u._target_pos[:3], self.rot, g], settle=5)
        contacts_closed = self.fingers_touch_cube()
        offset_grasp = self.cube_pos() - self.tcp()

        self.move_to(pose([cx, cy, cz + grasp_dz + lift], grip_close))
        max_w = 0.0
        rel_z = []
        for _ in range(hold_steps):
            self.step(np.zeros(5))
            max_w = max(max_w, self.cube_angvel())
            rel_z.append(self.cube_pos()[2] - self.tcp()[2])
        cube_end = self.cube_pos()
        slip = float(np.linalg.norm((cube_end - self.tcp()) - offset_grasp))
        # downward creep of the cube relative to the fingers over the second
        # half of the hold (after the initial settling)
        half = len(rel_z) // 2
        creep = (rel_z[half] - rel_z[-1]) / ((len(rel_z) - half) * self.u._control_timestep)
        lifted = float(cube_end[2] - CUBE_REST_Z)

        if self.video_dir is not None and self.frames:
            import imageio
            self.video_dir.mkdir(parents=True, exist_ok=True)
            tag = "baseline" if self.baseline else "fixed"
            if self.impratio is not None:
                tag += f"_impratio{self.impratio:g}"
            if self.solref is not None:
                tag += f"_solref{self.solref[0]:g}"
            imageio.mimsave(self.video_dir / f"grasp_{tag}_yaw{grasp_yaw}_seed{seed}.mp4", self.frames, fps=20)

        return dict(
            seed=seed, yaw=grasp_yaw, cube_xy=cube_xy, start_contact=start_contact, tcp_xy_err=tcp_err,
            pushed_on_descend=float(np.linalg.norm(cube_before_close[:2] - cube0[:2])),
            contacts=contacts_closed, lifted=lifted, slip=slip, creep=creep, max_angvel=max_w,
            success=lifted > 0.5 * lift and slip < 0.005,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", action="store_true", help="disable the friction fix (pyramidal cone, impratio=1, condim=3)")
    p.add_argument("--xy-range", type=float, nargs=2, default=(0.04, 0.03), help="half-extent of random cube x/y (m)")
    p.add_argument("--z-hover", type=float, default=0.03, help="approach height above cube centre (m)")
    p.add_argument("--grasp-dz", type=float, default=0.0, help="grasp height relative to cube centre (m)")
    p.add_argument("--lift", type=float, default=0.02, help="lift distance (m)")
    p.add_argument("--squeeze", type=float, default=0.002, help="close fingers to cube_size - squeeze (m)")
    p.add_argument("--grasp-yaw", type=int, nargs="+", choices=(0, 90), default=[0],
                   help="grasp orientation(s) in deg; each trial runs once per listed yaw")
    p.add_argument("--hold-steps", type=int, default=200, help="steps to hold after lifting (0.05s each)")
    p.add_argument("--impratio", type=float, default=None, help="override model.opt.impratio (default: scene value)")
    p.add_argument("--solref", type=float, nargs=2, default=None, help="override the cube geom's solref")
    p.add_argument("--video-dir", type=Path, default=None, help="save one mp4 per trial (main camera, real-time 20fps)")
    p.add_argument("--video-size", type=int, default=480, help="video frame height/width (px)")
    p.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    p.add_argument("--speed", type=float, default=1.0, help="viewer playback speed (1.0 = real time)")
    p.add_argument("--camera", default="Camera_main", help="viewer start camera ('' for the free camera)")
    args = p.parse_args()

    env = gym.make("cloudgripper_mujoco/Tracking-v0", height=args.video_size, width=args.video_size,
                   max_episode_steps=100_000)
    tester = GraspTester(env, baseline=args.baseline, video_dir=args.video_dir,
                         impratio=args.impratio, solref=args.solref,
                         viewer=args.viewer, speed=args.speed, camera=args.camera)
    rng = np.random.default_rng(args.seed)

    # effective impratio: explicit override > baseline (1) > scene file
    env.reset(seed=args.seed)  # compiles the model so the scene value can be read
    impratio = args.impratio if args.impratio is not None else (
        1.0 if args.baseline else float(tester.u.model.opt.impratio))
    print(f"friction fix: {'OFF (baseline)' if args.baseline else 'ON'}"
          + (f", impratio={impratio:g}" if impratio is not None else "")
          + (f", cube solref={args.solref}" if args.solref is not None else "")
          + f", hold={args.hold_steps * tester.u._control_timestep:.1f}s")
    results = []
    try:
        for t in range(args.trials):
            xy = rng.uniform(-1, 1, 2) * np.array(args.xy_range)
            for yaw in args.grasp_yaw:
                r = tester.run_trial(args.seed + t, xy, args.z_hover, args.grasp_dz,
                                     args.lift, args.squeeze, args.hold_steps, grasp_yaw=yaw)
                results.append(r)
                print(f"[{t:2d}] yaw={yaw:2d} cube=({xy[0]:+.3f},{xy[1]:+.3f}) "
                      f"{'OK  ' if r['success'] else 'FAIL'} lifted={r['lifted'] * 1000:5.1f}mm "
                      f"slip={r['slip'] * 1000:4.1f}mm creep={r['creep'] * 1000:5.2f}mm/s "
                      f"spin={r['max_angvel']:5.2f}rad/s "
                      f"contacts={r['contacts']} tcp_err={r['tcp_xy_err'] * 1000:.1f}mm "
                      f"pushed={r['pushed_on_descend'] * 1000:.1f}mm"
                      + ("  [fingers touched cube at start]" if r['start_contact'] else ""))
    except KeyboardInterrupt as e:
        print(f"stopped: {e}")

    if results:
        print()
        for yaw in args.grasp_yaw:
            rs = [r for r in results if r["yaw"] == yaw]
            print(f"yaw {yaw:2d}: success {sum(r['success'] for r in rs)}/{len(rs)}  "
                  f"(rot={tester.rot_for_yaw[yaw]:.3f}, off-face {tester.yaw_err_deg[yaw]:.1f}deg)")
        print(f"(grip open={tester.grip_open:.0f}, max finger sep={tester.max_sep * 1000:.1f}mm)")
    tester.close_viewer()  # does not return in viewer mode
    env.close()


if __name__ == "__main__":
    main()
