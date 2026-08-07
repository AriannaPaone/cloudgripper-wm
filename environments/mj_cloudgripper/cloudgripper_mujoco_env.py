from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box
from ogbench.manipspace.envs.env import CustomMuJoCoEnv
from ogbench.manipspace.mjcf_utils import safe_find
from dm_control import mjcf

from stable_worldmodel import spaces as swm_spaces





class CloudgripperMuJoCoEnv(CustomMuJoCoEnv):
    """MuJoCo simulation of a CloudGripper cell."""

    metadata: dict = {
        "render_modes": ["rgb_array"],
        "render_fps": 20,
    }

    # DEFAULT CLOUDGRIPPER MUJOCO MODEL
    xml_file: str = Path(__file__).resolve().parent / "cloudgripper_scene.xml"

    # INIT VALUES OF DEFAULT MODEL
    initial_pose = np.zeros(5, dtype=np.float32)
    material_colors: dict = {
        "glass_floor": np.array([1.0, 1.0, 1.0]),
        "gray_mat": np.array([0.87294, 0.95294, 0.87294]),
        "yellow_plastic": np.array([1.0, 0.8, 0.15]),
        "led_light" : np.array([1.0, 1.0, 1.0]),
        "black_plastic": np.array([0.0, 0.0, 0.0]),
        "seethrough_plastic": np.array([1.0, 1.0, 1.0]),
        "metal": np.array([0.8, 0.8, 0.8]),
    }
    light_intensity: float = 1.5

    # IDENTIFIERS OF DEFAULT MODEL
    main_camera: str = "Camera_main"
    camera_ids: list = [main_camera, "Camera_bottom"]
    arm_names: list = [
        "Frame",
        "Rail_slider",
        "Slider",
        "Arm_holder",
        "Arm_linear_gear",
        "Arm_rotation_base",
        "Arm_grip_base",
        "SpurGear1_v1"
        "Arm_grip_finger_right",
        "SpurGear2_v1",
        "Arm_grip_finger_left",
        "Arm_grip_link_back_left",
        "Arm_grip_link_back_right",
        "Arm_grip_link_front_left",
        "Arm_grip_link_front_right",
        "Arm_linear_pinion_gear",
    ]
    servo_names: list = [
        "Servo_gripper",
        "Servo_rotation",
        "Servo_linear",
    ]
    cage_names: list = [
        "Frame",
        "Ground_plate"
    ]
    light_names: list = [
        "wall_right_light_upper",
        "wall_right_light_lower",
        "wall_left_light_upper",
        "wall_left_light_lower",
    ]
    lightbulb_names: list = [
        "led_right_upper1",
        "led_right_upper2",
        "led_right_upper3",
        "led_right_lower1",
        "led_right_lower2",
        "led_right_lower3",
        "led_left_upper1",
        "led_left_upper2",
        "led_left_upper3",
        "led_left_lower1",
        "led_left_lower2",
        "led_left_lower3",
    ]
    wall_names: list = [
        "wall_front",
        "wall_back",
        "wall_right",
        "wall_left",
        "wall_top",
    ]
    material_names: list = [
        "glass_floor",
        "gray_mat",
        "led_light",
        "yellow_plastic",
        "black_plastic",
        "seethrough_plastic",
        "metal",
    ]
    actuator_names: list = [
        "x_actuator",
        "y_actuator",
        "z_actuator",
        "rotate_actuator",
        "gripper_right_actuator",
    ]
    joint_names: list =[
        "Rail_joint",
        "Slider_joint",
        "Linear_joint",
        "Rotation_joint",
        "RightSpur_joint", # remaining gripper joints posses equ. constraints to this one
    ]

    def __init__(
        self,
        mode: str = "data_collection",
        max_delta: float = 0.05,
        step_threshold: float = 0.01,
        physics_timestep: float = 0.002,
        control_timestep: float = 0.05,
        **kwargs,
    ):
        """Initialize the wrapper for Cloudgripper MuJoCo model. 

        Args:
            mode: Environment mode (data_collection or task)
            max_delta: Sampling range of action_space for all five joints
            step_threshold: Threshold for moving to a new position for all five joints.
                Mirrors real cloudgripper robot to only actuate joints if the relative
                distance between current and target state exceeds the threshold.
            physics_timestep: Physics timestep.
            control_timestep: Control timestep.
        """
        super().__init__(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            **kwargs,
        )

        self._mode: str = mode
        self._max_delta:float = max_delta
        self._step_thresh: float = step_threshold

        self._current_pos:np.ndarray = self.initial_pose.copy()
        self._target_pos: np.ndarray = self.initial_pose.copy()

        self._colors: dict = dict(
            red=np.array([0.96, 0.26, 0.33, 1.0]),
            orange=np.array([1.0, 0.69, 0.21, 1.0]),
            yellow=np.array([0.76, 0.96, 0.04, 1.0]),
            green=np.array([0.06, 0.74, 0.21, 1.0]),
            blue=np.array([0.35, 0.55, 0.91, 1.0]),
            purple=np.array([0.61, 0.28, 0.82, 1.0]),
            magenta=np.array([0.82, 0.28, 0.61, 1.0]),
            lightred=np.array([0.99, 0.85, 0.86, 1.0]),
            lightorange=np.array([1.0, 0.94, 0.84, 1.0]),
            lightyellow=np.array([0.95, 0.99, 0.8, 1.0]),
            lightgreen=np.array([0.77, 0.95, 0.81, 1.0]),
            lightblue=np.array([0.86, 0.9, 0.98, 1.0]),
            lightpurple=np.array([0.91, 0.84, 0.96, 1.0]),
            lightmagenta=np.array([0.96, 0.84, 0.91, 1.0]),
            white=np.array([0.9, 0.9, 0.9, 1.0]),
            lightgray=np.array([0.7, 0.7, 0.7, 1.0]),
            gray=np.array([0.5, 0.5, 0.5, 1.0]),
            darkgray=np.array([0.3, 0.3, 0.3, 1.0]),
            black=np.array([0.1, 0.1, 0.1, 1.0]),
        )
        

    def build_mjcf_model(self) -> mjcf.RootElement:
        """Loads the default cloudgripper mujoco environment.

        Also extracts names of the model's assets, including
            - body names
            - camera names
            - joint names
            - light names

        mjcf/xml : "environments/mj_cloudgripper/cloudgripper_scene.xml"
        blender  : "environments/mj_cloudgripper/cloudgripper_scene.blend"
        
        World frame is placed in the middle of the ground plate,
        as visualized below. For custom workspace layouts define
        a translation function in swm-child class.

        ______________________
        |                    |
        |                    |
        |                    |
        |             y      |
        |          -->       |
        |         |          |
        |         v          |
        |           x        |
        |                    |
        |________cam_________|
                (main)
        """
        mjcf_model: mjcf.RootElement = mjcf.from_path(str(self.xml_file))
        
        return mjcf_model

    def post_compilation(self) -> None:
        # get any robot data here to store after mujoco model was compiled

        # TODO: assert that cameras in self._cloudgripper_cameras exists in mujoco model
        cg_assets = (
            self.arm_names + 
            self.cage_names + 
            self.servo_names + 
            self.light_names + 
            self.lightbulb_names +
            self.wall_names + 
            self.material_names + 
            self.actuator_names +
            self.joint_names
        )


        # adresses of control joints
        self._active_joint_adrs = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in self.joint_names
        ]

        self.post_compilation_objects()
        

    def post_compilation_objects(self) -> None:
        # get any object data here to store after mujoco model was compiled
        pass

    def compute_observation(self) -> dict:
        # images flow through render() -> AddPixelsWrapper's info["pixels"],
        # not through obs (matches the real-robot CloudGripperEnv convention,
        # and avoids rendering twice per step).
        return {"state": self._current_pos.astype(np.float32)}

    def compute_reward(self) -> float:
        return 0.0

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "state": Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
            }
        )

    @property
    def action_space(self):
        return swm_spaces.Box(
            low=-self._max_delta,
            high=self._max_delta,
            shape=(5,),
            dtype=np.float32,
        )

    def unnormalize(self, val, joint):
        """Unnormalize value depending on joint range
        """
        l_lim, u_lim = joint.range
        return (u_lim - l_lim) * val + l_lim

    def set_active_joints(self, values: np.ndarray) -> None:
        """Controls active joints and solves kinematics for stable configuration.

        Passive joints are linked to active joints via equality constraints.
        Active joints are by default
            Rail_joint
            Slider_joint
            Linear_joint
            Rotation_joint
            RightSpur_joint

        Args:
            values: normalized [0, 1] target for each of self.joint_names,
                same convention as set_control()'s action.
        """
        for val, joint_id, adr in zip(values, self.joint_names, self._active_joint_adrs):
            self.data.qpos[adr] = self.unnormalize(val, self.model.joint(joint_id))

        # <equality><joint> constraints are enforced by constraint forces during
        # mj_step, not by mj_forward/mj_kinematics — so directly poking qpos here
        # leaves passive (linked) joints stale. Solve their polynomial relation
        # explicitly instead: qpos2 = qpos0_2 + polycoef(qpos1 - qpos0_1).
        #
        # Constraints can chain (e.g. RightSpur_joint -> LeftSpur_joint ->
        # LeftFinger_joint) and aren't necessarily declared in dependency order
        # in the XML, so repeat the full pass once per constraint — enough
        # passes to propagate through a chain of any depth up to neq.
        for _ in range(self.model.neq):
            for i in range(self.model.neq):
                if self.model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                    continue
                adr1 = self.model.jnt_qposadr[self.model.eq_obj1id[i]]
                adr2 = self.model.jnt_qposadr[self.model.eq_obj2id[i]]
                c = self.model.eq_data[i, :5]
                x = self.data.qpos[adr1] - self.model.qpos0[adr1]
                self.data.qpos[adr2] = self.model.qpos0[adr2] + (
                    c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4
                )

        mujoco.mj_forward(self.model, self.data)


    def set_control(self, action) -> None:
        self._target_pos = np.clip(self._target_pos + action, 0., 1.)
        dx, dy, dz, dr, dg = np.abs(self._target_pos - self._current_pos)

        # mirros real robot actuation (ref. cloudgripper_env.py)
        next_pos = self._current_pos.copy()
        move_xy = (dx >= self._step_thresh) or (dy >= self._step_thresh)

        next_pos[0] = self._target_pos[0] if move_xy else next_pos[0]
        next_pos[1] = self._target_pos[1] if move_xy else next_pos[1]
        next_pos[2] = self._target_pos[2] if dz >= self._step_thresh else next_pos[2]
        next_pos[3] = self._target_pos[3] if dr >= self._step_thresh else next_pos[3]
        next_pos[4] = self._target_pos[4] if dg >= self._step_thresh else next_pos[4]

        ctrl = np.zeros(len(self.joint_names))
        for val, j_id, a_id in zip(next_pos, self.joint_names, self.actuator_names):

            joint = self._model.joint(j_id)
            actuator = self._model.actuator(a_id)

            # unnormalize target joint position
            next_pos_mj = self.unnormalize(val, joint)

            # unnormalized distance
            current_pos_mj = self._data.qpos[joint.qposadr[0]]
            error = next_pos_mj - current_pos_mj

            # distance per control step
            v_max = actuator.ctrlrange[1]
            deadband = v_max * self._control_timestep

            # can't converge closer than one control step --> hold still
            if abs(error) <= deadband:
                ctrl[actuator.id] = 0.0
            else:
                ctrl[actuator.id] = v_max if error > 0.0 else -v_max

        self._data.ctrl[:] = ctrl
        self._current_pos = next_pos


    def render(
        self,
        camera=None,
        *args,
        **kwargs,
    ):
        """Render image of a cloudgripper camera {"Camera_main" or "Camera_bottom"}.
        """
        camera = camera or self.main_camera
        return super().render(camera=camera, *args, **kwargs)

    def close(self):
        """Releases the offscreen renderer's GL context.

        mujoco.Renderer holds a GLFW context that it frees in __del__. If
        that happens during interpreter shutdown instead, module globals
        glfw's cleanup relies on may already be torn down, causing harmless
        but noisy "Exception ignored in: ... TypeError: 'NoneType' object is
        not callable" messages. Calling close() explicitly (e.g. at the end
        of a script) avoids that by freeing the context deterministically
        while the interpreter is still fully alive.
        """
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

