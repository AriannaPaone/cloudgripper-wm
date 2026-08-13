from gymnasium.envs.registration import register

register(
    id="cloudgripper_mujoco/Tracking-v0",
    entry_point="environments.mj_cloudgripper.cloudgripper_mujoco_tracking_env:CloudgripperMuJoCoTracking",
    max_episode_steps=100,
)
