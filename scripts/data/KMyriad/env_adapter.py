"""Adapter presenting CloudGripper's gymnasium env the way Isaac Lab's env looks.

collect_particles expects:
  - obs["policy"] as a float32 tensor of shape [num_envs, num_features]
  - terminated / truncated as tensors
  - actions as tensors it can pass straight to step()

CloudGripper gives numpy, one env at a time, with obs split across two keys.
"""

import gymnasium as gym
import numpy as np
import torch

import environments.mj_cloudgripper  # noqa: F401  (registers the env ids)


# Workspace bounds in world metres, used to normalise the cube position
# so it shares a scale with the arm's [0, 1] state.
OBJ_LOW = np.array([-0.10, -0.08, 0.0], dtype=np.float32)
OBJ_HIGH = np.array([0.10, 0.08, 0.05], dtype=np.float32)


class MaxEntEnvAdapter:
    """Vectorised CloudGripper env with an Isaac-shaped interface.

    Observation layout (num_features = 8):
        [0:5] arm state    — x, y, z, rot, grip     (already [0, 1])
        [5:8] cube position — x, y, z               (normalised to [0, 1])
    """

    def __init__(self, num_envs, env_id="cloudgripper_mujoco/Tracking-v0",
                 device=None, normalise_object=True, **env_kwargs):
        self.num_envs = num_envs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalise_object = normalise_object
        self.epoch_options = None

        def make():
            return gym.make(env_id, **env_kwargs)

        self.venv = gym.vector.SyncVectorEnv([make for _ in range(num_envs)])

        single = self.venv.single_observation_space
        self.num_features = (single["state"].shape[0]
                             + single["object_position"].shape[0])
        self.num_actions = self.venv.single_action_space.shape[0]
        self.action_space = self.venv.single_action_space

    def _to_policy_obs(self, obs):
        """Concatenate the two observation keys into one flat tensor."""
        state = obs["state"] # shape [num_envs, 5]
        obj = obs["object_position"] # shape [num_envs, 3]
        if self.normalise_object:
            obj = (obj - OBJ_LOW) / (OBJ_HIGH - OBJ_LOW) # rescales to [0,1]
            obj = np.clip(obj, 0.0, 1.0)        # cube outside the workspace
                                                # collapses to the boundary, so
                                                # escaping it carries no entropy
        flat = np.concatenate([state, obj], axis=-1).astype(np.float32) #joins along the feature dim. 
        return {"policy": torch.as_tensor(flat, device=self.device)}

    def reset(self, seed=None, options=None):
        options = options or self.epoch_options
        obs, info = self.venv.reset(seed=seed, options=options)
        return self._to_policy_obs(obs), info

    def step(self, action):
        if torch.is_tensor(action): #The policy produces a tensor, but the env expects a numpy array.
            action = action.detach().cpu().numpy()
        action = action.astype(np.float32)

        obs, reward, terminated, truncated, info = self.venv.step(action) #all N envs are stepped in parallel, so the outputs are arrays of length N.

        as_t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=self.device)
        return (
            self._to_policy_obs(obs),
            as_t(reward, torch.float32),
            as_t(terminated, torch.bool),
            as_t(truncated, torch.bool),
            info,
        )

    def close(self):
        self.venv.close()