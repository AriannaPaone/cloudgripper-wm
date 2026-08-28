import numpy as np
import torch
from stable_worldmodel.policy import BasePolicy

from scripts.data.KMyriad.policy_multihead import PolicyMultiheadNetwork
from scripts.data.KMyriad.env_adapter import OBJ_LOW, OBJ_HIGH


class MaxEntPolicy(BasePolicy):
    """Runs a trained MaxEnt multihead policy during data collection."""

    def __init__(self, checkpoint, num_agents, num_envs,
                 hidden_sizes=(64, 64), latent_proj_dim=2, agent_id=None, seed=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = num_agents
        self.agent_id = agent_id      # None = spread envs across heads
        self.num_envs = num_envs
        self._ckpt = checkpoint
        self._hidden = list(hidden_sizes)
        self.net = None
        self._latent_dim = latent_proj_dim

    def set_env(self, envs):
        self.net = PolicyMultiheadNetwork(
            hidden_sizes=self._hidden, adapter_hidden=self._hidden[-1],
            activation=torch.nn.ReLU, num_envs=self.num_envs,
            num_agents=self.num_agents, state_dim=8, action_dim=5,
            action_space=envs.single_action_space, latent_proj_dim = self._latent_dim,
        ).to(self.device)
        self.net.load_state_dict(torch.load(self._ckpt, map_location=self.device))
        self.net.eval()

        if self.agent_id is None:
            per = max(1, self.num_envs // self.num_agents)
            self.head = (torch.arange(self.num_envs, device=self.device) // per
                         ).clamp(max=self.num_agents - 1)
        else:
            self.head = torch.full((self.num_envs,), self.agent_id,
                                   dtype=torch.long, device=self.device)

    def get_action(self, infos):
        state = np.asarray(infos["state"]).reshape(self.num_envs, -1)
        obj = np.asarray(infos["object_position"]).reshape(self.num_envs, -1)
        obj = (obj - OBJ_LOW) / (OBJ_HIGH - OBJ_LOW)
        obs = torch.as_tensor(np.concatenate([state, obj], -1),
                              dtype=torch.float32, device=self.device)

        with torch.no_grad():
            actions, _, _ = self.net.sample(obs)          # [E, H, A]
            a = actions[torch.arange(self.num_envs, device=self.device), self.head]
        return a.cpu().numpy()