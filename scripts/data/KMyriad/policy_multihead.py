from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random


SAFE_EPS = 1e-6

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity,device="cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = []    
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(action).to(self.device),
            torch.FloatTensor(reward).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(done).unsqueeze(1).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)

# --- Torch Replay Buffer ---
class TorchReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim,num_envs, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0  # current size of buffer

        # Pre-allocate tensors
        self.states = torch.zeros((capacity,num_envs, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity,num_envs, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity,num_envs, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,num_envs, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity,num_envs, state_dim), dtype=torch.float32, device=device)

    def push(self, states, actions, rewards, next_states, dones):
        """
        Push a batch of transitions (from trajectories).
        Each input tensor must be [N, dim] — unsqueeze to save as collected by the head.
        """
        states = states.unsqueeze(0).to(self.device)
        actions = actions.unsqueeze(0).to(self.device)
        rewards = rewards.unsqueeze(0).to(self.device)
        next_states = next_states.unsqueeze(0).to(self.device)
        dones = dones.unsqueeze(0).to(self.device)

        batch_size = states.shape[0]
        idxs = (torch.arange(batch_size) + self.position) % self.capacity

        self.states[idxs] = states
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards.unsqueeze(-1)
        self.next_states[idxs] = next_states
        self.dones[idxs] = dones.type(torch.float32).unsqueeze(-1)
        #self.policy_id[idxs] = policy_id.unsqueeze(-1) if policy_id is not None and policy_id.ndim == 1 else policy_id

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size, device=None):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.states.device)
        states      = self.states[idxs]
        actions     = self.actions[idxs]
        rewards     = self.rewards[idxs]
        next_states = self.next_states[idxs]
        dones       = self.dones[idxs]
        if device is not None:
            states      = states.to(device, non_blocking=True)
            actions     = actions.to(device, non_blocking=True)
            rewards     = rewards.to(device, non_blocking=True)
            next_states = next_states.to(device, non_blocking=True)
            dones       = dones.to(device, non_blocking=True)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

# --- Multi-Head Linear ---
class MultiHeadLinear(nn.Module):
    def __init__(self, num_heads, in_features, out_features):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features

        # Independent weights & biases for each head
        self.weight = nn.Parameter(torch.Tensor(num_heads, in_features,out_features))
        self.bias = nn.Parameter(torch.Tensor(num_heads, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier init per head
        for i in range(self.num_heads):
            nn.init.xavier_uniform_(self.weight[i])
            nn.init.zeros_(self.bias[i])

    def forward(self, x):
        """Supports shared or per-head inputs."""
        # weight: [num_heads, in_features, out_features]
        # bias: [num_heads, out_features]
        if x.dim() == 2:
            # Shared trunk case: x is [batch, in_features]
            out = torch.einsum("bi,hio->bho", x, self.weight) + self.bias
            return out

        if x.dim() == 3:
            # Per-head features: x is [batch, num_heads, in_features]
            out = torch.einsum("bhi,hio->bho", x, self.weight) + self.bias.unsqueeze(0)
            return out

        raise ValueError("MultiHeadLinear expects input with rank 2 or 3")

# --- Policy Multihead Network ---
class PolicyMultiheadNetwork(nn.Module):
    def __init__(self, hidden_sizes,adapter_hidden, activation, num_envs,num_agents, 
                 state_dim, action_dim, action_space, latent_proj_dim=2,use_fixed_std=False):
        super(PolicyMultiheadNetwork, self).__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.activation = activation
        self.num_agents = num_agents
        self.use_fixed_std = use_fixed_std  # Toggle between fixed and state-dependent std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        envs_per_agent = max(1, num_envs // num_agents)
        env2agent = (torch.arange(num_envs, device=self.device) // envs_per_agent).clamp(max=num_agents - 1)
        self.env2agent = env2agent

        layers = []
        # Input Layer
        layers.extend((
            nn.Linear(state_dim, hidden_sizes[0]),
            self.activation()
        ))
        # Hidden Layers
        for i in range(len(hidden_sizes) - 1):
            layers.extend((
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                self.activation()
            ))

        self.net = nn.Sequential(*layers)

        # Lightweight adapters let each head specialize the shared embedding.
        #adapter_hidden = 256
        self.adapter_hidden = adapter_hidden
        self.head_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_sizes[-1], adapter_hidden),
                self.activation()
            )
            for _ in range(num_agents)
        ])
        
        # Multi-head linear layers (vectorized heads)
        self.fc_mean = MultiHeadLinear(num_agents,adapter_hidden, action_dim)
        self.fc_log_std = MultiHeadLinear(num_agents, adapter_hidden, action_dim)  # State-dependent std
        
        # Optional: fixed learnable std per head (activate with use_fixed_std=True)
        self.log_std_constant = nn.Parameter(torch.ones(num_agents, action_dim)*-2.0)

        # Learned projection for latent-space entropy (trunk output -> low-dim)
        # bias=True so the projection can match the offset of state-filter dims
        self.latent_proj = nn.Linear(hidden_sizes[-1], latent_proj_dim, bias=True)

        # Add these buffers for robust scaling
        # When action_dim > primitive action dim (chunked policy), tile to match
        primitive_scale = torch.tensor((action_space.high - action_space.low) 
                         / 2.0, dtype=torch.float32)
        primitive_bias = torch.tensor((action_space.high + action_space.low) 
                         / 2.0, dtype=torch.float32)
        num_repeats = action_dim // len(primitive_scale)
        self.register_buffer(
            "action_scale",
            primitive_scale.repeat(num_repeats)
        )
        self.register_buffer(
            "action_bias",
            primitive_bias.repeat(num_repeats)
        )

        self._detach_trunk = False  # when True, forward_select detaches trunk output

        self.initialize_weights()

    # ------------------------------------------------------------------
    #  Trunk / Head parameter helpers
    # ------------------------------------------------------------------
    def get_trunk_params(self):
        """Yield parameters belonging to the shared trunk (self.net)."""
        # add latent_proj to trunk params since it's shared across heads
        return list(self.net.parameters()) + list(self.latent_proj.parameters())

    def get_head_params(self):
        """Yield parameters belonging to head-specific layers."""
        from itertools import chain
        return chain(
            self.head_adapters.parameters(),
            self.fc_mean.parameters(),
            self.fc_log_std.parameters(),
            [self.log_std_constant],
        )

    @contextmanager
    def detached_trunk(self):
        """Context manager: inside this block, forward_select detaches the
        trunk output so that gradients only flow into head parameters."""
        self._detach_trunk = True
        try:
            yield
        finally:
            self._detach_trunk = False

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.xavier_uniform_(self.fc_log_std.weight)
        # log_std_constant initialized to 0.5 (std = exp(0.5) in exp space)

        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
        
        nn.init.xavier_uniform_(self.latent_proj.weight)
        nn.init.zeros_(self.latent_proj.bias)

    def init_multihead_layer(self,weight, bias):
        """
        weight: [num_heads, out_dim, in_dim]
        bias:   [num_heads, out_dim]
        """
        num_heads, out_dim, in_dim = weight.shape

        for h in range(num_heads):
            # Orthogonal init for each head
            nn.init.orthogonal_(weight[h])
            # Different bias offsets per head (spread out)
            nn.init.uniform_(bias[h], -0.5, 0.5)
    
    def linear_by_index_old(self,x, head_idx, weight, bias):
        # ... (docstring) ...
        # Select the per-row head parameters
        W_gathered = weight[head_idx]   # Shape: [N, in_features, out_features]
        
        # Transpose the last two dimensions for bmm
        W = W_gathered.transpose(1, 2)  # Shape: [N, out_features, in_features]
        
        b = bias[head_idx]              # Shape: [N, out]
        
        # Now the shapes are correct for bmm:
        # W:               [N, out, in]
        # x.unsqueeze(-1): [N, in, 1]
        out = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1) + b  # [N, out]
        return out
    
    def linear_by_index_o(self, x, head_idx, weight, bias):
        # Select the correct weights and biases for each head
        w = weight[head_idx]  # Shape: [N, in_features, out_features]
        b = bias[head_idx]    # Shape: [N, out_features]

        # Reshape x for batched matrix multiplication
        # x has shape [N, in_features], need to unsqueeze to [N, in_features, 1]
        x_unsqueezed = x.unsqueeze(-1)

        # Transpose w for multiplication (out x in)
        # The weight tensor needs to be [N, out_features, in_features]
        w_transposed = w.transpose(1, 2)

        # Perform batched matrix multiplication
        # torch.bmm([N, out, in], [N, in, 1]) -> [N, out, 1]
        out = torch.bmm(w_transposed, x_unsqueezed)

        # Remove the extra dimension and add the bias
        # Squeeze the last dimension to get [N, out_features]
        out = out.squeeze(-1) + b

        return out
    
    def linear_by_index(self, x, head_idx, weight, bias):
        """Apply per-head linear layers without materialising full [N, in, out] tensors."""
        out_features = weight.shape[-1]
        out = torch.empty((x.size(0), out_features), dtype=x.dtype, device=x.device)

        unique_heads = head_idx.unique(sorted=True)
        for hid in unique_heads:
            mask = head_idx == hid
            if mask.any():
                w = weight[hid].transpose(0, 1)  # [out, in] for F.linear
                b = bias[hid]
                out[mask] = F.linear(x[mask], w, b)

        return out

    def _apply_head_adapters(self, x):
        """Apply every head adapter to the shared embedding."""
        head_outputs = [adapter(x) for adapter in self.head_adapters]
        return torch.stack(head_outputs, dim=1)

    def _apply_head_adapter_select(self, x, head_idx):
        """Apply only the adapter corresponding to each row's head index."""
        out = torch.empty((x.size(0), self.adapter_hidden), dtype=x.dtype, device=x.device)
        for hid, adapter in enumerate(self.head_adapters):
            mask = head_idx == hid
            if mask.any():
                out[mask] = adapter(x[mask])
        return out

    def forward(self, state):
        """
        state: [batch, state_dim]
        returns: mean, log_std each of shape [batch, num_agents, action_dim]
        """
        x = self.net(state)  # [batch, hidden_sizes[-1]]
        head_features = self._apply_head_adapters(x)
        mean = self.fc_mean(head_features)  # [batch, num_agents, action_dim]
        
        if self.use_fixed_std:
            # Use fixed learnable std per head
            batch_size = mean.shape[0]
            log_std = self.log_std_constant.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_agents, action_dim]
        else:
            # Use state-dependent std (original behavior)
            log_std = self.fc_log_std(head_features)  # [batch, num_agents, action_dim]
            log_std = torch.tanh(log_std)
            log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        
        return mean, log_std
    
    def forward_select(self, states, head_idx):
        """
        states:   [N, state_dim]
        head_idx: [N] long (which head each row should use)
        returns:
            means:    [N, action_dim]
            log_std:  [N, action_dim]
        """
        x = self.net(states)  # [N, hidden]
        if self._detach_trunk:
            x = x.detach()
        adapted = self._apply_head_adapter_select(x, head_idx)
        means = self.linear_by_index(adapted, head_idx, self.fc_mean.weight, self.fc_mean.bias)
        
        if self.use_fixed_std:
            # Use fixed learnable std per head (gather by head_idx)
            log_std = self.log_std_constant[head_idx]  # [N, action_dim]
        else:
            # Use state-dependent std (original behavior)
            log_std = self.linear_by_index(adapted, head_idx, self.fc_log_std.weight, self.fc_log_std.bias)
            log_std = torch.tanh(log_std)
            log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        
        return means, log_std
    
    def sample_o(self, state):
        means, log_stds = self.forward(state)
        stds = log_stds.exp()

        normals = Normal(means, stds)               # [batch, num_agents, action_dim]
        x_t = normals.rsample()                     # reparam trick
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normals.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)   # [batch, num_agents, 1]

        mean_action = torch.tanh(means) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def sample_select_o(self, states, head_idx):
        """
        states:   [N, state_dim]
        head_idx: [N] long - which head each row uses
        returns:
            action:    [N, action_dim]
            log_prob:  [N, 1]
            mean_act:  [N, action_dim]
        """
        eps = 1e-6
        means, log_stds = self.forward_select(states, head_idx)  # [N, A] each
        stds = log_stds.exp()

        normal = Normal(means, stds)
        x_t = normal.rsample()              # [N, A]
        y_t = torch.tanh(x_t)               # [N, A]

        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)     # [N, A]
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(-1, keepdim=True)  # [N, 1]

        mean_action = torch.tanh(means) * self.action_scale + self.action_bias
        return action, log_prob, mean_action
    
    def sample(self, state):
        """
        state: [batch, state_dim]
        returns:
            action:      [batch, num_agents, action_dim]
            log_prob:    [batch, num_agents, 1]
            mean_action: [batch, num_agents, action_dim]
        """
        means, log_stds = self.forward(state)
        stds = log_stds.exp() 

        normals = Normal(means, stds)   # [B, H, A]
        x_t = normals.rsample()         # reparam trick
        y_t = torch.tanh(x_t)           # squash to [-1,1]

        # rescale to env action space
        action = y_t * self.action_scale + self.action_bias

        # log prob with tanh correction
        log_prob = normals.log_prob(x_t)             # [B, H, A]
        log_prob = log_prob.sum(-1, keepdim=True)    # sum over action dims
        log_prob -= torch.sum(torch.log(1 - y_t.pow(2) + 1e-6), dim=-1, keepdim=True)

        mean_action = torch.tanh(means) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def sample_select(self, states, head_idx):
        """
        states:   [N, state_dim]
        head_idx: [N] long - which head each row uses
        returns:
            action:    [N, action_dim]
            log_prob:  [N, 1]
            mean_act:  [N, action_dim]
        """
        eps = 1e-6
        means, log_stds = self.forward_select(states, head_idx)  # [N, A]
        stds = log_stds.exp()

        normal = Normal(means, stds)
        x_t = normal.rsample()   # [N, A]
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)     # [N, 1]
        log_prob -= torch.sum(torch.log(1 - y_t.pow(2) + eps), dim=-1, keepdim=True)

        mean_action = torch.tanh(means) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def get_log_p(self, states, actions, env2agent):
        """
        states:    [N, state_dim]
        actions:   [N, action_dim]  (rescaled actions)
        env2agent: [N] LongTensor mapping each row -> agent id
        """
        eps = 1e-6
        N = states.size(0)
        device = states.device

        # Invert rescaling
        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -1 + eps, 1 - eps)
        x_t = torch.atanh(y_t)

        means, log_stds = self.forward_select(states, env2agent)
        stds = log_stds.exp()

        # Normal distribution
        normal = Normal(means, stds)
        log_prob = normal.log_prob(x_t)

        # Correction for tanh squashing
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)

        # Sum over action dimensions
        log_prob = log_prob.sum(dim=-1)  # [N]

        return log_prob
    
    def get_log_p_select(self, states, actions, env2agent):
        """
        states:    [N, state_dim]
        actions:   [N, action_dim]  (rescaled)
        env2agent: [N] long (row -> head)
        returns:   [N]
        """
        eps = 1e-6

        # invert scaling & tanh
        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -1 + eps, 1 - eps)
        x_t = torch.atanh(y_t)

        means, log_stds = self.forward_select(states, env2agent)
        stds = log_stds.exp()

        normal = Normal(means, stds)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(dim=-1)  # [N]
        return log_prob


# ======================================================================
#  Recurrent Multihead policy: GRU trunk, same head API
# ======================================================================
class PolicyRecurrentMultiheadNetwork(nn.Module):
    """Drop-in replacement for PolicyMultiheadNetwork where the shared
    feedforward trunk is replaced by a GRU-based recurrent trunk.

    The external API (forward, forward_select, sample, sample_select,
    get_log_p, get_log_p_select, get_trunk_params, get_head_params,
    detached_trunk) is identical to PolicyMultiheadNetwork so that
    training code can swap between the two with a single flag.

    Hidden-state convention
    -----------------------
    * ``reset_hidden(batch_size)`` initialises the hidden state to zeros.
    * ``forward`` / ``forward_select`` / ``sample`` / ``sample_select``
      all accept an optional ``hidden`` argument and return the updated
      hidden state as an extra output (appended after the original
      return values).
    * When ``hidden=None`` is passed (default) the network uses an
      internal zero-initialised state — this makes the API backward-
      compatible with code that ignores the recurrent state.
    """

    def __init__(self, hidden_sizes, adapter_hidden, activation, num_envs,
                 num_agents, state_dim, action_dim, action_space,
                 latent_proj_dim=2, use_fixed_std=False,
                 num_rnn_layers=1):
        super().__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.activation = activation
        self.num_agents = num_agents
        self.use_fixed_std = use_fixed_std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        envs_per_agent = max(1, num_envs // num_agents)
        env2agent = (torch.arange(num_envs, device=self.device)
                     // envs_per_agent).clamp(max=num_agents - 1)
        self.env2agent = env2agent

        # --- Recurrent trunk ---
        self.rnn_hidden_size = hidden_sizes[-1]
        self.num_rnn_layers = num_rnn_layers

        # Linear input projection: state_dim -> hidden_sizes[0]
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            self.activation(),
        )

        # GRU core: hidden_sizes[0] -> rnn_hidden_size
        self.gru = nn.GRU(
            input_size=hidden_sizes[0],
            hidden_size=self.rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
        )

        # Post-GRU readout: maps bounded tanh output [-1,1] back to
        # ReLU-activated space so downstream adapters see the same
        # distribution as the feedforward trunk.
        self.post_gru = nn.Sequential(
            nn.LayerNorm(self.rnn_hidden_size),
            nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size),
            self.activation(),
        )

        # --- Heads (identical to PolicyMultiheadNetwork) ---
        self.adapter_hidden = adapter_hidden
        self.head_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.rnn_hidden_size, adapter_hidden),
                self.activation(),
            )
            for _ in range(num_agents)
        ])

        self.fc_mean = MultiHeadLinear(num_agents, adapter_hidden, action_dim)
        self.fc_log_std = MultiHeadLinear(num_agents, adapter_hidden, action_dim)

        self.log_std_constant = nn.Parameter(
            torch.ones(num_agents, action_dim) * -2.0)

        self.latent_proj = nn.Linear(self.rnn_hidden_size, latent_proj_dim,
                                     bias=True)

        # Action scaling buffers
        primitive_scale = torch.tensor(
            (action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        primitive_bias = torch.tensor(
            (action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        num_repeats = action_dim // len(primitive_scale)
        self.register_buffer("action_scale",
                             primitive_scale.repeat(num_repeats))
        self.register_buffer("action_bias",
                             primitive_bias.repeat(num_repeats))

        self._detach_trunk = False

        self.initialize_weights()

    # ------------------------------------------------------------------
    #  Hidden-state helpers
    # ------------------------------------------------------------------
    def reset_hidden(self, batch_size, device=None):
        """Return a zero hidden state: [num_rnn_layers, batch_size, rnn_hidden_size]."""
        device = device or self.device
        return torch.zeros(self.num_rnn_layers, batch_size,
                           self.rnn_hidden_size, device=device)

    # ------------------------------------------------------------------
    #  Trunk / Head parameter helpers
    # ------------------------------------------------------------------
    def get_trunk_params(self):
        return (list(self.input_proj.parameters())
                + list(self.gru.parameters())
                + list(self.post_gru.parameters())
                + list(self.latent_proj.parameters()))

    def get_head_params(self):
        from itertools import chain
        return chain(
            self.head_adapters.parameters(),
            self.fc_mean.parameters(),
            self.fc_log_std.parameters(),
            [self.log_std_constant],
        )

    @contextmanager
    def detached_trunk(self):
        self._detach_trunk = True
        try:
            yield
        finally:
            self._detach_trunk = False

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.xavier_uniform_(self.fc_log_std.weight)

        for layer in self.input_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        # GRU uses orthogonal init by default; reinforce it
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for layer in self.post_gru:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.latent_proj.weight)
        nn.init.zeros_(self.latent_proj.bias)

    # ------------------------------------------------------------------
    #  Trunk forward
    # ------------------------------------------------------------------
    def _trunk(self, state, hidden=None):
        """Run the recurrent trunk.

        Parameters
        ----------
        state : Tensor [N, state_dim] or [N, T, state_dim]
            If 2-D, treated as a single time-step (T=1).
        hidden : Tensor [num_layers, N, rnn_hidden] or None
            If None a zero state is created.

        Returns
        -------
        x : Tensor [N, rnn_hidden]   (last time-step output)
        h : Tensor [num_layers, N, rnn_hidden]
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)          # [N, 1, state_dim]

        proj = self.input_proj(state)           # [N, T, hidden_sizes[0]]

        if hidden is None:
            hidden = self.reset_hidden(state.size(0), device=state.device)

        gru_out, h_n = self.gru(proj, hidden)   # gru_out: [N, T, rnn_hidden]
        x = gru_out[:, -1, :]                   # last step: [N, rnn_hidden]
        x = self.post_gru(x)                    # LayerNorm + Linear + ReLU
        return x, h_n

    # ------------------------------------------------------------------
    #  Adapter helpers (same as PolicyMultiheadNetwork)
    # ------------------------------------------------------------------
    def _apply_head_adapters(self, x):
        head_outputs = [adapter(x) for adapter in self.head_adapters]
        return torch.stack(head_outputs, dim=1)

    def _apply_head_adapter_select(self, x, head_idx):
        out = torch.empty((x.size(0), self.adapter_hidden),
                          dtype=x.dtype, device=x.device)
        for hid, adapter in enumerate(self.head_adapters):
            mask = head_idx == hid
            if mask.any():
                out[mask] = adapter(x[mask])
        return out

    def linear_by_index(self, x, head_idx, weight, bias):
        out_features = weight.shape[-1]
        out = torch.empty((x.size(0), out_features),
                          dtype=x.dtype, device=x.device)
        unique_heads = head_idx.unique(sorted=True)
        for hid in unique_heads:
            mask = head_idx == hid
            if mask.any():
                w = weight[hid].transpose(0, 1)
                b = bias[hid]
                out[mask] = F.linear(x[mask], w, b)
        return out

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, state, hidden=None):
        """
        state: [batch, state_dim]
        returns: mean, log_std each [batch, num_agents, action_dim], hidden
        """
        x, h_n = self._trunk(state, hidden)

        head_features = self._apply_head_adapters(x)
        mean = self.fc_mean(head_features)

        if self.use_fixed_std:
            batch_size = mean.shape[0]
            log_std = (self.log_std_constant.unsqueeze(0)
                       .expand(batch_size, -1, -1))
        else:
            log_std = self.fc_log_std(head_features)
            log_std = torch.tanh(log_std)
            log_std = (self.LOG_STD_MIN
                       + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN)
                       * (log_std + 1))

        return mean, log_std, h_n

    def forward_select(self, states, head_idx, hidden=None):
        """
        states:   [N, state_dim]
        head_idx: [N] long
        returns:  means [N, A], log_std [N, A], hidden
        """
        x, h_n = self._trunk(states, hidden)
        if self._detach_trunk:
            x = x.detach()
        adapted = self._apply_head_adapter_select(x, head_idx)
        means = self.linear_by_index(adapted, head_idx,
                                     self.fc_mean.weight, self.fc_mean.bias)

        if self.use_fixed_std:
            log_std = self.log_std_constant[head_idx]
        else:
            log_std = self.linear_by_index(adapted, head_idx,
                                           self.fc_log_std.weight,
                                           self.fc_log_std.bias)
            log_std = torch.tanh(log_std)
            log_std = (self.LOG_STD_MIN
                       + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN)
                       * (log_std + 1))

        return means, log_std, h_n

    # ------------------------------------------------------------------
    #  Sample
    # ------------------------------------------------------------------
    def sample(self, state, hidden=None):
        """
        state: [batch, state_dim]
        returns:
            action      [batch, num_agents, action_dim]
            log_prob    [batch, num_agents, 1]
            mean_action [batch, num_agents, action_dim]
            hidden      [num_layers, batch, rnn_hidden]
        """
        means, log_stds, h_n = self.forward(state, hidden)
        stds = log_stds.exp()

        normals = Normal(means, stds)
        x_t = normals.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = normals.log_prob(x_t).sum(-1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - y_t.pow(2) + 1e-6),
                              dim=-1, keepdim=True)

        mean_action = torch.tanh(means) * self.action_scale + self.action_bias
        return action, log_prob, mean_action, h_n

    def sample_select(self, states, head_idx, hidden=None):
        """
        states:   [N, state_dim]
        head_idx: [N] long
        returns:  action [N, A], log_prob [N, 1], mean_act [N, A], hidden
        """
        eps = 1e-6
        means, log_stds, h_n = self.forward_select(states, head_idx, hidden)
        stds = log_stds.exp()

        normal = Normal(means, stds)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - y_t.pow(2) + eps),
                              dim=-1, keepdim=True)

        mean_action = torch.tanh(means) * self.action_scale + self.action_bias
        return action, log_prob, mean_action, h_n

    # ------------------------------------------------------------------
    #  Log probability
    # ------------------------------------------------------------------
    def get_log_p(self, states, actions, env2agent, hidden=None):
        """
        states:    [N, state_dim]
        actions:   [N, action_dim] (rescaled)
        env2agent: [N] long
        returns:   log_prob [N], hidden
        """
        eps = 1e-6

        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -1 + eps, 1 - eps)
        x_t = torch.atanh(y_t)

        means, log_stds, h_n = self.forward_select(states, env2agent, hidden)
        stds = log_stds.exp()

        normal = Normal(means, stds)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(dim=-1)
        return log_prob, h_n

    def get_log_p_select(self, states, actions, env2agent, hidden=None):
        """
        states:    [N, state_dim]
        actions:   [N, action_dim] (rescaled)
        env2agent: [N] long
        returns:   log_prob [N], hidden
        """
        eps = 1e-6

        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -1 + eps, 1 - eps)
        x_t = torch.atanh(y_t)

        means, log_stds, h_n = self.forward_select(states, env2agent, hidden)
        stds = log_stds.exp()

        normal = Normal(means, stds)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(dim=-1)
        return log_prob, h_n


# ======================================================================
#  Siamese (fully parallel) policy: N independent networks, same API
# ======================================================================
class PolicySiameseNetwork(nn.Module):
    """Drop-in replacement for PolicyMultiheadNetwork where every agent
    owns a completely independent network (trunk + mean/log_std heads).
    No weight sharing at all.

    The external API (forward, forward_select, sample, sample_select,
    get_log_p, get_log_p_select, get_trunk_params, get_head_params,
    detached_trunk) is identical to PolicyMultiheadNetwork so that
    training code can swap between the two with a single flag.
    """

    def __init__(self, hidden_sizes, adapter_hidden, activation, num_envs,
                 num_agents, state_dim, action_dim, action_space,
                 latent_proj_dim=2, use_fixed_std=False):
        super().__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.activation = activation
        self.num_agents = num_agents
        self.use_fixed_std = use_fixed_std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        envs_per_agent = max(1, num_envs // num_agents)
        env2agent = (torch.arange(num_envs, device=self.device) // envs_per_agent).clamp(max=num_agents - 1)
        self.env2agent = env2agent

        # --- Build N independent trunks ---
        self.trunks = nn.ModuleList()
        for _ in range(num_agents):
            layers = []
            layers.extend((nn.Linear(state_dim, hidden_sizes[0]), activation()))
            for i in range(len(hidden_sizes) - 1):
                layers.extend((nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), activation()))
            self.trunks.append(nn.Sequential(*layers))

        # --- Per-agent adapter + mean/log_std heads ---
        self.adapter_hidden = adapter_hidden
        self.head_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_sizes[-1], adapter_hidden), activation())
            for _ in range(num_agents)
        ])
        self.mean_heads = nn.ModuleList([
            nn.Linear(adapter_hidden, action_dim)
            for _ in range(num_agents)
        ])
        self.log_std_heads = nn.ModuleList([
            nn.Linear(adapter_hidden, action_dim)
            for _ in range(num_agents)
        ])

        # Fixed learnable std per agent (used when use_fixed_std=True)
        self.log_std_constant = nn.Parameter(torch.ones(num_agents, action_dim) * -2.0)

        # Per-agent latent projections (one per trunk)
        self.latent_projs = nn.ModuleList([
            nn.Linear(hidden_sizes[-1], latent_proj_dim, bias=True)
            for _ in range(num_agents)
        ])

        # Action scaling buffers (same as multihead)
        primitive_scale = torch.tensor(
            (action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        primitive_bias = torch.tensor(
            (action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        num_repeats = action_dim // len(primitive_scale)
        self.register_buffer("action_scale", primitive_scale.repeat(num_repeats))
        self.register_buffer("action_bias", primitive_bias.repeat(num_repeats))

        self._detach_trunk = False
        self.initialize_weights()

    # ------------------------------------------------------------------
    #  Weight init
    # ------------------------------------------------------------------
    def initialize_weights(self):
        for trunk in self.trunks:
            for layer in trunk:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
        for head in self.mean_heads:
            nn.init.xavier_uniform_(head.weight)
        for head in self.log_std_heads:
            nn.init.xavier_uniform_(head.weight)
        for adapter in self.head_adapters:
            for layer in adapter:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
        for proj in self.latent_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    # ------------------------------------------------------------------
    #  Trunk / Head parameter helpers (same interface as multihead)
    # ------------------------------------------------------------------
    def get_trunk_params(self):
        """Return all trunk + latent_proj parameters (list)."""
        params = []
        for trunk in self.trunks:
            params.extend(trunk.parameters())
        for proj in self.latent_projs:
            params.extend(proj.parameters())
        return params

    def get_head_params(self):
        """Return all head-specific parameters."""
        from itertools import chain
        return chain(
            self.head_adapters.parameters(),
            *[h.parameters() for h in self.mean_heads],
            *[h.parameters() for h in self.log_std_heads],
            [self.log_std_constant],
        )

    @contextmanager
    def detached_trunk(self):
        self._detach_trunk = True
        try:
            yield
        finally:
            self._detach_trunk = False

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, state):
        """
        state: [batch, state_dim]
        returns: mean [batch, num_agents, action_dim],
                 log_std [batch, num_agents, action_dim]
        """
        means = []
        log_stds = []
        for h in range(self.num_agents):
            x = self.trunks[h](state)                   # [B, hidden]
            adapted = self.head_adapters[h](x)           # [B, adapter_hidden]
            means.append(self.mean_heads[h](adapted))    # [B, A]
            if self.use_fixed_std:
                log_stds.append(
                    self.log_std_constant[h].unsqueeze(0).expand(state.size(0), -1))
            else:
                ls = self.log_std_heads[h](adapted)
                ls = torch.tanh(ls)
                ls = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (ls + 1)
                log_stds.append(ls)

        mean = torch.stack(means, dim=1)       # [B, H, A]
        log_std = torch.stack(log_stds, dim=1) # [B, H, A]
        return mean, log_std

    def forward_select(self, states, head_idx):
        """
        states:   [N, state_dim]
        head_idx: [N] long
        returns:  means [N, A], log_std [N, A]
        """
        N = states.size(0)
        action_dim = self.mean_heads[0].out_features
        means = torch.empty(N, action_dim, dtype=states.dtype, device=states.device)
        log_stds = torch.empty_like(means)

        for hid in range(self.num_agents):
            mask = head_idx == hid
            if not mask.any():
                continue
            x = self.trunks[hid](states[mask])
            if self._detach_trunk:
                x = x.detach()
            adapted = self.head_adapters[hid](x)
            means[mask] = self.mean_heads[hid](adapted)
            if self.use_fixed_std:
                log_stds[mask] = self.log_std_constant[hid]
            else:
                ls = self.log_std_heads[hid](adapted)
                ls = torch.tanh(ls)
                ls = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (ls + 1)
                log_stds[mask] = ls

        return means, log_stds

    # ------------------------------------------------------------------
    #  Sample
    # ------------------------------------------------------------------
    def sample(self, state):
        """
        state: [batch, state_dim]
        returns:
            action      [batch, num_agents, action_dim]
            log_prob    [batch, num_agents, 1]
            mean_action [batch, num_agents, action_dim]
        """
        means, log_stds = self.forward(state)
        stds = log_stds.exp()

        normals = Normal(means, stds)
        x_t = normals.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normals.log_prob(x_t).sum(-1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - y_t.pow(2) + 1e-6), dim=-1, keepdim=True)

        mean_action = torch.tanh(means) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def sample_select(self, states, head_idx):
        """
        states:   [N, state_dim]
        head_idx: [N] long
        returns:  action [N, A], log_prob [N, 1], mean_act [N, A]
        """
        eps = 1e-6
        means, log_stds = self.forward_select(states, head_idx)
        stds = log_stds.exp()

        normal = Normal(means, stds)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - y_t.pow(2) + eps), dim=-1, keepdim=True)

        mean_action = torch.tanh(means) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    # ------------------------------------------------------------------
    #  Log probability
    # ------------------------------------------------------------------
    def get_log_p(self, states, actions, env2agent):
        """
        states:    [N, state_dim]
        actions:   [N, action_dim] (rescaled)
        env2agent: [N] long
        returns:   [N]
        """
        eps = 1e-6
        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -1 + eps, 1 - eps)
        x_t = torch.atanh(y_t)

        means, log_stds = self.forward_select(states, env2agent)
        stds = log_stds.exp()

        normal = Normal(means, stds)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(dim=-1)
        return log_prob

    def get_log_p_select(self, states, actions, env2agent):
        """
        states:    [N, state_dim]
        actions:   [N, action_dim] (rescaled)
        env2agent: [N] long
        returns:   [N]
        """
        eps = 1e-6
        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -1 + eps, 1 - eps)
        x_t = torch.atanh(y_t)

        means, log_stds = self.forward_select(states, env2agent)
        stds = log_stds.exp()

        normal = Normal(means, stds)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(dim=-1)
        return log_prob


# --- Q-Network ---
class SingleQNetwork(nn.Module):
    def __init__(self, hidden_sizes, state_dim, action_dim, activation=nn.ReLU):
        super().__init__()
        input_dim = state_dim + action_dim

        # Q1
        layers1 = []
        layers1.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers1.append(activation())
        for i in range(len(hidden_sizes) - 1):
            layers1.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers1.append(activation())
        layers1.append(nn.Linear(hidden_sizes[-1], 1))
        self.q1_net = nn.Sequential(*layers1)

        # Q2
        layers2 = []
        layers2.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers2.append(activation())
        for i in range(len(hidden_sizes) - 1):
            layers2.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers2.append(activation())
        layers2.append(nn.Linear(hidden_sizes[-1], 1))
        self.q2_net = nn.Sequential(*layers2)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)   # [N, S+A]
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

# --- Multi-Head Q-Network ---
class MultiHeadQNetwork(nn.Module):
    """
    Shared trunk + per-head adapters for Q1/Q2.
    Fully vectorized: no Python loop over heads in forward().
    Uses MultiHeadLinear (same as your policy) and an indexed batched linear.
    """
    def __init__(self, hidden_sizes, state_dim, action_dim, num_heads, adapter_hidden=256, activation=nn.ReLU):
        super().__init__()
        self.num_heads = num_heads
        self.activation = activation
        self.adapter_hidden = adapter_hidden

        input_dim = state_dim + action_dim

        # ---- Shared trunk (heavy) ----
        layers = [nn.Linear(input_dim, hidden_sizes[0]), self.activation()]
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self.activation()]
        self.trunk = nn.Sequential(*layers)
        self.trunk_out = hidden_sizes[-1]

        # ---- Per-head adapters (light) for Q1 ----
        self.q1_h1 = MultiHeadLinear(num_heads, self.trunk_out, adapter_hidden)  # head-specific hidden
        self.q1_h2 = MultiHeadLinear(num_heads, adapter_hidden, 1)               # head-specific output

        # ---- Per-head adapters (light) for Q2 ----
        self.q2_h1 = MultiHeadLinear(num_heads, self.trunk_out, adapter_hidden)
        self.q2_h2 = MultiHeadLinear(num_heads, adapter_hidden, 1)

        self.act = self.activation()

    @staticmethod
    def _linear_by_index_o(x, head_idx, weight, bias):
        """
        Vectorized per-head linear:
        x:       [N, in_features]
        head_idx:[N] (long)
        weight:  [H, in_features, out_features]
        bias:    [H, out_features]
        returns: [N, out_features]
        """
        # Gather per-row params
        w = weight          # [N, in, out]
        b = bias[head_idx]            # [N, out]
        # (x @ w) + b with batched mm
        out = torch.bmm(x.unsqueeze(1), w).squeeze(1) + b  # [N, out]
        return out

    def _linear_by_index(self,x, head_idx, weight, bias):
        """
        Apply per-head linear layers without materialising [N, in, out].
        x:       [N, in_features]
        head_idx:[N] long
        weight:  [H, in_features, out_features]
        bias:    [H, out_features]
        returns: [N, out_features]
        """
        out_features = weight.shape[-1]
        out = torch.empty((x.size(0), out_features), dtype=x.dtype, device=x.device)

        unique_heads = head_idx.unique(sorted=True)
        for hid in unique_heads:
            mask = head_idx == hid
            if mask.any():
                w = weight[hid].transpose(0, 1)  # [out, in] for F.linear
                b = bias[hid]
                out[mask] = F.linear(x[mask], w, b)

        return out

    def forward(self, state, action, head_idx):
        """
        state:    [N, state_dim]
        action:   [N, action_dim]
        head_idx: [N] long (row -> head)
        returns:  q1, q2 each [N, 1]
        """
        sa = torch.cat([state, action], dim=-1)     # [N, S+A]
        shared = self.trunk(sa)                     # [N, trunk_out]

        # ---- Q1 path (no loops) ----
        q1_h = self._linear_by_index(shared, head_idx, self.q1_h1.weight, self.q1_h1.bias)   # [N, H]
        q1_h = self.act(q1_h)
        q1    = self._linear_by_index(q1_h, head_idx, self.q1_h2.weight, self.q1_h2.bias)    # [N, 1]

        # ---- Q2 path (no loops) ----
        q2_h = self._linear_by_index(shared, head_idx, self.q2_h1.weight, self.q2_h1.bias)   # [N, H]
        q2_h = self.act(q2_h)
        q2    = self._linear_by_index(q2_h, head_idx, self.q2_h2.weight, self.q2_h2.bias)    # [N, 1]

        return q1, q2

# --- Discretizer ---
class Discretizer:
    def __init__(self, features_ranges, bins_sizes, lambda_transform=None, device='cpu'):
        assert len(features_ranges) == len(bins_sizes)

        self.num_features = len(features_ranges)
        self.feature_ranges = features_ranges
        self.bins_sizes = bins_sizes
        self.device = device

        self.bins = [
            torch.linspace(start, end, steps=bins + 1, device=self.device)[1:-1]
            for (start, end), bins in zip(features_ranges, bins_sizes)
        ]

        self.lambda_transform = lambda_transform

    def discretize(self, features):
        """
        features: torch.Tensor of shape (num_features,)
        returns: tuple of indices for each feature bin
        """
        if self.lambda_transform:
            features = self.lambda_transform(features)

        assert isinstance(features, torch.Tensor), "Input must be a torch.Tensor"
        features = features.to(self.device)

        return tuple(
            torch.bucketize(features[i], self.bins[i])
            for i in range(self.num_features)
        )

    def get_empty_mat(self):
        return torch.zeros(*self.bins_sizes, device=self.device)

def init_single_policy(env,obs_dim, act_dim, hidden_sizes=[64, 64], activation=torch.nn.ReLU,device='cuda'):
    tmp_policy = PolicyNetwork(hidden_sizes=hidden_sizes,activation=activation, state_dim=obs_dim, action_dim=act_dim, action_space=env.cfg.action_space).to(device)
    tmp_policy = train_supervised(env, tmp_policy)
    return tmp_policy

def soft_update_target_net(q_target_net, q_net, TAU=0.005):
    for target_param, q_param in zip(q_target_net.parameters(), q_net.parameters()):
        target_param.data.copy_(TAU * q_param.data + (1.0 - TAU) * target_param.data)

def train_supervised(env, policy, train_steps=100, batch_size=5000):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.00025)
    dict_like_obs = isinstance(env.observation_manager.compute(), dict)

    for _ in range(train_steps):
        optimizer.zero_grad()

        if dict_like_obs: # In this case, it's likely a gymnasium Dict space
            states = [env.observation_manager.compute()["policy"][0] for _ in range(1000)]
        else:
            states = [env.observation_manager.compute()["policy"][:env.observation_manager.group_obs_dim["policy"][0]] for _ in range(1000)]

        states = torch.stack(states)
        actions = policy(states)[0]
        pseudo_targets = torch.zeros_like(actions)
        loss = torch.mean((actions - pseudo_targets) ** 2)

        loss.backward()
        optimizer.step()

    return policy

def safe_atanh(x: torch.Tensor, eps: float = SAFE_EPS) -> torch.Tensor:
    # Clamp inside (-1, 1) to avoid NaNs
    return 0.5 * torch.log((1 + torch.clamp(x, -1 + eps, 1 - eps)) /
                           (1 - torch.clamp(x, -1 + eps, 1 - eps)))