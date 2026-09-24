"""Two-phase training with split-env rollout.

Phase 1 (trunk):  envs 0..N_trunk-1 → all head 0 → optimize trunk + head 0
Phase 2 (heads):  envs N_trunk..N_total-1 → heads 1..N-1 → trunk frozen, optimize heads

Single rollout, both phases have correct IS ratios.
"""

import sys, os
sys.path.append(os.getcwd())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from stable_worldmodel.envs.two_room import env
import torch
import scipy
import scipy.special
import time
import math
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from .estimators import knn_entropy_estimation_scipy, get_heatmap_fast, knn_entropy_estimation_torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Env-to-agent mapping builder
# ======================================================================
def build_env2agent_twophase(num_trunk_envs, num_head_envs, num_agents, device):
    """Build env2agent for the 2K-env scheme.

    Args:
        num_trunk_envs: number of envs assigned to head 0 (Phase 1).
        num_head_envs:  number of envs split across heads 1..N-1 (Phase 2).
        num_agents:     total agent count N.

    Returns:
        env2agent: [num_trunk_envs + num_head_envs] long tensor.
            First num_trunk_envs entries are 0.
            Remaining num_head_envs entries are evenly split across 1..N-1.
    """
    # Phase 1: all trunk envs → head 0
    trunk_map = torch.zeros(num_trunk_envs, dtype=torch.long, device=device)

    # Phase 2: split across heads 1..N-1
    num_diversity_agents = num_agents - 1
    envs_per_head = num_head_envs // num_diversity_agents
    remainder = num_head_envs % num_diversity_agents

    head_map_parts = []
    for h in range(num_diversity_agents):
        count = envs_per_head + (1 if h < remainder else 0)
        head_map_parts.append(torch.full((count,), h + 1, dtype=torch.long, device=device))
    head_map = torch.cat(head_map_parts)

    return torch.cat([trunk_map, head_map])


# ======================================================================
#  Collection (reuses the standard collect_particles logic)
# ======================================================================
def collect_particles(env, policy, num_trajectories, trajectory_length,
                      num_features, num_actions, num_agents, num_envs, env2agent):
    """Collect trajectories using sample_select for efficiency."""
    assert num_agents <= num_envs

    states = torch.zeros((num_trajectories, trajectory_length + 1, num_envs, num_features),
                         dtype=torch.float32, device=device)
    actions = torch.zeros((num_trajectories, trajectory_length, num_envs, num_actions),
                          dtype=torch.float32, device=device)
    real_traj_lengths = torch.full((num_trajectories, 1, num_envs),
                                   trajectory_length, dtype=torch.int32, device=device)
    env_idx = torch.arange(num_envs, device=device)
    policy.eval()
    for traj in tqdm(range(num_trajectories)):
        s, _ = env.reset()
        for t in range(trajectory_length):
            states[traj, t] = s["policy"]

            with torch.no_grad():
                
                all_actions, all_logp, all_mean = policy.sample(s["policy"])  # [num_envs, num_agents, action_dim]
                a = all_actions[env_idx, env2agent]   # -> [num_envs, action_dim]

            actions[traj, t] = a
            s, _, terminated, truncated, _ = env.step(a)

            if terminated.any() or truncated.any():
                mask = terminated | truncated
                already_done = real_traj_lengths[traj, 0] < trajectory_length
                newly_done = mask & ~already_done
                real_traj_lengths[traj, 0] = torch.where(newly_done, t + 1, real_traj_lengths[traj, 0])

        states[traj, t + 1] = s["policy"]

    return states, actions, real_traj_lengths


def collect_particles_recurrent(env, policy, num_trajectories, trajectory_length,
                                num_features, num_actions, num_agents, num_envs, env2agent):
    """Collect trajectories with a recurrent policy (PolicyRecurrentMultiheadNetwork).

    Carries the GRU hidden state across timesteps within each trajectory and
    resets it at the start of each trajectory and for individual envs that
    terminate / truncate mid-trajectory.
    """
    assert num_agents <= num_envs

    states = torch.zeros((num_trajectories, trajectory_length + 1, num_envs, num_features),
                         dtype=torch.float32, device=device)
    actions = torch.zeros((num_trajectories, trajectory_length, num_envs, num_actions),
                          dtype=torch.float32, device=device)
    real_traj_lengths = torch.full((num_trajectories, 1, num_envs),
                                   trajectory_length, dtype=torch.int32, device=device)
    env_idx = torch.arange(num_envs, device=device)
    policy.eval()
    for traj in tqdm(range(num_trajectories)):
        s, _ = env.reset()
        hidden = policy.reset_hidden(num_envs)  # [num_layers, num_envs, rnn_hidden]

        for t in range(trajectory_length):
            states[traj, t] = s["policy"]

            with torch.no_grad():
                all_actions, all_logp, all_mean, hidden = policy.sample(s["policy"], hidden)
                a = all_actions[env_idx, env2agent]   # -> [num_envs, action_dim]

            actions[traj, t] = a
            s, _, terminated, truncated, _ = env.step(a)

            if terminated.any() or truncated.any():
                mask = terminated | truncated
                # Reset hidden state for done envs
                done_idx = mask.squeeze(-1) if mask.dim() > 1 else mask
                hidden[:, done_idx, :] = 0.0

                already_done = real_traj_lengths[traj, 0] < trajectory_length
                newly_done = mask & ~already_done
                real_traj_lengths[traj, 0] = torch.where(newly_done, t + 1, real_traj_lengths[traj, 0])

        states[traj, t + 1] = s["policy"]

    return states, actions, real_traj_lengths


def unpack_results(res):
    states, actions, real_traj_lengths = res
    states = states.contiguous()
    actions = actions.contiguous()
    real_traj_lengths = real_traj_lengths.squeeze(1).contiguous()
    return states, actions, real_traj_lengths


# ======================================================================
#  k-NN distances (state-space)
# ======================================================================
def compute_knn(states, states_filter, k):
    """Compute k-NN on flattened states[:, :-1, :, :] for a subset of envs.

    Args:
        states: [B, T+1, H_sub, D]
        states_filter: list of int (obs dims for k-NN)
        k: int

    Returns:
        distances: [N, k+1] float32 CPU tensor
        indices:   [N, k+1] int64 CPU tensor
    """
    num_features = states.shape[-1]
    valid_states = states[:, :-1, :, :]
    flat_states = valid_states.reshape(-1, num_features)
    filtered = flat_states[:, states_filter].cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', algorithm='auto')
    nbrs.fit(filtered)
    distances, indices = nbrs.kneighbors(filtered)

    return (
        torch.tensor(distances, dtype=torch.float32, device='cpu'),
        torch.tensor(indices, dtype=torch.int64, device='cpu'),
    )


# ======================================================================
#  Importance weights (non-chunked, single-step)
# ======================================================================
def compute_importance_weights(
    behavioral_policy, target_policy, states, actions,
    real_traj_lengths, env2agent, *, mini_batch_size=16384,
):
    """Compute IS weights. states is [B, T, H, D], actions is [B, T, H, A]."""
    dev = states.device
    B, T, H, state_dim = states.shape
    action_dim = actions.shape[-1]

    flat_states = states.reshape(-1, state_dim)
    flat_actions = actions.reshape(-1, action_dim)
    flat_heads = env2agent.repeat(B * T)

    num_samples = flat_states.shape[0]
    target_lp_chunks = []
    behavior_lp_chunks = []

    for start in range(0, num_samples, mini_batch_size):
        end = min(start + mini_batch_size, num_samples)
        s = flat_states[start:end]
        a = flat_actions[start:end]
        h = flat_heads[start:end]

        target_lp_chunks.append(target_policy.get_log_p_select(s, a, h))
        with torch.no_grad():
            behavior_lp_chunks.append(behavioral_policy.get_log_p_select(s, a, h))

    target_lp = torch.cat(target_lp_chunks, dim=0)
    behavior_lp = torch.cat(behavior_lp_chunks, dim=0)

    log_ratios = (target_lp - behavior_lp).reshape(B, T, H)

    rtl = real_traj_lengths.squeeze(-1) if real_traj_lengths.dim() == 3 else real_traj_lengths
    if rtl.shape[0] != B:
        rtl = rtl.transpose(0, 1)
    rtl = rtl.to(dev)

    time_idx = torch.arange(T, device=dev, dtype=rtl.dtype).view(1, T, 1)
    valid_mask = (time_idx < rtl.unsqueeze(1)).to(log_ratios.dtype)

    cum_log_ratios = torch.cumsum(log_ratios * valid_mask, dim=1)
    iw = torch.exp(cum_log_ratios) * valid_mask
    iw = iw / (iw.sum() + 1e-12)
    return iw.reshape(-1)


# ======================================================================
#  Entropy computation (k-NN based, with IS weights)
# ======================================================================
def compute_entropy(behavioral_policy, target_policy, states, actions,
                    k, distances, indices, states_filter,
                    real_traj_lengths, env2agent):
    """Compute IS-weighted k-NN entropy.

    Args:
        states:  [B, T+1, H_sub, D]
        actions: [B, T, H_sub, A]
        env2agent: [H_sub] — agent IDs for this subset
        distances, indices: from compute_knn on the same subset
    """
    filtered_states = states[:, :-1, :]
    importance_weights = compute_importance_weights(
        behavioral_policy, target_policy, filtered_states, actions,
        real_traj_lengths, env2agent,
    )

    d = len(states_filter)
    eps = 1e-6

    distances = distances.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True)

    k_tensor = torch.tensor(k, dtype=torch.float32)
    B_const = torch.log(k_tensor) - torch.tensor(scipy.special.digamma(k), dtype=torch.float32)
    G = torch.tensor(scipy.special.gamma(d / 2 + 1), dtype=torch.float32)

    weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)
    volumes = (torch.pow(distances[:, k], d) * torch.pow(torch.tensor(torch.pi), d / 2)) / G

    entropy = -torch.sum((weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)) + B_const
    return entropy


# ======================================================================
#  Two-phase train step
# ======================================================================
def train_step_twophase(writer, epoch, states, actions, state_filter,
                        real_traj_lengths, policy, behavior_policy,
                        optimizer, scheduler, k, entropy,
                        num_trunk_envs, num_agents):
    """Two-phase training with split-env data.

    Args:
        states:  [B, T+1, num_total_envs, D]   (full rollout)
        actions: [B, T,   num_total_envs, A]    (full rollout)
        real_traj_lengths: [B, num_total_envs]
        num_trunk_envs: how many envs belong to head 0 (Phase 1)
        num_agents: total number of agents N
    """
    error = False
    num_total_envs = states.shape[2]
    num_head_envs = num_total_envs - num_trunk_envs

    # --- Slice data ---
    trunk_states = states[:, :, :num_trunk_envs, :]
    trunk_actions = actions[:, :, :num_trunk_envs, :]
    trunk_rtl = real_traj_lengths[:, :num_trunk_envs]

    head_states = states[:, :, num_trunk_envs:, :]
    head_actions = actions[:, :, num_trunk_envs:, :]
    head_rtl = real_traj_lengths[:, num_trunk_envs:]

    # env2agent for each subset
    trunk_env2agent = torch.zeros(num_trunk_envs, dtype=torch.long, device=device)  # all head 0

    num_diversity = num_agents - 1
    envs_per_head = num_head_envs // num_diversity
    remainder = num_head_envs % num_diversity
    parts = []
    for h in range(num_diversity):
        count = envs_per_head + (1 if h < remainder else 0)
        parts.append(torch.full((count,), h + 1, dtype=torch.long, device=device))
    head_env2agent = torch.cat(parts)  # heads 1..N-1

    # --- k-NN for each phase (separate flat arrays, separate indices) ---
    print("  Computing k-NN for Phase 1 (trunk)...")
    trunk_distances, trunk_indices = compute_knn(trunk_states, state_filter, k)
    print("  Computing k-NN for Phase 2 (heads)...")
    head_distances, head_indices = compute_knn(head_states, state_filter, k)

    # --- Freeze behavior policy ---
    behavior_policy.eval()
    for p in behavior_policy.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    start_loss = time.time()

    # ==================================================================
    # Phase 1: Trunk + Head 0 (single-agent entropy maximization)
    # ==================================================================
    trunk_entropy_loss = -compute_entropy(
        behavior_policy, policy,
        trunk_states, trunk_actions,
        k=k, distances=trunk_distances, indices=trunk_indices,
        states_filter=state_filter,
        real_traj_lengths=trunk_rtl,
        env2agent=trunk_env2agent,
    )
    trunk_entropy_loss.backward()
    # trunk + head 0 now have gradients

    # --- Diagnostic: check trunk gradient magnitude ---
    trunk_grad_norm = 0.0
    for p in policy.get_trunk_params():
        if p.grad is not None:
            trunk_grad_norm += p.grad.data.norm(2).item() ** 2
    trunk_grad_norm = trunk_grad_norm ** 0.5
    print(f"  Phase 1 trunk grad norm: {trunk_grad_norm:.6f}")

    # ==================================================================
    # Phase 2: Heads 1..N-1 (trunk frozen)
    # ==================================================================
    with policy.detached_trunk():
        head_entropy_loss = -compute_entropy(
            behavior_policy, policy,
            head_states, head_actions,
            k=k, distances=head_distances, indices=head_indices,
            states_filter=state_filter,
            real_traj_lengths=head_rtl,
            env2agent=head_env2agent,
        )
        head_entropy_loss.backward()
    # trunk grads unchanged (from Phase 1). Heads 1..N-1 grads from Phase 2.

    # ==================================================================
    # Single optimizer step
    # ==================================================================
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    print(f"  Time for loss computation: {time.time() - start_loss:.2f}s")

    # --- Safety check ---
    with torch.no_grad():
        sample_states = states[0, 0].contiguous().view(states.shape[2], -1)
        try:
            sample_actions, sample_logp, _ = policy.sample(sample_states)
            if (not torch.isfinite(sample_actions).all()) or (not torch.isfinite(sample_logp).all()):
                print("  Warning: Non-finite values; reverting weights.")
                policy.load_state_dict(behavior_policy.state_dict())
                error = True
            else:
                print("  Policy outputs valid after update.")
        except Exception as e:
            print(f"  Error during safety check: {e}")
            policy.load_state_dict(behavior_policy.state_dict())
            error = True

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # --- Logging ---
    writer.add_scalar('Loss/Phase1_Trunk', trunk_entropy_loss.item(), epoch)
    writer.add_scalar('Loss/Phase2_Heads', head_entropy_loss.item(), epoch)
    total_loss = trunk_entropy_loss.item() + head_entropy_loss.item()
    writer.add_scalar('Loss/Total', total_loss, epoch)
    writer.add_scalar('Entropy', entropy.mean().item(), epoch)
    writer.add_scalar('Learning_Rate/Agent', current_lr, epoch)

    try:
        if epoch % 10 == 0:
            for name, param in policy.named_parameters():
                if param.grad is not None and param.grad.numel() > 0:
                    if param.grad.isfinite().all():
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    except Exception as e:
        print(f"  Error logging gradients: {e}")

    print(f"  Loss: trunk={trunk_entropy_loss.item():.4f}  heads={head_entropy_loss.item():.4f}  Entropy={entropy.mean().item():.4f}")

    del trunk_distances, trunk_indices, head_distances, head_indices
    torch.cuda.empty_cache()
    return error


# ======================================================================
#  Main loop: collect + train + log
# ======================================================================
def reinforce_twophase(writer, epoch, env, policy, behavior_policy,
                       optimizer, scheduler, discretizer,
                       num_trajectories, trajectory_length, state_filter,
                       num_agents, num_total_envs, num_trunk_envs,
                       k, env2agent, log_entropy_interval=40):
    """Single-rollout two-phase collection and training.

    Args:
        num_total_envs: total envs (trunk + head envs).
        num_trunk_envs: how many envs are assigned to head 0.
        env2agent: [num_total_envs] long — built by build_env2agent_twophase.
    """
    # num_features = env.observation_manager.group_obs_dim["policy"][0]
    # num_actions = env.action_manager.action_term_dim[0]
    num_features = env.num_features
    num_actions = env.num_actions

    # --- Collect single rollout across all envs ---
    states, actions, real_traj_lengths = collect_particles(
        env, policy, 1, trajectory_length,
        num_features, num_actions, num_agents, num_total_envs, env2agent,
    )
    states, actions, real_traj_lengths = unpack_results((states, actions, real_traj_lengths))

    # --- Record action distributions (full rollout) ---
    record_actions_step(num_agents, env2agent, actions, states, epoch, writer)

    # --- Entropy estimation (diversity heads only, exclude head 0 trunk envs) ---
    head_states = states[:, :, num_trunk_envs:, :]
    head_rtl = real_traj_lengths[:, num_trunk_envs:]
    start_entropy = time.time()
    with torch.no_grad():
        entropy = knn_entropy_estimation_torch(head_states, state_filter, head_rtl, k=k)
    print(f"  Time for entropy estimation: {time.time() - start_entropy:.2f}s")
    torch.cuda.empty_cache()

    # --- Two-phase training ---
    error = train_step_twophase(
        writer, epoch, states, actions, state_filter,
        real_traj_lengths, policy, behavior_policy,
        optimizer, scheduler, k, entropy,
        num_trunk_envs, num_agents,
    )

    # --- Heatmap logging ---
    if epoch % log_entropy_interval == 0:
        state_filter_heatmap = [0, 1]
        head_vis = states[:, 1:, num_trunk_envs:, state_filter_heatmap]
        num_head_envs = num_total_envs - num_trunk_envs
        vis_states = head_vis.reshape(
            num_head_envs, trajectory_length, len(state_filter_heatmap)
        )
        _, _, image_fig = get_heatmap_fast(vis_states, discretizer)
        writer.add_figure('Heatmap entropy', image_fig, epoch)
        plt.close(image_fig)

    return None, error


# ======================================================================
#  Logging helpers
# ======================================================================
def record_actions_step(num_agents, env2agent, actions, states, epoch, writer):
    for agent_id in range(num_agents):
        env_mask = (env2agent == agent_id).nonzero(as_tuple=True)[0]
        agent_actions = actions[:, :, env_mask, :]
        agent_states = states[:, :, env_mask, :]

        flat_actions = agent_actions.reshape(-1, agent_actions.shape[-1])
        flat_states = agent_states.reshape(-1, agent_states.shape[-1])

        for action_dim in range(flat_actions.shape[-1]):
            writer.add_histogram(
                f'Action_Distributions/Agent_{agent_id}/Action_{action_dim}',
                flat_actions[:, action_dim], epoch,
            )
        writer.add_histogram(
            f'State_Distributions/Agent_{agent_id}/States',
            flat_states[:, :2], epoch,
        )


# ======================================================================
#  PCGrad: Gradient Surgery for Multi-Task Learning
# ======================================================================
def pcgrad(grads):
    """Project conflicting gradients (PCGrad).

    For each pair (i, j), if grad_i · grad_j < 0, project grad_i onto
    the normal plane of grad_j to remove the conflicting component.

    Args:
        grads: list of N 1-D tensors (flattened per-head trunk gradients)

    Returns:
        1-D tensor: surgered gradient (mean of projected gradients)
    """
    num_tasks = len(grads)
    grads_pc = [g.clone() for g in grads]

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                continue
            dot = torch.dot(grads_pc[i], grads[j])
            if dot < 0:
                grads_pc[i] -= (dot / (torch.dot(grads[j], grads[j]) + 1e-12)) * grads[j]

    return torch.stack(grads_pc).mean(dim=0)

import random

def pcgrad_sum(grads):
    """
    Improved PCGrad implementation with random task ordering.
    """
    num_tasks = len(grads)
    # 1. Create a copy to store the "surgered" gradients
    grads_pc = [g.clone() for g in grads]
    
    # 2. Randomize task order to avoid directional bias
    indices = list(range(num_tasks))
    random.shuffle(indices)

    for i in indices:
        # Create a shuffled list of other tasks to project against
        other_indices = [j for j in range(num_tasks) if j != i]
        random.shuffle(other_indices)
        
        for j in other_indices:
            # Calculate cosine similarity (dot product)
            dot = torch.dot(grads_pc[i], grads[j])
            if dot < 0:
                # Project g_i onto the normal plane of g_j
                # Adding a small epsilon to avoid division by zero
                mag_sq = torch.dot(grads[j], grads[j]) + 1e-12
                grads_pc[i] -= (dot / mag_sq) * grads[j]

    # 3. Return the sum (standard for PCGrad) or mean
    return torch.stack(grads_pc).sum(dim=0)




# ======================================================================
#  Train step with gradient surgery
# ======================================================================
def train_step_gradsurgery(writer, epoch, states, actions, state_filter,
                           real_traj_lengths, policy, behavior_policy,
                           optimizer, scheduler, k, entropy,
                           env2agent, num_agents):
    """Train with PCGrad on trunk gradients.

    For each head:
      1. Slice that head's env data
      2. Compute per-head k-NN + entropy loss
      3. Backward → save trunk grad, zero it (head grads accumulate)
    Then apply PCGrad to the N trunk gradients and step once.

    No head-0 privilege. No detached_trunk. All heads contribute to trunk
    update with conflicts resolved.

    Args:
        states:  [B, T+1, num_envs, D]
        actions: [B, T,   num_envs, A]
        real_traj_lengths: [B, num_envs]
        env2agent: [num_envs] long
        num_agents: N
    """
    error = False

    # --- Freeze behavior policy ---
    behavior_policy.eval()
    for p in behavior_policy.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    start_loss = time.time()

    trunk_params = list(policy.get_trunk_params())

    per_head_trunk_grads = []
    per_head_losses = []

    # ==================================================================
    # N backward passes: one per head
    # ==================================================================
    for head_id in range(num_agents):
        # Slice data for this head's envs
        head_env_indices = (env2agent == head_id).nonzero(as_tuple=True)[0]
        num_head_envs = head_env_indices.shape[0]

        h_states = states[:, :, head_env_indices, :]
        h_actions = actions[:, :, head_env_indices, :]
        h_rtl = real_traj_lengths[:, head_env_indices]
        h_env2agent = torch.full((num_head_envs,), head_id,
                                 dtype=torch.long, device=device)

        # Per-head k-NN (on this head's states only)
        h_distances, h_indices = compute_knn(h_states, state_filter, k)

        # Per-head entropy loss
        h_loss = -compute_entropy(
            behavior_policy, policy,
            h_states, h_actions,
            k=k, distances=h_distances, indices=h_indices,
            states_filter=state_filter,
            real_traj_lengths=h_rtl,
            env2agent=h_env2agent,
        )
        per_head_losses.append(h_loss.item())

        # Backward — head param grads accumulate, trunk grads saved separately
        h_loss.backward()

        # Save trunk grad and zero it
        head_trunk_grad = []
        for p in trunk_params:
            if p.grad is not None:
                head_trunk_grad.append(p.grad.data.clone().flatten())
                p.grad.data.zero_()
            else:
                head_trunk_grad.append(torch.zeros(p.numel(), device=device))
        per_head_trunk_grads.append(torch.cat(head_trunk_grad))

        del h_distances, h_indices

    # ==================================================================
    # PCGrad on trunk gradients
    # ==================================================================
    surgered_grad = pcgrad_sum(per_head_trunk_grads)

    # Write surgered gradient back into trunk params
    offset = 0
    for p in trunk_params:
        numel = p.numel()
        if p.grad is None:
            p.grad = surgered_grad[offset:offset + numel].view(p.shape)
        else:
            p.grad.data.copy_(surgered_grad[offset:offset + numel].view(p.shape))
        offset += numel

    # --- Diagnostics ---
    raw_norms = [g.norm().item() for g in per_head_trunk_grads]
    surgered_norm = surgered_grad.norm().item()

    # Count pairwise conflicts
    num_conflicts = 0
    num_pairs = 0
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            dot = torch.dot(per_head_trunk_grads[i], per_head_trunk_grads[j])
            num_pairs += 1
            if dot < 0:
                num_conflicts += 1

    print(f"  PCGrad: {num_conflicts}/{num_pairs} conflicts | "
          f"raw norms: [{min(raw_norms):.4f}, {max(raw_norms):.4f}] | "
          f"surgered norm: {surgered_norm:.4f}")

    # ==================================================================
    # Single optimizer step
    # ==================================================================
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    print(f"  Time for loss computation: {time.time() - start_loss:.2f}s")

    # --- Safety check ---
    with torch.no_grad():
        sample_states = states[0, 0].contiguous().view(states.shape[2], -1)
        try:
            sample_actions, sample_logp, _ = policy.sample(sample_states)
            if (not torch.isfinite(sample_actions).all()) or (not torch.isfinite(sample_logp).all()):
                print("  Warning: Non-finite values; reverting weights.")
                policy.load_state_dict(behavior_policy.state_dict())
                error = True
            else:
                print("  Policy outputs valid after update.")
        except Exception as e:
            print(f"  Error during safety check: {e}")
            policy.load_state_dict(behavior_policy.state_dict())
            error = True

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # --- Logging ---
    mean_head_loss = sum(per_head_losses) / len(per_head_losses)
    writer.add_scalar('Loss/Mean_PerHead', mean_head_loss, epoch)
    for i, l in enumerate(per_head_losses):
        writer.add_scalar(f'Loss/Head_{i}', l, epoch)
    writer.add_scalar('Loss/Total', sum(per_head_losses), epoch)
    writer.add_scalar('Entropy', entropy.mean().item(), epoch)
    writer.add_scalar('Learning_Rate/Trunk', current_lr, epoch)
    writer.add_scalar('GradSurgery/Conflicts', num_conflicts, epoch)
    writer.add_scalar('GradSurgery/SurgeredNorm', surgered_norm, epoch)

    try:
        if epoch % 10 == 0:
            trunk_param_ids = {id(p) for p in trunk_params}
            for name, param in policy.named_parameters():
                if id(param) in trunk_param_ids and param.grad is not None and param.grad.numel() > 0:
                    if param.grad.isfinite().all():
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    except Exception as e:
        print(f"  Error logging gradients: {e}")

    print(f"  Per-head losses: {[f'{l:.4f}' for l in per_head_losses]}  Entropy={entropy.mean().item():.4f}")

    torch.cuda.empty_cache()
    return error


# ======================================================================
#  Main loop: gradient surgery
# ======================================================================
def reinforce_gradsurgery(writer, epoch, env, policy, behavior_policy,
                          optimizer, scheduler, discretizer,
                          num_trajectories, trajectory_length, state_filter,
                          num_agents, num_envs, k, env2agent,
                          log_entropy_interval=40):
    """Single-rollout collection + gradient surgery training.

    Uses standard env layout (num_envs split evenly across all N agents).
    """
    # num_features = env.observation_manager.group_obs_dim["policy"][0]
    # num_actions = env.action_manager.action_term_dim[0]
    num_features = env.num_features
    num_actions = env.num_actions

    # --- Collect single rollout ---
    states, actions, real_traj_lengths = collect_particles(
        env, policy, 1, trajectory_length,
        num_features, num_actions, num_agents, num_envs, env2agent,
    )
    states, actions, real_traj_lengths = unpack_results((states, actions, real_traj_lengths))

    # --- Record action distributions ---
    record_actions_step(num_agents, env2agent, actions, states, epoch, writer)

    # --- Entropy estimation (all states pooled) ---
    start_entropy = time.time()
    with torch.no_grad():
        entropy = knn_entropy_estimation_torch(states, state_filter, real_traj_lengths, k=k)
    print(f"  Time for entropy estimation: {time.time() - start_entropy:.2f}s")
    torch.cuda.empty_cache()

    # --- Gradient surgery training ---
    error = train_step_gradsurgery(
        writer, epoch, states, actions, state_filter,
        real_traj_lengths, policy, behavior_policy,
        optimizer, scheduler, k, entropy,
        env2agent, num_agents,
    )

    # --- Heatmap logging ---
    if epoch % log_entropy_interval == 0:
        state_filter_heatmap = [0, 1]
        vis_states = states[:, 1:, :, state_filter_heatmap].reshape(
            num_envs, trajectory_length, len(state_filter_heatmap)
        )
        _, _, image_fig = get_heatmap_fast(vis_states, discretizer)
        writer.add_figure('Heatmap entropy', image_fig, epoch)
        plt.close(image_fig)

    return None, error


# ======================================================================
#  Train step with gradient surgery on COLLECTIVE entropy
# ======================================================================
def train_step_gradsurgery_collective(writer, epoch, states, actions, state_filter,
                                      real_traj_lengths, policy, behavior_policy,
                                      optimizer, scheduler, k, entropy,
                                      env2agent, num_agents):
    """Gradient surgery with COLLECTIVE k-NN entropy.

    Unlike train_step_gradsurgery (per-head k-NN), this uses a single joint
    k-NN over ALL agents' states, preserving the diversity pressure.

    For each head h:
      1. Compute target_lp WITH gradient only for head h's samples
      2. Build partial IS weights (head h has gradient, others detached)
      3. Compute collective entropy from partial weights + joint k-NN
      4. backward() -> save trunk grad, zero it; head grads accumulate
    Then PCGrad on the N trunk gradient components.

    Args:
        states:  [B, T+1, num_envs, D]
        actions: [B, T, num_envs, A]
        real_traj_lengths: [B, num_envs]
        env2agent: [num_envs] long
        num_agents: N
    """
    error = False
    mini_batch = 16384

    # --- 1. Joint k-NN over ALL states ---
    print("  Computing joint k-NN...")
    distances, indices = compute_knn(states, state_filter, k)

    # --- 2. Freeze behavior policy ---
    behavior_policy.eval()
    for p in behavior_policy.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    start_loss = time.time()

    trunk_params = list(policy.get_trunk_params())

    # --- 3. Flatten data ---
    filtered_states = states[:, :-1, :]  # [B, T, H, D]
    B_batch, T, H_envs, state_dim = filtered_states.shape
    action_dim = actions.shape[-1]

    flat_states = filtered_states.reshape(-1, state_dim)
    flat_actions = actions.reshape(-1, action_dim)
    flat_heads = env2agent.repeat(B_batch * T)
    num_samples = flat_states.shape[0]

    # --- 4. Pre-compute behavior_lp (constant, no grad) ---
    behavior_lp_chunks = []
    with torch.no_grad():
        for start in range(0, num_samples, mini_batch):
            end = min(start + mini_batch, num_samples)
            behavior_lp_chunks.append(
                behavior_policy.get_log_p_select(
                    flat_states[start:end], flat_actions[start:end], flat_heads[start:end]
                )
            )
    full_behavior_lp = torch.cat(behavior_lp_chunks)

    # --- 5. Pre-compute detached target_lp (reused as constant for non-grad heads) ---
    with torch.no_grad():
        target_lp_det_chunks = []
        for start in range(0, num_samples, mini_batch):
            end = min(start + mini_batch, num_samples)
            target_lp_det_chunks.append(
                policy.get_log_p_select(
                    flat_states[start:end], flat_actions[start:end], flat_heads[start:end]
                )
            )
    full_target_lp_detached = torch.cat(target_lp_det_chunks)

    # --- 6. k-NN constants (reused across heads) ---
    d = len(state_filter)
    eps = 1e-6
    dist_dev = distances.to(device, non_blocking=True)
    idx_dev = indices.to(device, non_blocking=True)
    k_tensor = torch.tensor(k, dtype=torch.float32)
    B_const = torch.log(k_tensor) - torch.tensor(scipy.special.digamma(k), dtype=torch.float32)
    G = torch.tensor(scipy.special.gamma(d / 2 + 1), dtype=torch.float32)
    vol_coeff = torch.pow(torch.tensor(torch.pi), d / 2) / G

    # --- RTL mask (constant) ---
    rtl = real_traj_lengths
    if rtl.dim() == 3:
        rtl = rtl.squeeze(-1)
    if rtl.shape[0] != B_batch:
        rtl = rtl.transpose(0, 1)
    rtl = rtl.to(device)
    time_idx = torch.arange(T, device=device, dtype=rtl.dtype).view(1, T, 1)
    valid_mask_full = (time_idx < rtl.unsqueeze(1)).float()

    # --- 7. Per-head backward passes ---
    per_head_trunk_grads = []
    per_head_losses = []

    for head_id in range(num_agents):
        mask_h = (flat_heads == head_id)
        h_indices = mask_h.nonzero(as_tuple=True)[0]

        # Forward WITH gradient only for head h's samples (~N_samples/num_agents)
        h_lp = policy.get_log_p_select(
            flat_states[h_indices], flat_actions[h_indices], flat_heads[h_indices]
        )

        # Build partial_lp: gradient for head h, detached for the rest
        h_lp_scattered = torch.zeros(num_samples, device=device, dtype=h_lp.dtype)
        h_lp_scattered = h_lp_scattered.scatter(0, h_indices, h_lp)
        mask_float = mask_h.float()
        partial_target_lp = h_lp_scattered * mask_float + full_target_lp_detached * (1 - mask_float)

        # IS weights from partial_lp
        log_ratios = (partial_target_lp - full_behavior_lp).reshape(B_batch, T, H_envs)
        cum_log_ratios = torch.cumsum(log_ratios * valid_mask_full, dim=1)
        iw = torch.exp(cum_log_ratios) * valid_mask_full
        importance_weights = (iw / (iw.sum() + 1e-12)).reshape(-1)

        # Collective entropy with joint k-NN
        weights_sum = torch.sum(importance_weights[idx_dev[:, :-1]], dim=1)
        volumes = torch.pow(dist_dev[:, k], d) * vol_coeff
        h_entropy = -torch.sum(
            (weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)
        ) + B_const

        h_loss = -h_entropy
        per_head_losses.append(h_loss.item())

        # Backward: gradient flows only through head h's target_lp
        h_loss.backward()

        # Save trunk grad and zero it
        head_trunk_grad = []
        for p in trunk_params:
            if p.grad is not None:
                head_trunk_grad.append(p.grad.data.clone().flatten())
                p.grad.data.zero_()
            else:
                head_trunk_grad.append(torch.zeros(p.numel(), device=device))
        per_head_trunk_grads.append(torch.cat(head_trunk_grad))

    # --- 8. PCGrad on trunk gradients ---
    surgered_grad = pcgrad(per_head_trunk_grads)

    # Write surgered gradient back into trunk params
    offset = 0
    for p in trunk_params:
        numel = p.numel()
        if p.grad is None:
            p.grad = surgered_grad[offset:offset + numel].view(p.shape)
        else:
            p.grad.data.copy_(surgered_grad[offset:offset + numel].view(p.shape))
        offset += numel

    # --- Diagnostics ---
    raw_norms = [g.norm().item() for g in per_head_trunk_grads]
    surgered_norm = surgered_grad.norm().item()

    num_conflicts = 0
    num_pairs = 0
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if torch.dot(per_head_trunk_grads[i], per_head_trunk_grads[j]) < 0:
                num_conflicts += 1
            num_pairs += 1

    print(f"  PCGrad: {num_conflicts}/{num_pairs} conflicts | "
          f"raw norms: [{min(raw_norms):.4f}, {max(raw_norms):.4f}] | "
          f"surgered norm: {surgered_norm:.4f}")

    # --- 9. Optimizer step ---
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    print(f"  Time for loss computation: {time.time() - start_loss:.2f}s")

    # --- Safety check ---
    with torch.no_grad():
        sample_states = states[0, 0].contiguous().view(states.shape[2], -1)
        try:
            sample_actions, sample_logp, _ = policy.sample(sample_states)
            if (not torch.isfinite(sample_actions).all()) or (not torch.isfinite(sample_logp).all()):
                print("  Warning: Non-finite values; reverting weights.")
                policy.load_state_dict(behavior_policy.state_dict())
                error = True
            else:
                print("  Policy outputs valid after update.")
        except Exception as e:
            print(f"  Error during safety check: {e}")
            policy.load_state_dict(behavior_policy.state_dict())
            error = True

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # --- Logging ---
    mean_head_loss = sum(per_head_losses) / len(per_head_losses)
    writer.add_scalar('Loss/Mean_PerHead', mean_head_loss, epoch)
    for i, hl in enumerate(per_head_losses):
        writer.add_scalar(f'Loss/Head_{i}', hl, epoch)
    writer.add_scalar('Loss/Total', sum(per_head_losses), epoch)
    writer.add_scalar('Entropy', entropy.mean().item(), epoch)
    writer.add_scalar('Learning_Rate/Trunk', current_lr, epoch)
    writer.add_scalar('GradSurgery/Conflicts', num_conflicts, epoch)
    writer.add_scalar('GradSurgery/SurgeredNorm', surgered_norm, epoch)

    try:
        if epoch % 10 == 0:
            trunk_param_ids = {id(p) for p in trunk_params}
            for name, param in policy.named_parameters():
                if id(param) in trunk_param_ids and param.grad is not None and param.grad.numel() > 0:
                    if param.grad.isfinite().all():
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    except Exception as e:
        print(f"  Error logging gradients: {e}")

    print(f"  Per-head losses: {[f'{l:.4f}' for l in per_head_losses]}  Entropy={entropy.mean().item():.4f}")

    del distances, indices, dist_dev, idx_dev, full_behavior_lp, full_target_lp_detached
    torch.cuda.empty_cache()
    return error


# ======================================================================
#  Main loop: gradient surgery with collective entropy
# ======================================================================
def reinforce_gradsurgery_collective(writer, epoch, env, policy, behavior_policy,
                                     optimizer, scheduler, discretizer,
                                     num_trajectories, trajectory_length, state_filter,
                                     num_agents, num_envs, k, env2agent,
                                     log_entropy_interval=40):
    """Single-rollout collection + gradient surgery with collective k-NN entropy."""
    # num_features = env.observation_manager.group_obs_dim["policy"][0]
    # num_actions = env.action_manager.action_term_dim[0]
    num_features = env.num_features
    num_actions = env.num_actions

    # --- Collect single rollout ---
    states, actions, real_traj_lengths = collect_particles(
        env, policy, 1, trajectory_length,
        num_features, num_actions, num_agents, num_envs, env2agent,
    )
    states, actions, real_traj_lengths = unpack_results((states, actions, real_traj_lengths))

    # --- Record action distributions ---
    record_actions_step(num_agents, env2agent, actions, states, epoch, writer)

    # --- Entropy estimation (all states pooled) ---
    start_entropy = time.time()
    with torch.no_grad():
        entropy = knn_entropy_estimation_torch(states, state_filter, real_traj_lengths, k=k)
    print(f"  Time for entropy estimation: {time.time() - start_entropy:.2f}s")
    torch.cuda.empty_cache()

    # --- Gradient surgery training (collective) ---
    error = train_step_gradsurgery_collective(
        writer, epoch, states, actions, state_filter,
        real_traj_lengths, policy, behavior_policy,
        optimizer, scheduler, k, entropy,
        env2agent, num_agents,
    )

    # --- Heatmap logging ---
    if epoch % log_entropy_interval == 0:
        state_filter_heatmap = [0, 1]
        vis_states = states[:, 1:, :, state_filter_heatmap].reshape(
            num_envs, trajectory_length, len(state_filter_heatmap)
        )
        _, _, image_fig = get_heatmap_fast(vis_states, discretizer)
        writer.add_figure('Heatmap entropy', image_fig, epoch)
        plt.close(image_fig)

    return None, error


# ======================================================================
#  Train step for Siamese (parallel independent nets) + collective k-NN
# ======================================================================
def train_step_siamese_collective(writer, epoch, states, actions, state_filter,
                                  real_traj_lengths, policy, behavior_policy,
                                  optimizer, scheduler, k, entropy,
                                  env2agent, num_agents):
    """Train N fully independent networks with COLLECTIVE k-NN entropy.

    Since PolicySiameseNetwork has no shared trunk, we do NOT need
    gradient surgery.  Each agent's loss is computed against the joint
    k-NN graph so that agents are repelled from each other's visited states.
    Gradients from agent h's loss naturally flow only into agent h's
    independent network.

    Approach:
        1. Joint k-NN over ALL agents' states (diversity pressure).
        2. Pre-compute detached target_lp & behavior_lp for all samples.
        3. For each agent h:
             - Re-compute target_lp WITH gradient for agent h's samples only.
             - Build partial_target_lp (h with grad, others detached).
             - Compute IS weights → collective entropy → accumulate loss.
        4. Single backward() + optimizer.step().  Each agent's subnet
           receives gradient only through its own samples.

    Args:
        states:  [B, T+1, num_envs, D]
        actions: [B, T, num_envs, A]
        real_traj_lengths: [B, num_envs]
        env2agent: [num_envs] long
        num_agents: N
    """
    error = False
    mini_batch = 16384

    # --- 1. Joint k-NN over ALL states ---
    print("  [Siamese] Computing joint k-NN...")
    distances, indices = compute_knn(states, state_filter, k)

    # --- 2. Freeze behavior policy ---
    behavior_policy.eval()
    for p in behavior_policy.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    start_loss = time.time()

    # --- 3. Flatten data ---
    filtered_states = states[:, :-1, :]  # [B, T, H, D]
    B_batch, T, H_envs, state_dim = filtered_states.shape
    action_dim = actions.shape[-1]

    flat_states = filtered_states.reshape(-1, state_dim)
    flat_actions = actions.reshape(-1, action_dim)
    flat_heads = env2agent.repeat(B_batch * T)
    num_samples = flat_states.shape[0]

    # --- 4. Compute behavior_lp (constant, no grad) ---
    behavior_lp_chunks = []
    with torch.no_grad():
        for start in range(0, num_samples, mini_batch):
            end = min(start + mini_batch, num_samples)
            behavior_lp_chunks.append(
                behavior_policy.get_log_p_select(
                    flat_states[start:end], flat_actions[start:end], flat_heads[start:end]
                )
            )
    full_behavior_lp = torch.cat(behavior_lp_chunks)

    # --- 5. Compute target_lp WITH gradient (single pass, all agents) ---
    target_lp_chunks = []
    for start in range(0, num_samples, mini_batch):
        end = min(start + mini_batch, num_samples)
        target_lp_chunks.append(
            policy.get_log_p_select(
                flat_states[start:end], flat_actions[start:end], flat_heads[start:end]
            )
        )
    full_target_lp = torch.cat(target_lp_chunks)

    # --- 6. k-NN constants ---
    d = len(state_filter)
    eps = 1e-6
    dist_dev = distances.to(device, non_blocking=True)
    idx_dev = indices.to(device, non_blocking=True)
    k_tensor = torch.tensor(k, dtype=torch.float32)
    B_const = torch.log(k_tensor) - torch.tensor(scipy.special.digamma(k), dtype=torch.float32)
    G = torch.tensor(scipy.special.gamma(d / 2 + 1), dtype=torch.float32)
    vol_coeff = torch.pow(torch.tensor(torch.pi), d / 2) / G

    # --- RTL mask ---
    rtl = real_traj_lengths
    if rtl.dim() == 3:
        rtl = rtl.squeeze(-1)
    if rtl.shape[0] != B_batch:
        rtl = rtl.transpose(0, 1)
    rtl = rtl.to(device)
    time_idx = torch.arange(T, device=device, dtype=rtl.dtype).view(1, T, 1)
    valid_mask_full = (time_idx < rtl.unsqueeze(1)).float()

    # --- 7. IS weights + collective entropy (single pass, no per-agent loop) ---
    log_ratios = (full_target_lp - full_behavior_lp).reshape(B_batch, T, H_envs)
    cum_log_ratios = torch.cumsum(log_ratios * valid_mask_full, dim=1)
    iw = torch.exp(cum_log_ratios) * valid_mask_full
    importance_weights = (iw / (iw.sum() + 1e-12)).reshape(-1)

    weights_sum = torch.sum(importance_weights[idx_dev[:, :-1]], dim=1)
    volumes = torch.pow(dist_dev[:, k], d) * vol_coeff
    collective_entropy = -torch.sum(
        (weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)
    ) + B_const

    loss = -collective_entropy

    # --- 8. Single backward + optimizer step ---
    loss.backward()

    # --- Diagnostics: per-agent gradient norms ---
    agent_grad_norms = []
    for agent_id in range(num_agents):
        norm_sq = 0.0
        for p in policy.trunks[agent_id].parameters():
            if p.grad is not None:
                norm_sq += p.grad.data.norm(2).item() ** 2
        for p in policy.head_adapters[agent_id].parameters():
            if p.grad is not None:
                norm_sq += p.grad.data.norm(2).item() ** 2
        for p in policy.mean_heads[agent_id].parameters():
            if p.grad is not None:
                norm_sq += p.grad.data.norm(2).item() ** 2
        for p in policy.log_std_heads[agent_id].parameters():
            if p.grad is not None:
                norm_sq += p.grad.data.norm(2).item() ** 2
        agent_grad_norms.append(norm_sq ** 0.5)

    print(f"  [Siamese] Agent grad norms: [{min(agent_grad_norms):.4f}, {max(agent_grad_norms):.4f}]")

    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    print(f"  [Siamese] Time for loss computation: {time.time() - start_loss:.2f}s")

    # --- Safety check ---
    with torch.no_grad():
        sample_states = states[0, 0].contiguous().view(states.shape[2], -1)
        try:
            sample_actions, sample_logp, _ = policy.sample(sample_states)
            if (not torch.isfinite(sample_actions).all()) or (not torch.isfinite(sample_logp).all()):
                print("  [Siamese] Warning: Non-finite values; reverting weights.")
                policy.load_state_dict(behavior_policy.state_dict())
                error = True
            else:
                print("  [Siamese] Policy outputs valid after update.")
        except Exception as e:
            print(f"  [Siamese] Error during safety check: {e}")
            policy.load_state_dict(behavior_policy.state_dict())
            error = True

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # --- Logging ---
    writer.add_scalar('Loss/Collective', loss.item(), epoch)
    writer.add_scalar('Entropy', entropy.mean().item(), epoch)
    writer.add_scalar('Learning_Rate', current_lr, epoch)
    for i, gn in enumerate(agent_grad_norms):
        writer.add_scalar(f'GradNorm/Agent_{i}', gn, epoch)

    try:
        if epoch % 10 == 0:
            for name, param in policy.named_parameters():
                if param.grad is not None and param.grad.numel() > 0:
                    if param.grad.isfinite().all():
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    except Exception as e:
        print(f"  [Siamese] Error logging gradients: {e}")

    print(f"  [Siamese] Loss={loss.item():.4f}  Entropy={entropy.mean().item():.4f}")

    del distances, indices, dist_dev, idx_dev, full_behavior_lp
    torch.cuda.empty_cache()
    return error


# ======================================================================
#  Main loop: Siamese + collective entropy
# ======================================================================
def reinforce_siamese_collective(writer, epoch, env, policy, behavior_policy,
                                 optimizer, scheduler, discretizer,
                                 num_trajectories, trajectory_length, state_filter,
                                 num_agents, num_envs, k, env2agent,
                                 log_entropy_interval=40,chunk_length=1):
    """Single-rollout collection + Siamese training with collective k-NN entropy."""
    # num_features = env.observation_manager.group_obs_dim["policy"][0]
    # num_actions = env.action_manager.action_term_dim[0]
    num_features = env.num_features
    num_actions = env.num_actions

    # --- Collect single rollout ---
    states, actions, real_traj_lengths = collect_particles(
        env, policy, 1, trajectory_length,
        num_features, num_actions, num_agents, num_envs, env2agent,
    )
    states, actions, real_traj_lengths = unpack_results((states, actions, real_traj_lengths))

    # --- Record action distributions ---
    record_actions_step(num_agents, env2agent, actions, states, epoch, writer)

    # --- Entropy estimation (all states pooled) ---
    start_entropy = time.time()
    with torch.no_grad():
        entropy = knn_entropy_estimation_torch(states, state_filter, real_traj_lengths, k=k)
    print(f"  Time for entropy estimation: {time.time() - start_entropy:.2f}s")
    torch.cuda.empty_cache()

    # --- Siamese training with collective k-NN ---
    error = train_step_siamese_collective(
        writer, epoch, states, actions, state_filter,
        real_traj_lengths, policy, behavior_policy,
        optimizer, scheduler, k, entropy,
        env2agent, num_agents,
    )

    # --- Heatmap logging ---
    if epoch % log_entropy_interval == 0:
        state_filter_heatmap = [0, 1]
        vis_states = states[:, 1:, :, state_filter_heatmap].reshape(
            num_envs, trajectory_length, len(state_filter_heatmap)
        )
        _, _, image_fig = get_heatmap_fast(vis_states, discretizer)
        writer.add_figure('Heatmap entropy', image_fig, epoch)
        plt.close(image_fig)

    return None, error


# ======================================================================
#  Recurrent policy: importance weights (discards hidden returns)
# ======================================================================
def compute_importance_weights_recurrent(
    behavioral_policy, target_policy, states, actions,
    real_traj_lengths, env2agent, *, mini_batch_size=16384,
):
    """Compute IS weights for recurrent policies.

    Same as compute_importance_weights but unpacks the (log_prob, hidden)
    tuple returned by recurrent get_log_p_select.
    """
    dev = states.device
    B, T, H, state_dim = states.shape
    action_dim = actions.shape[-1]

    flat_states = states.reshape(-1, state_dim)
    flat_actions = actions.reshape(-1, action_dim)
    flat_heads = env2agent.repeat(B * T)

    num_samples = flat_states.shape[0]
    target_lp_chunks = []
    behavior_lp_chunks = []

    for start in range(0, num_samples, mini_batch_size):
        end = min(start + mini_batch_size, num_samples)
        s = flat_states[start:end]
        a = flat_actions[start:end]
        h = flat_heads[start:end]

        lp_target, _ = target_policy.get_log_p_select(s, a, h)
        target_lp_chunks.append(lp_target)
        with torch.no_grad():
            lp_behav, _ = behavioral_policy.get_log_p_select(s, a, h)
            behavior_lp_chunks.append(lp_behav)

    target_lp = torch.cat(target_lp_chunks, dim=0)
    behavior_lp = torch.cat(behavior_lp_chunks, dim=0)

    log_ratios = (target_lp - behavior_lp).reshape(B, T, H)

    rtl = real_traj_lengths.squeeze(-1) if real_traj_lengths.dim() == 3 else real_traj_lengths
    if rtl.shape[0] != B:
        rtl = rtl.transpose(0, 1)
    rtl = rtl.to(dev)

    time_idx = torch.arange(T, device=dev, dtype=rtl.dtype).view(1, T, 1)
    valid_mask = (time_idx < rtl.unsqueeze(1)).to(log_ratios.dtype)

    cum_log_ratios = torch.cumsum(log_ratios * valid_mask, dim=1)
    iw = torch.exp(cum_log_ratios) * valid_mask
    iw = iw / (iw.sum() + 1e-12)
    return iw.reshape(-1)


# ======================================================================
#  Recurrent policy: entropy (k-NN based, with IS weights)
# ======================================================================
def compute_entropy_recurrent(behavioral_policy, target_policy, states, actions,
                              k, distances, indices, states_filter,
                              real_traj_lengths, env2agent):
    """Compute IS-weighted k-NN entropy for recurrent policies."""
    filtered_states = states[:, :-1, :]
    importance_weights = compute_importance_weights_recurrent(
        behavioral_policy, target_policy, filtered_states, actions,
        real_traj_lengths, env2agent,
    )

    d = len(states_filter)
    eps = 1e-6

    distances = distances.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True)

    k_tensor = torch.tensor(k, dtype=torch.float32)
    B_const = torch.log(k_tensor) - torch.tensor(scipy.special.digamma(k), dtype=torch.float32)
    G = torch.tensor(scipy.special.gamma(d / 2 + 1), dtype=torch.float32)

    weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)
    volumes = (torch.pow(distances[:, k], d) * torch.pow(torch.tensor(torch.pi), d / 2)) / G

    entropy = -torch.sum((weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)) + B_const
    return entropy


# ======================================================================
#  Recurrent: train step with gradient surgery on COLLECTIVE entropy
# ======================================================================
def train_step_recurrent_gradsurgery_collective(writer, epoch, states, actions, state_filter,
                                                real_traj_lengths, policy, behavior_policy,
                                                optimizer, scheduler, k, entropy,
                                                env2agent, num_agents):
    """Train recurrent policy with COLLECTIVE k-NN entropy (no PCGrad).

    Single forward pass over all agents, single backward, single optimizer step.
    Same structure as train_step_siamese_collective but with recurrent
    (log_prob, hidden) tuple unpacking.

    Args:
        states:  [B, T+1, num_envs, D]
        actions: [B, T, num_envs, A]
        real_traj_lengths: [B, num_envs]
        env2agent: [num_envs] long
        num_agents: N
    """
    error = False
    mini_batch = 16384

    # --- 1. Joint k-NN over ALL states ---
    print("  [Recurrent] Computing joint k-NN...")
    distances, indices = compute_knn(states, state_filter, k)

    # --- 2. Freeze behavior policy ---
    behavior_policy.eval()
    for p in behavior_policy.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    start_loss = time.time()

    # --- 3. Flatten data ---
    filtered_states = states[:, :-1, :]  # [B, T, H, D]
    B_batch, T, H_envs, state_dim = filtered_states.shape
    action_dim = actions.shape[-1]

    flat_states = filtered_states.reshape(-1, state_dim)
    flat_actions = actions.reshape(-1, action_dim)
    flat_heads = env2agent.repeat(B_batch * T)
    num_samples = flat_states.shape[0]

    # --- 4. Compute behavior_lp (constant, no grad) ---
    behavior_lp_chunks = []
    with torch.no_grad():
        for start in range(0, num_samples, mini_batch):
            end = min(start + mini_batch, num_samples)
            lp, _ = behavior_policy.get_log_p_select(
                flat_states[start:end], flat_actions[start:end], flat_heads[start:end]
            )
            behavior_lp_chunks.append(lp)
    full_behavior_lp = torch.cat(behavior_lp_chunks)

    # --- 5. Compute target_lp WITH gradient (single pass, all agents) ---
    target_lp_chunks = []
    for start in range(0, num_samples, mini_batch):
        end = min(start + mini_batch, num_samples)
        lp, _ = policy.get_log_p_select(
            flat_states[start:end], flat_actions[start:end], flat_heads[start:end]
        )
        target_lp_chunks.append(lp)
    full_target_lp = torch.cat(target_lp_chunks)

    # --- 6. k-NN constants ---
    d = len(state_filter)
    eps = 1e-6
    dist_dev = distances.to(device, non_blocking=True)
    idx_dev = indices.to(device, non_blocking=True)
    k_tensor = torch.tensor(k, dtype=torch.float32)
    B_const = torch.log(k_tensor) - torch.tensor(scipy.special.digamma(k), dtype=torch.float32)
    G = torch.tensor(scipy.special.gamma(d / 2 + 1), dtype=torch.float32)
    vol_coeff = torch.pow(torch.tensor(torch.pi), d / 2) / G

    # --- RTL mask ---
    rtl = real_traj_lengths
    if rtl.dim() == 3:
        rtl = rtl.squeeze(-1)
    if rtl.shape[0] != B_batch:
        rtl = rtl.transpose(0, 1)
    rtl = rtl.to(device)
    time_idx = torch.arange(T, device=device, dtype=rtl.dtype).view(1, T, 1)
    valid_mask_full = (time_idx < rtl.unsqueeze(1)).float()

    # --- 7. IS weights + collective entropy (single pass, no per-agent loop) ---
    log_ratios = (full_target_lp - full_behavior_lp).reshape(B_batch, T, H_envs)
    cum_log_ratios = torch.cumsum(log_ratios * valid_mask_full, dim=1)
    iw = torch.exp(cum_log_ratios) * valid_mask_full
    importance_weights = (iw / (iw.sum() + 1e-12)).reshape(-1)

    weights_sum = torch.sum(importance_weights[idx_dev[:, :-1]], dim=1)
    volumes = torch.pow(dist_dev[:, k], d) * vol_coeff
    collective_entropy = -torch.sum(
        (weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)
    ) + B_const

    loss = -collective_entropy

    # --- 8. Single backward + optimizer step ---
    loss.backward()

    # --- Diagnostics: trunk + head gradient norms ---
    trunk_grad_norm = 0.0
    for p in policy.get_trunk_params():
        if p.grad is not None:
            trunk_grad_norm += p.grad.data.norm(2).item() ** 2
    trunk_grad_norm = trunk_grad_norm ** 0.5

    head_grad_norm = 0.0
    for p in policy.get_head_params():
        if hasattr(p, 'grad') and p.grad is not None:
            head_grad_norm += p.grad.data.norm(2).item() ** 2
    head_grad_norm = head_grad_norm ** 0.5

    print(f"  [Recurrent] Trunk grad norm: {trunk_grad_norm:.4f} | Head grad norm: {head_grad_norm:.4f}")

    # GRU sigmoid gates never kill gradients (unlike ReLU), and 3 gates
    # amplify gradient ~3x per parameter → raw norms are ~1000x larger
    # than a feedforward trunk. Use a proportionally larger clip so the
    # effective step size matches the feedforward baseline.
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    print(f"  [Recurrent] Time for loss computation: {time.time() - start_loss:.2f}s")

    # --- Safety check (unpack 4 returns from recurrent sample) ---
    with torch.no_grad():
        sample_states = states[0, 0].contiguous().view(states.shape[2], -1)
        try:
            sample_actions, sample_logp, _, _ = policy.sample(sample_states)
            if (not torch.isfinite(sample_actions).all()) or (not torch.isfinite(sample_logp).all()):
                print("  [Recurrent] Warning: Non-finite values; reverting weights.")
                policy.load_state_dict(behavior_policy.state_dict())
                error = True
            else:
                print("  [Recurrent] Policy outputs valid after update.")
        except Exception as e:
            print(f"  [Recurrent] Error during safety check: {e}")
            policy.load_state_dict(behavior_policy.state_dict())
            error = True

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # --- Logging ---
    writer.add_scalar('Loss/Collective', loss.item(), epoch)
    writer.add_scalar('Entropy', entropy.mean().item(), epoch)
    writer.add_scalar('Learning_Rate', current_lr, epoch)
    writer.add_scalar('GradNorm/Trunk', trunk_grad_norm, epoch)
    writer.add_scalar('GradNorm/Heads', head_grad_norm, epoch)

    try:
        if epoch % 10 == 0:
            for name, param in policy.named_parameters():
                if param.grad is not None and param.grad.numel() > 0:
                    if param.grad.isfinite().all():
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    except Exception as e:
        print(f"  [Recurrent] Error logging gradients: {e}")

    print(f"  [Recurrent] Loss={loss.item():.4f}  Entropy={entropy.mean().item():.4f}")

    del distances, indices, dist_dev, idx_dev, full_behavior_lp
    torch.cuda.empty_cache()
    return error


# ======================================================================
#  Main loop: recurrent policy + gradient surgery with collective entropy
# ======================================================================
def reinforce_recurrent_gradsurgery_collective(writer, epoch, env, policy, behavior_policy,
                                               optimizer, scheduler, discretizer,
                                               num_trajectories, trajectory_length, state_filter,
                                               num_agents, num_envs, k, env2agent,
                                               log_entropy_interval=40,chunk_length=1):
    """Single-rollout collection + gradient surgery with collective k-NN entropy
    for recurrent (GRU-based) policies.

    Uses collect_particles_recurrent to carry hidden state across timesteps.
    """
    # num_features = env.observation_manager.group_obs_dim["policy"][0]
    # num_actions = env.action_manager.action_term_dim[0]
    num_features = env.num_features
    num_actions = env.num_actions

    # --- Collect single rollout (recurrent: carries hidden across steps) ---
    states, actions, real_traj_lengths = collect_particles_recurrent(
        env, policy, 1, trajectory_length,
        num_features, num_actions, num_agents, num_envs, env2agent,
    )
    states, actions, real_traj_lengths = unpack_results((states, actions, real_traj_lengths))

    # --- Record action distributions ---
    record_actions_step(num_agents, env2agent, actions, states, epoch, writer)

    # --- Entropy estimation (all states pooled) ---
    start_entropy = time.time()
    with torch.no_grad():
        entropy = knn_entropy_estimation_torch(states, state_filter, real_traj_lengths, k=k)
    print(f"  [Recurrent] Time for entropy estimation: {time.time() - start_entropy:.2f}s")
    torch.cuda.empty_cache()

    # --- Gradient surgery training (recurrent) ---
    error = train_step_recurrent_gradsurgery_collective(
        writer, epoch, states, actions, state_filter,
        real_traj_lengths, policy, behavior_policy,
        optimizer, scheduler, k, entropy,
        env2agent, num_agents,
    )

    # --- Heatmap logging ---
    if epoch % log_entropy_interval == 0:
        state_filter_heatmap = [0, 1]
        vis_states = states[:, 1:, :, state_filter_heatmap].reshape(
            num_envs, trajectory_length, len(state_filter_heatmap)
        )
        _, _, image_fig = get_heatmap_fast(vis_states, discretizer)
        writer.add_figure('Heatmap entropy', image_fig, epoch)
        plt.close(image_fig)

    return None, error