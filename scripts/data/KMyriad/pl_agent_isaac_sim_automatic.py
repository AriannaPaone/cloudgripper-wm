
from csv import writer
import sys, os
sys.path.append(os.getcwd() )
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#from stable_worldmodel.envs.pusht import env
import torch
import scipy
import scipy.special
import time
import io
import math
from tqdm import tqdm
import copy
from torch.amp import autocast

from sklearn.neighbors import NearestNeighbors
from .estimators import knn_entropy_estimation_scipy, get_heatmap_fast,knn_entropy_estimation_torch,get_heatmap_fast_colored

import torch
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import numpy as np
import torch.nn.functional as F


#torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
float_type = torch.float64
int_type = torch.int64
eps = 1e-7


def collect_particles_chunked(
    env,
    policy,
    num_trajectories,
    trajectory_length,
    num_features,
    primitive_action_dim,
    chunk_length,
    num_agents,
    num_envs,
    env2agent,
):
    """
    Collect trajectories using an open-loop chunked policy (pre-allocated tensors).

    Returns:
        states:  [num_traj, T+1, num_envs, num_features]
        actions: [num_traj, T,   num_envs, primitive_action_dim]
        real_traj_lengths: [num_traj, 1, num_envs]
    """

    assert num_agents <= num_envs, "This function assumes num_agents <= num_envs."

    states = torch.zeros((num_trajectories, trajectory_length + 1, num_envs, num_features),
                         dtype=torch.float32, device=device)
    actions = torch.zeros((num_trajectories, trajectory_length, num_envs, primitive_action_dim),
                          dtype=torch.float32, device=device)
    real_traj_lengths = torch.full((num_trajectories, 1, num_envs),
                                   trajectory_length, dtype=torch.int32, device=device)

    env_idx = torch.arange(num_envs, device=device)
    policy.eval()

    for traj in tqdm(range(num_trajectories)):
        s, _ = env.reset()
        s = s["policy"]  # [num_envs, num_features]

        for t in range(trajectory_length):
            states[traj, t] = s

            # sample a new chunk at every chunk boundary
            if t % chunk_length == 0:
                with torch.no_grad():
                    a_chunk, _, _ = policy.sample(s)
                    a_chunk = a_chunk[env_idx, env2agent]
                    a_chunk = a_chunk.view(num_envs, chunk_length, primitive_action_dim)

            a = a_chunk[:, t % chunk_length]  # [num_envs, primitive_action_dim]
            actions[traj, t] = a

            obs, _, terminated, truncated, logs = env.step(a)
            s = obs["policy"]

            # optionally check termination flags returned by env.step and break if needed
            if terminated.any() or truncated.any():
                mask = terminated | truncated
                already_done = real_traj_lengths[traj, 0] < trajectory_length
                newly_done = mask & ~already_done
                real_traj_lengths[traj, 0] = torch.where(newly_done, t + 1, real_traj_lengths[traj, 0])

        states[traj, t + 1] = s

    return states, actions, real_traj_lengths

def collect_particles(env, policy, num_trajectories, trajectory_length,
                      num_features, num_actions, num_agents, num_envs, env2agent):
    assert num_agents <= num_envs, "This function assumes num_agents <= num_envs."

    states = torch.zeros((num_trajectories, trajectory_length + 1, num_envs, num_features),
                         dtype=torch.float32, device=device)
    actions = torch.zeros((num_trajectories, trajectory_length, num_envs, num_actions),
                          dtype=torch.float32, device=device)
    real_traj_lengths = torch.zeros((num_trajectories, 1, num_envs), 
                                    dtype=torch.int32, device=device)

    # Build env -> agent mapping (contiguous blocks; last agent takes remainder)
    # OLD
    # envs_per_agent = max(1, num_envs // num_agents)
    # env2agent = (torch.arange(num_envs, device=device) // envs_per_agent).clamp(max=num_agents - 1)
    real_traj_lengths = torch.full((num_trajectories, 1, num_envs), 
                                trajectory_length, dtype=torch.int32, device=device)

    env_idx = torch.arange(num_envs, device=device)
    policy.eval()
    for traj in tqdm(range(num_trajectories)):
        s, _ = env.reset()
        terminated, truncated = False, False

        for t in range(trajectory_length):
            states[traj, t] = s ["policy"]

            start_time = time.time()
            with torch.no_grad():
                # Single forward: returns [num_envs, num_agents, action_dim]
                all_actions, all_logp, all_mean = policy.sample(s["policy"])  # [num_envs, num_agents, action_dim]
                a = all_actions[env_idx, env2agent]   # -> [num_envs, action_dim]
            #print("Time for policy sample: ", time.time() - start_time)

            actions[traj, t] = a

            # step environments (keep same type env expects; original code passed torch)
            s, _,terminated,truncated,log = env.step(a)

            # optionally check termination flags returned by env.step and break if needed
            if terminated.any() or truncated.any():
                mask = terminated | truncated
                # Only update envs that haven't already been marked terminated
                already_done = real_traj_lengths[traj, 0] < trajectory_length
                newly_done = mask & ~already_done
                real_traj_lengths[traj, 0] = torch.where(newly_done, t + 1, real_traj_lengths[traj, 0])
            
        states[traj, t + 1] = s["policy"]

    return states, actions, real_traj_lengths

def unpack_results(res):
    """
    Args:
        res = (states, actions, real_traj_lengths, env2agent)
          states: [num_traj, T+1, num_envs, state_dim]
          actions: [num_traj, T,   num_envs, action_dim]
          real_traj_lengths: [num_traj, 1, num_envs]
    Returns:
        states, actions, real_traj_lengths, env2agent
        (all shapes consistent with collect_particles)
    """
    states, actions, real_traj_lengths = res

    # Ensure contiguous layout for safety
    states = states.contiguous()
    actions = actions.contiguous()
    real_traj_lengths = real_traj_lengths.squeeze(1).contiguous()  # [num_traj, num_envs]

    return states, actions, real_traj_lengths

def reinforce_collection_and_compute_knn(writer,epoch,env, policy,behavior_policy,optimizer,
                                         scheduler,discretizer, num_trajectories, 
                                         trajectory_length, state_filter,num_agents, num_envs, 
                                         k,env2agent, chunk_length=5, log_entropy_interval=40, 
                                         log_kl_interval=None, dim_weights=None):
    
    
    # num_features = env.observation_manager.group_obs_dim["policy"][0]
    # num_actions = env.action_manager.action_term_dim[0]
    num_features = env.num_features
    num_actions = env.num_actions
    
    if chunk_length > 1:
        states, actions, real_traj_lengths = collect_particles_chunked(
                                                env,
                                                policy,
                                                1,
                                                trajectory_length=trajectory_length,
                                                num_features=num_features,
                                                primitive_action_dim=num_actions,
                                                chunk_length=chunk_length,
                                                num_agents=num_agents,
                                                num_envs=num_envs,
                                                env2agent=env2agent
                                                )
    else:
        states,actions, real_traj_lengths = collect_particles(env, policy, 1, trajectory_length, 
                                            num_features,num_actions,num_agents,num_envs,env2agent)


    states, actions, real_traj_lengths = unpack_results((states, actions, real_traj_lengths))


    # --- temporary diagnostic ---
    cube = states[..., 5:7]
    print(f"  cube range: {cube.min().item():.4f} .. {cube.max().item():.4f}  "
          f"unique positions: {len(torch.unique(cube.reshape(-1, 2), dim=0))} / {cube.reshape(-1,2).shape[0]}")
    # ---

    #torch.save(states, f'states_epoch_{epoch}_agents_{num_agents}_time_{time.time()}_trajs_{num_trajectories}.pt')
    record_actions_step(num_agents, env2agent, actions, states, epoch, writer)  

    start_entropy = time.time()
    with torch.no_grad():
        entropy = knn_entropy_estimation_torch(states,state_filter,real_traj_lengths, k=k, dim_weights=dim_weights)
    print("Time for entropy computation: ", time.time() - start_entropy)
    torch.cuda.empty_cache()

    # Training phase: update policy networks
    start_update = time.time()
    error = False

    error = train_step(writer,epoch, states,actions,state_filter,
                    real_traj_lengths, policy,behavior_policy,
                    optimizer,scheduler,k,entropy,env2agent, chunk_length=chunk_length, dim_weights=dim_weights)

    # if num_agents == 1:
    #     error = train_step(writer,epoch, states,actions,state_filter,
    #             real_traj_lengths, policy,behavior_policy,
    #             optimizer,scheduler,k,entropy,env2agent, chunk_length=chunk_length)
    # else:
    #     error = train_step_twophase(writer,epoch, states,actions,state_filter,
    #             real_traj_lengths, policy,behavior_policy,
    #             optimizer,scheduler,k,entropy,env2agent, chunk_length=chunk_length)

    print("Time for policy update: ", time.time() - start_update)
    
    # EVALUATION STEP
    if epoch % log_entropy_interval == 0:
    ## Calculate total Heatmap
        state_filter_heatmap = [5,6] #[0,1] this way its a heatmap on the cube position
        #     # take randomly 1-d vector of 100 num_env indices
        #     #trajectory_filter = 10
        #     #random_env_indices = torch.randperm(num_envs)[:100]
        #     #random_env_indices = torch.arange(trajectory_filter)
        _, _, image_fig = get_heatmap_fast(
            states[:, 1:, :, state_filter_heatmap].reshape(
                num_trajectories * num_agents, trajectory_length, len(state_filter_heatmap)),
            discretizer)
        writer.add_figure(f'Heatmap entropy', image_fig, epoch)
        plt.close(image_fig)
    print("Time for policy update: ", time.time() - start_update)

    kl_divs = None
    if log_kl_interval is not None: 
        if epoch % log_kl_interval == 0:
            # Calculate kl divergence between agents
            #kl_divs = calculate_vector_kl_by_agent(states,env2agent)
            kl_divs = knn_kl_agents_subspace_scipy(
                states.cpu().numpy()[:, :, :, :len(state_filter)],  # only first 2 dims for kl
                env2agent.cpu().numpy(),
                k=k,
                subspace_dims=state_filter,
                standardize=False,
                rng=None,
                eps=1e-12,
            )
            return kl_divs, error
        
    return None,error

def train_step(writer, epoch, states, actions, state_filter, real_traj_lengths,
               policy, behavior_policy, optimizer, scheduler, k, entropy, env2agent, chunk_length=10, dim_weights=None):

    error = False
    #with torch.no_grad():
        #distances, indices = compute_distances_all_envs_global_knn(states, state_filter, real_traj_lengths, k)
        #distances, indices = compute_distances_all_envs_global_knn_torch(states, state_filter,real_traj_lengths, k)
    
    distances, indices = compute_distances_all_envs_global_knn_torch(states, state_filter,real_traj_lengths, k, dim_weights=dim_weights)
    
    behavior_policy.eval()
    for p in behavior_policy.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    start_loss = time.time()

    entropy_loss = -compute_entropy(behavior_policy, policy, states, actions,
        num_traj=states.shape[0],
        traj_length=states.shape[1] - 1,
        k=k,
        distances=distances,
        indices=indices,
        states_filter=state_filter,
        real_traj_lengths=real_traj_lengths,
        env2agent=env2agent,
        chunk_length=chunk_length
    )
    
    #mutual information divergence
    #mi_div = head_state_mutual_information(states, env2agent, state_filter)

    #kl regularization
    #means, log_stds = policy.forward(states.reshape(-1, states.shape[-1]))   # [N, H, A]
    #kl_div = multihead_kl_divergence(means, log_stds)
    
    #means, log_stds = policy.forward(states.reshape(-1, states.shape[-1]))   # [N, H, A]
    #cosine_div = head_cosine_diversity(means)

    loss = entropy_loss 
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

    mem_info = get_model_memory(policy,optimizer)
    #print(f"Memory Info before step: {mem_info}")

    optimizer.step()
    print(f"Time for loss computation: ", time.time() - start_loss)

    ## Safety Check - if the loss is nan or inf, revert to behavior policy weights and skip scheduler step
    ### add safety check to see if a new policy.sample does not give nans
    with torch.no_grad():
        sample_states = states[0, 0].contiguous().view(states.shape[2], -1)  # [num_envs, state_dim]
        try:
            sample_actions, sample_logp, _ = policy.sample(sample_states)
            if (not torch.isfinite(sample_actions).all()) or (not torch.isfinite(sample_logp).all()):
                print("Warning: Non-finite values detected in policy outputs after update; reverting weights.")
                policy.load_state_dict(behavior_policy.state_dict())
                error = True
            else:
                print("Policy outputs valid after update.")
        except Exception as e:
            print(f"Error during policy sampling after update: {e}")
            policy.load_state_dict(behavior_policy.state_dict())
            error = True

    #use scheduler
    #behavior_policy[i].load_state_dict(policy[i].state_dict())

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    record_train_step(policy,loss,entropy_loss,current_lr,entropy,epoch,writer)

    print(f"Loss: {loss.item()}, Entropy: {entropy.mean().item()}")
    del distances, indices, loss
    torch.cuda.empty_cache()
    return error

def compute_latent_entropy(policy, states, real_traj_lengths, k=5,
                           state_filter=None, recon_weight=1.0, forward_chunk=65536):
    """Compute k-NN entropy in the trunk's projected latent space.

    Uses ALL collected states (no subsampling) so the trunk gets complete feedback.
    Forward pass is chunked to avoid OOM.  scipy NearestNeighbors runs k-NN on
    CPU, then the k-th neighbor distance is recomputed in torch for gradient flow.

    Args:
        policy: PolicyMultiheadNetwork (uses policy.net + policy.latent_proj)
        states: [B, T+1, H, D] full collected states
        real_traj_lengths: [B, H] valid trajectory lengths
        k: number of neighbors
        state_filter: list of int, observation indices used by Phase 2 entropy.
            When provided, a reconstruction loss is computed.
        recon_weight: float, weight for the reconstruction loss (default 1.0)
        forward_chunk: batch size for the trunk forward pass (avoid OOM)

    Returns:
        entropy: scalar tensor WITH gradient through policy.net and policy.latent_proj
        recon_loss: scalar tensor (0 if state_filter is None)
    """
    # flatten valid states (exclude final timestep, same as importance weights)
    valid_states = states[:, :-1, :, :]  # [B, T, H, D]
    flat_states = valid_states.reshape(-1, states.shape[-1])  # [N, D]
    N_total = flat_states.shape[0]

    # chunked forward through trunk + projection (differentiable, avoids OOM)
    latent_chunks = []
    for start in range(0, N_total, forward_chunk):
        end = min(start + forward_chunk, N_total)
        trunk_out = policy.net(flat_states[start:end])
        latent_chunks.append(policy.latent_proj(trunk_out))
    latent = torch.cat(latent_chunks, dim=0)  # [N, latent_proj_dim]

    # --- Reconstruction loss: anchor latent_proj to state-filter subspace ---
    if state_filter is not None:
        target = flat_states[:, state_filter].detach()  # [N, len(state_filter)]
        recon_loss = F.mse_loss(latent, target)
    else:
        recon_loss = torch.tensor(0.0, device=latent.device)

    N, d = latent.shape

    latent_np = latent.detach().cpu().numpy().astype(np.float32)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', algorithm='auto')
    nbrs.fit(latent_np)
    _, indices_np = nbrs.kneighbors(latent_np)

    # recompute k-th distance in torch for gradient flow
    kth_indices = torch.tensor(indices_np[:, k], dtype=torch.long, device=latent.device)
    diff = latent - latent[kth_indices].detach()
    kth_dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # [N]

    # H ≈ d * mean(log r_k) + log(N) - digamma(k) + log(V_d)
    log_V_d = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1)
    entropy = (d * torch.mean(torch.log(kth_dist + 1e-12))
               + math.log(N) - scipy.special.digamma(k) + log_V_d)
    return entropy, recon_loss

def compute_latent_entropy_from_knn(policy, states, indices, k=5, forward_chunk=65536):
    """Compute k-NN entropy of the trunk's latent space using STATE-SPACE k-NN indices.

    Instead of running a separate k-NN in latent space, reuses the state-space
    k-NN indices (already computed by compute_distances_all_envs_global_knn_torch).
    For each point, the k-th neighbor is defined in state space, but the distance
    is measured in latent space.  This aligns the trunk objective with the head
    entropy objective.

    Args:
        policy:  PolicyMultiheadNetwork (uses policy.net + policy.latent_proj)
        states:  [B, T+1, H, D] full collected states
        indices: [N, k+1] int64, k-NN indices from state-space (on cpu or gpu)
        k:       which neighbor to use
        forward_chunk: batch size for the trunk forward pass (avoid OOM)

    Returns:
        entropy: scalar tensor WITH gradient through policy.net and policy.latent_proj
    """
    # Flatten valid states in the same order as compute_distances_all_envs_global_knn_torch
    valid_states = states[:, :-1, :, :]  # [B, T, H, D]
    flat_states = valid_states.reshape(-1, states.shape[-1])  # [N, D]
    N = flat_states.shape[0]

    # Chunked forward pass through trunk + latent_proj (differentiable)
    latent_chunks = []
    for start in range(0, N, forward_chunk):
        end = min(start + forward_chunk, N)
        trunk_out = policy.net(flat_states[start:end])
        latent_chunks.append(policy.latent_proj(trunk_out))
    latent = torch.cat(latent_chunks, dim=0)  # [N, latent_dim]

    d = latent.shape[1]  # latent dimensionality

    # Use the k-th state-space neighbor index
    kth_indices = indices[:, k].to(latent.device, dtype=torch.long)  # [N]

    # Compute latent distance to the state-space k-th neighbor (detach neighbor)
    diff = latent - latent[kth_indices].detach()
    kth_dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # [N]

    # k-NN entropy estimator: H ≈ d * mean(log r_k) + log(N) - digamma(k) + log(V_d)
    log_V_d = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1)
    entropy = (d * torch.mean(torch.log(kth_dist + 1e-12))
               + math.log(N) - scipy.special.digamma(k) + log_V_d)
    return entropy

def train_step_twophase(writer, epoch, states, actions, state_filter, real_traj_lengths,
                        policy, behavior_policy, optimizer, scheduler, k, entropy,
                        env2agent, chunk_length=10, balancer=None):
    """Two-phase single-step training: latent entropy trunk + per-head entropy heads.

    Phase 1: Maximize k-NN entropy of the trunk's projected latent space using ALL
             trajectories pooled (no head distinction). Only trunk + latent_proj get grads.
             Uses state-space k-NN indices to align trunk and head objectives.
    Phase 2: Maximize per-head importance-weighted entropy with trunk detached.
             Only head params get grads.
    Single optimizer.step() at the end.
    """
    error = False

    # === k-NN distances for head-phase entropy (computed ONCE) ===
    distances, indices = compute_distances_all_envs_global_knn_torch(
        states, state_filter, real_traj_lengths, k)

    behavior_policy.eval()
    for p in behavior_policy.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()
    policy.train()
    start_loss = time.time()
    optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Phase 1: Trunk — latent entropy using state-space k-NN indices
    #          + reconstruction loss to align latent_proj with state_filter
    # ------------------------------------------------------------------
    #latent_ent = compute_latent_entropy_from_knn(policy, states, indices, k=k)
    recon_weight = 1.0
    latent_ent,recon_loss = compute_latent_entropy(
        policy, states, real_traj_lengths, k=k,
        state_filter=state_filter, recon_weight=recon_weight)
    latent_loss = -(latent_ent) + recon_weight * recon_loss
    latent_loss.backward()
    # trunk + latent_proj have grads. Head params have None.

    # ------------------------------------------------------------------
    # Phase 2: Heads — per-head importance-weighted entropy (trunk frozen)
    # ------------------------------------------------------------------
    with policy.detached_trunk():
        head_entropy_loss = -compute_entropy(
            behavior_policy, policy, states, actions,
            num_traj=states.shape[0],
            traj_length=states.shape[1] - 1,
            k=k, distances=distances, indices=indices,
            states_filter=state_filter,
            real_traj_lengths=real_traj_lengths,
            env2agent=env2agent,
            chunk_length=chunk_length,
        )
        head_entropy_loss.backward()
    # trunk grads unchanged (phase 1 only). Head grads accumulated from phase 2.

    # ------------------------------------------------------------------
    # Single optimiser step
    # ------------------------------------------------------------------
    #torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    print(f"Time for loss computation: ", time.time() - start_loss)

    # === safety check ===
    with torch.no_grad():
        sample_states = states[0, 0].contiguous().view(states.shape[2], -1)
        try:
            sample_actions, sample_logp, _ = policy.sample(sample_states)
            if (not torch.isfinite(sample_actions).all()) or (not torch.isfinite(sample_logp).all()):
                print("Warning: Non-finite values detected in policy outputs after update; reverting weights.")
                policy.load_state_dict(behavior_policy.state_dict())
                error = True
            else:
                print("Policy outputs valid after update.")
        except Exception as e:
            print(f"Error during policy sampling after update: {e}")
            policy.load_state_dict(behavior_policy.state_dict())
            error = True

    scheduler.step()
    trunk_actual_lr = optimizer.param_groups[0]['lr']
    head_actual_lr = optimizer.param_groups[1]['lr']  # same optimizer, same lr
    record_train_step(policy, head_entropy_loss, head_entropy_loss, trunk_actual_lr, entropy, epoch, writer)

    writer.add_scalar('Loss/Latent_Entropy', latent_ent.item(), epoch)
    writer.add_scalar('Loss/Head_Entropy', -head_entropy_loss.item(), epoch)
    #writer.add_scalar('Loss/Recon_Loss', recon_loss.item(), epoch)

    del distances, indices
    torch.cuda.empty_cache()
    return error

def train_step_new(writer, epoch, states, actions, state_filter, real_traj_lengths,
               policy, behavior_policy, optimizer, scheduler, k, entropy, env2agent, 
               chunk_length=10, num_update_steps=5):

    error = False

    # Compute k-NN ONCE on the collected data
    distances, indices = compute_distances_all_envs_global_knn_torch(
        states, state_filter, real_traj_lengths, k)

    behavior_policy.eval()
    for p in behavior_policy.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()
    policy.train()
    start_loss = time.time()

    # Reuse experience for multiple gradient steps
    for step in range(num_update_steps):
        optimizer.zero_grad(set_to_none=True)

        entropy_loss = -compute_entropy(
            behavior_policy, policy, states, actions,
            num_traj=states.shape[0],
            traj_length=states.shape[1] - 1,
            k=k, distances=distances, indices=indices,
            states_filter=state_filter,
            real_traj_lengths=real_traj_lengths,
            env2agent=env2agent,
            chunk_length=chunk_length)

        loss = entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    print(f"Time for loss computation: ", time.time() - start_loss)

    # Safety check (unchanged)
    with torch.no_grad():
        sample_states = states[0, 0].contiguous().view(states.shape[2], -1)
        try:
            sample_actions, sample_logp, _ = policy.sample(sample_states)
            if (not torch.isfinite(sample_actions).all()) or (not torch.isfinite(sample_logp).all()):
                print("Warning: Non-finite values detected; reverting weights.")
                policy.load_state_dict(behavior_policy.state_dict())
                error = True
            else:
                print("Policy outputs valid after update.")
        except Exception as e:
            print(f"Error during policy sampling after update: {e}")
            policy.load_state_dict(behavior_policy.state_dict())
            error = True

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    record_train_step(policy, loss, entropy_loss, current_lr, entropy, epoch, writer)

    print(f"Loss: {loss.item()}, Entropy: {entropy.mean().item()}")
    del distances, indices, loss
    torch.cuda.empty_cache()
    return error

def multi_train_step(writer, epoch, states, actions, state_filter, real_traj_lengths,
               policy, behavior_policy, optimizer, scheduler, k, entropy, env2agent, chunk_length=10):

    # --- 1. Freeze the behavior policy ---
    # The behavior_policy represents the policy that collected the data. It must not change.
    behavior_policy.load_state_dict(policy.state_dict()) # Create a frozen copy
    behavior_policy.eval()
    for p in behavior_policy.parameters():
        p.requires_grad_(False)

    # --- 2. Define update loop hyperparameters ---
    num_update_epochs = 4  # How many times to iterate over the data
    #mini_batch_size = 4096 # The size of mini-batches for each update

    distances, indices = compute_distances_all_envs_global_knn_torch(states, state_filter,real_traj_lengths, k)

    # --- 3. The Update Loop ---
    policy.train() # Set the target policy to training mode
    for i in range(num_update_epochs):
        # Optional but recommended: Reshuffle data each epoch
        # (This requires flattening all your data first)
        
        # Calculate loss on the *entire* batch for simplicity here
        # A full PPO would use mini-batches inside this loop
        
        optimizer.zero_grad()

        entropy_loss = -compute_entropy(behavior_policy, policy, states, actions,
                                num_traj=states.shape[0],
                                traj_length=states.shape[1] - 1,
                                k=k,
                                distances=distances,
                                indices=indices,
                                states_filter=state_filter,
                                real_traj_lengths=real_traj_lengths,
                                env2agent=env2agent,
                                chunk_length=chunk_length)
        
        # ---- KL HEAD DIVERSITY REGULARIZATION ----
        with torch.no_grad():
            sample_states = states[:, :-1].reshape(-1, states.shape[-1])

        # kl regularization between heads
        #means, log_stds = policy.forward(sample_states)   # [N, H, A]
        #kl_div = multihead_kl_divergence(means, log_stds)

        # mutual information regularization between heads
        mi_div = head_state_mutual_information(states, env2agent, state_filter)


        lambda_kl = 0.05   # start small
        loss = entropy_loss - lambda_kl * mi_div

        loss.backward()
        # limit gradient
        torch.nn.utils.clip_grad_value_(policy.parameters(), 10.0)
        writer.add_scalar(f'Loss/Entropy_Loss_', entropy_loss.item(), int(epoch + i))
        writer.add_scalar(f'Loss/MI_Head_State_', mi_div.item(), int(epoch + i))
        writer.add_scalar(f'Loss/MiniBatch_Update_Step_', loss.item(), int(epoch + i))
        optimizer.step()

    # Log the loss for each update step if desired
    writer.add_scalar(f'Loss/Update_Step_{i}', loss.item(), epoch)

    writer.add_scalar(f'Loss/Agent', loss.item(), epoch)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar(f'Learning_Rate/Agent', current_lr, epoch)

    try:
        if epoch % 10 == 0:  # Less frequent logging
            for name, param in policy.named_parameters():
                if param.grad is not None and param.grad.numel() > 0:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    except Exception as e:
        print(f"Error logging gradients for Agent: {e}")

    writer.add_scalar('Entropy',  entropy.mean().item(), epoch)

    print(f"Loss: {loss.item()}, Entropy: {entropy.mean().item()}")
    del distances, indices, loss
    torch.cuda.empty_cache()

# def evaluate_entropy(env,num_envs,policy,writer,state_filter,epoch,k,num_workers=10,num_trajectories=100, trajectory_length=400):
#     num_features = env.unwrapped.observation_space.shape[0]
#     num_actions = env.unwrapped.action_space.shape[0]
    
#     states = torch.zeros((num_trajectories, trajectory_length + 1, num_envs, env.unwrapped.observation_space.shape[0]),
#                         dtype=torch.float32,device="cpu")
    
    
#     res = Parallel(n_jobs=num_workers)(
#         delayed(collect_particles)(env, policy, int(num_trajectories/num_workers), trajectory_length, state_filter, num_features,num_actions,num_envs, k)
#         for _ in range(num_workers)
#     )

#     # Unpack results
#     states_list, _ = zip(*res)
#     states = torch.cat(states_list, dim=0)
    
#     writer.add_histogram('State_Distribution_100t', states[:, :, :, state_filter].reshape(-1, len(state_filter)), epoch)
    
#     datetime = time.strftime("%Y%m%d-%H%M%S")

#     #Save states to file
#     #torch.save(states[:, :, :, state_filter], f"/content/logs/states_{datetime}.pt")

#     #Combine Trajectories
#     entropy = knn_entropy_estimation_scipy(states[:, :, :, state_filter].reshape(1,-1 , len(state_filter)), k=k)

#     writer.add_scalar('Entropy_100tau', entropy.mean().item(), epoch)
#     return

def compute_importance_weights(
    behavioral_policy,
    target_policy,
    states,
    actions,
    num_traj,
    real_traj_lengths,
    env2agent,
    *,
    mini_batch_size: int = 16384,
):
    device = states.device
    num_traj = states.shape[0]
    T = actions.shape[1]
    num_agents = actions.shape[2]
    state_dim = states.shape[-1]

    # Flatten B x T x H tensors to (B·T·H)
    flat_states = states.reshape(-1, state_dim)
    flat_actions = actions.reshape(-1, actions.shape[-1])
    # flat_heads = (
    #     env2agent.to(device)
    #     .view(1, -1)
    #     .expand(num_traj * T, -1)
    #     .reshape(-1)
    # )

    flat_heads = env2agent.repeat(num_traj * T)
    
    num_total_samples = flat_states.shape[0]

    target_lp_chunks = []
    behavior_lp_chunks = []

    for start in range(0, num_total_samples, mini_batch_size):
        end = min(start + mini_batch_size, num_total_samples)
        s = flat_states[start:end]
        a = flat_actions[start:end]
        h = flat_heads[start:end]

        target_lp_chunks.append(target_policy.get_log_p_select(s, a, h))

        with torch.no_grad():
            behavior_lp_chunks.append(behavioral_policy.get_log_p_select(s, a, h))

    target_lp_flat = torch.cat(target_lp_chunks, dim=0)
    behavior_lp_flat = torch.cat(behavior_lp_chunks, dim=0)

    log_ratios = (target_lp_flat - behavior_lp_flat).reshape(num_traj, T, num_agents)

    rtl = real_traj_lengths.squeeze(-1) if real_traj_lengths.dim() == 3 else real_traj_lengths
    if rtl.shape[0] != num_traj:
        rtl = rtl.transpose(0, 1)
    rtl = rtl.to(device)

    time_idx = torch.arange(T, device=device, dtype=rtl.dtype).view(1, T, 1)
    valid_mask = time_idx < rtl.unsqueeze(1)
    valid_mask_f = valid_mask.to(log_ratios.dtype)

    cumulative_log_ratios = torch.cumsum(log_ratios * valid_mask_f, dim=1)
    importance_weights = torch.exp(cumulative_log_ratios) * valid_mask_f

    # valid_iw = importance_weights[valid_mask]
    # if valid_iw.numel() == 0:
    #     return log_ratios.new_zeros(0, requires_grad=True)

    # eps = torch.finfo(valid_iw.dtype).eps
    # normalized_iw = valid_iw / valid_iw.sum()
    #return normalized_iw

    weighted_iw = importance_weights * valid_mask_f
    normalized_iw = weighted_iw / (weighted_iw.sum() + 1e-12)
    return normalized_iw.reshape(-1)  # always [B*T*H]

def compute_importance_weights_chunked(
    behavioral_policy,
    target_policy,
    states,
    actions,
    real_traj_lengths,
    env2agent,
    *,
    mini_batch_size: int = 16384*100,
    chunk_length: int = 5,
    dense_evaluation: bool = True,
):
    """
    Importance sampling for open-loop chunked policies.
    
    Fixed: target_policy calculation is now OUTSIDE torch.no_grad()
    """
    device = states.device
    B, T, H, primitive_action_dim = actions.shape
    state_dim = states.shape[-1]
    
    if dense_evaluation:
        # Dense evaluation: Evaluate policy at every step using sliding window
        # This provides gradients for every state, solving the "sparse example" problem.
        stride = 1
        num_valid_starts = T - chunk_length + 1
        if num_valid_starts <= 0:
             return torch.zeros(B * T * H, device=device, dtype=torch.float32, requires_grad=True)
             
        # States: [B, num_valid_starts, H, D]
        valid_states = states[:, :num_valid_starts, :, :]
        
        # Actions: Unfold to get sliding windows [B, num_valid_starts, H, D, K]
        # Then permute to [B, num_valid_starts, H, K, D]
        chunked_actions = actions.unfold(1, chunk_length, stride)
        chunked_actions = chunked_actions.permute(0, 1, 2, 4, 3)
        
        # Flatten
        flat_states = valid_states.reshape(-1, state_dim)
        flat_actions = chunked_actions.reshape(-1, chunk_length * primitive_action_dim)
        flat_heads = env2agent.repeat(B * num_valid_starts)
        
    else:
        # Sparse evaluation: Only at chunk boundaries
        assert T % chunk_length == 0
        T_chunks = T // chunk_length
        
        # 1. Pick states at the start of each chunk
        chunk_start_states = states[:, 0:T:chunk_length, :, :]
        
        # 2. Reshape actions into chunks: [B, T_chunks, H, chunk_len * act_dim]
        # We need to ensure the dimensions align for get_log_p_select
        flat_chunk_actions = actions.view(B, T_chunks, chunk_length, H, -1).permute(0, 1, 3, 2, 4)
        flat_chunk_actions = flat_chunk_actions.reshape(B, T_chunks, H, -1)
        
        # 3. Flatten for policy evaluation
        flat_states = chunk_start_states.reshape(-1, state_dim)
        flat_actions = flat_chunk_actions.reshape(-1, chunk_length * actions.shape[-1])
        flat_heads = env2agent.repeat(B * T_chunks)

    num_samples = flat_states.shape[0]

    # -----------------------------------------------------------
    #  Batched Log-Prob Evaluation
    # -----------------------------------------------------------
    target_lp_chunks   = []
    behavior_lp_chunks = []

    for start in range(0, num_samples, mini_batch_size):
        end = min(start + mini_batch_size, num_samples)
        s = flat_states[start:end]
        a = flat_actions[start:end]
        h = flat_heads[start:end]

        with torch.no_grad():
            # with autocast(device_type="cuda", dtype=torch.float16):
            #     blp = behavioral_policy.get_log_p_select(s, a, h)
            blp = behavioral_policy.get_log_p_select(s, a, h)
            behavior_lp_chunks.append(blp)

        # with autocast(device_type="cuda", dtype=torch.float16):
        #     tlp = target_policy.get_log_p_select(s, a, h)
        tlp = target_policy.get_log_p_select(s, a, h)
        target_lp_chunks.append(tlp)

    target_lp_flat   = torch.cat(target_lp_chunks, dim=0)
    behavior_lp_flat = torch.cat(behavior_lp_chunks, dim=0)

    # -----------------------------------------------------------
    #  Compute Cumulative Weights
    # -----------------------------------------------------------
    if dense_evaluation:
        
        # limit the degeneration of the impoerance weight
        log_ratios = target_lp_flat - behavior_lp_flat
        
        # [B, num_valid_starts, H]
        log_ratios = log_ratios.view(B, num_valid_starts, H)
        # For dense evaluation, we treat each sliding window as an independent sample (RHC style)
        # We do NOT cumsum because windows are overlapping.
        importance_weights_valid = torch.exp(log_ratios)
        
        # Pad to length T
        padding = torch.ones(B, chunk_length - 1, H, device=device, dtype=target_lp_flat.dtype)
        importance_weights = torch.cat([importance_weights_valid, padding], dim=1)
        
    else:
        # Calculate log ratio per chunk
        log_ratios_chunk = (target_lp_flat - behavior_lp_flat).view(B, T_chunks, H)
        
        # Cumulative sum across CHUNKS (Product of importance weights)
        cum_log_ratios_chunk = torch.cumsum(log_ratios_chunk, dim=1)
        
        # BROADCAST: Every step within a chunk shares the weight of that chunk's likelihood
        # We use repeat_interleave so that index 0..chunk_length-1 gets chunk weight 0
        importance_weights = torch.exp(cum_log_ratios_chunk).repeat_interleave(chunk_length, dim=1)

    # -----------------------------------------------------------
    #  Masking and Normalization
    # -----------------------------------------------------------
    
    rtl = real_traj_lengths.squeeze(-1) if real_traj_lengths.dim() == 3 else real_traj_lengths
    time_idx = torch.arange(T, device=device).view(1, T, 1)
    valid_mask = time_idx < rtl.to(device).unsqueeze(1)
    
    # Apply mask and normalize
    weighted_iw = importance_weights * valid_mask
    normalized_iw = weighted_iw / (weighted_iw.sum() + 1e-12)

    # Return as float32 to match network weights
    # Return full flattened tensor (B*T*H) to align with states/indices in compute_entropy
    return normalized_iw.reshape(-1)#.to(torch.float32)

def compute_temporal_influence_weights(
    behavioral_policy,
    target_policy,
    states,
    actions,
    real_traj_lengths,
    env2agent,
    *,
    chunk_length: int = 5,
    mini_batch_size: int = 16384,
):
    device = states.device
    B, T, H, action_dim = actions.shape
    state_dim = states.shape[-1]

    # ------------------------------------------------------------
    # 1. Flatten
    # ------------------------------------------------------------
    flat_states  = states.reshape(-1, state_dim)
    flat_actions = actions.reshape(-1, action_dim)
    flat_heads   = env2agent.repeat(B * T)

    num_samples = flat_states.shape[0]

    target_lp   = torch.empty(num_samples, device=device, dtype=torch.float32)
    behavior_lp = torch.empty_like(target_lp)

    # ------------------------------------------------------------
    # 2. Step-wise log ratios
    # ------------------------------------------------------------
    for start in range(0, num_samples, mini_batch_size):
        end = min(start + mini_batch_size, num_samples)
        s = flat_states[start:end]
        a = flat_actions[start:end]
        h = flat_heads[start:end]

        target_lp[start:end] = target_policy.get_log_p_select(s, a, h)

        with torch.no_grad():
            behavior_lp[start:end] = behavioral_policy.get_log_p_select(s, a, h)

    log_ratios = (target_lp - behavior_lp).view(B, T, H)

    # ------------------------------------------------------------
    # 3. Sliding-window accumulation (NO in-place ops)
    # ------------------------------------------------------------
    influence_log = torch.zeros_like(log_ratios)

    for k in range(chunk_length):
        if k == 0:
            influence_log = influence_log + log_ratios
        else:
            influence_log[:, :-k, :] = influence_log[:, :-k, :] + log_ratios[:, k:, :]

    influence = torch.exp(influence_log)

    # ------------------------------------------------------------
    # 4. Mask invalid steps (NO in-place ops)
    # ------------------------------------------------------------
    if real_traj_lengths.dim() == 3:
        real_traj_lengths = real_traj_lengths.squeeze(1)

    time_idx = torch.arange(T, device=device).view(1, T, 1)
    valid = time_idx < real_traj_lengths.unsqueeze(1)

    influence = influence * valid

    # ------------------------------------------------------------
    # 5. Normalize (NO in-place ops)
    # ------------------------------------------------------------
    influence_sum = influence.sum()
    influence = influence / (influence_sum + 1e-12)

    return influence.reshape(-1)

def compute_entropy(behavioral_policy, target_policy, states, actions,
                    num_traj, traj_length, k, distances, env2agent,
                    indices, states_filter=[0,1], real_traj_lengths=None, chunk_length=10,
                    entropy_row_chunk=65536):
    
    #trajectory_length = states.shape[1] - 1  # Exclude final state
    filtered_states = states[:, :-1, :]  # shape: [batch, time, dim]
    #importance weight are calculated over the entire observation dimensions while the entropy not
    # because you use the same policy network that generate the actions
    
    if chunk_length > 1:
        importance_weights = compute_importance_weights_chunked(behavioral_policy, target_policy, filtered_states, actions,
                                                                real_traj_lengths, env2agent, chunk_length=chunk_length)
    else:
        importance_weights = compute_importance_weights(behavioral_policy, target_policy, filtered_states, actions,
                                                        num_traj,real_traj_lengths,env2agent)
    
    #importance_weights = compute_temporal_influence_weights(behavioral_policy, target_policy, filtered_states, actions,
    #                                                       real_traj_lengths, env2agent, chunk_length=5)
    
        d = len(states_filter)
    
    d = len(states_filter)
    eps = 1e-6
    
    distances = distances.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True)

    k_tensor = torch.tensor(k, dtype=torch.float32)
    B = torch.log(k_tensor) - torch.tensor(scipy.special.digamma(k), dtype=torch.float32)
    G = torch.tensor(scipy.special.gamma(d / 2 + 1), dtype=torch.float32)

    # compute weights sum for each particle
    weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)

    # compute volume for each particle
    volumes = (torch.pow(distances[:, k], d) * torch.pow(torch.tensor(torch.pi), d/2)) / G

    entropy = - torch.sum((weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)) + B

    return entropy  # Return the entropy and the gradients for debugging purposes

def compute_distances_all_envs_global_knn_torch(states, states_filter, real_traj_lengths, k=500, device='cuda', dim_weights=None):
    """
    Compute k-NN globally across all trajectories and environments using a unified reshaping method.
    """
    # Original states shape: [num_trajectories, trajectory_length+1, num_envs, state_dim]
    num_features = states.shape[-1]

    # --- 1. Slice and Reshape (This now matches compute_importance_weights) ---
    # Slice to get the valid timesteps [B, T, H, D] to align with actions.
    valid_states = states[:, :-1, :, :]

    # Reshape to flatten the trajectory, time, and env dimensions into one.
    # The order will be identical to the one used for importance weights.
    flat_states = valid_states.reshape(-1, num_features)

    # Filter the state dimensions for the KNN calculation.
    filtered_states = flat_states[:, states_filter].cpu().numpy()  

    if dim_weights is not None:
            filtered_states = filtered_states * np.asarray(dim_weights, dtype=np.float32)

    # --- 2. Run KNN on the correctly ordered data ---
    # Using 'auto' algorithm is fine, it will choose the best one (e.g., ball_tree, kd_tree).
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto')
    
    # We run KNN on the CPU as scikit-learn requires it.
    nbrs.fit(filtered_states)
    distances, indices = nbrs.kneighbors(filtered_states)

    # --- 3. Return results (can be kept on CPU to save VRAM, as discussed) ---
    return (
        torch.tensor(distances, dtype=torch.float32, device='cpu'),
        torch.tensor(indices, dtype=torch.int64, device='cpu')
    )

# def compute_distances_all_envs_global_knn_faiss(states, states_filter, real_traj_lengths, k=500, device='cuda'):
#     num_features = states.shape[-1]
#     valid_states = states[:, :-1, :, :]
#     flat_states = valid_states.reshape(-1, num_features)

#     filtered_states = flat_states[:, states_filter].cpu().numpy().astype(np.float32)
#     n, d = filtered_states.shape

#     # FAISS CPU brute-force (still faster than sklearn)
#     index = faiss.IndexFlatL2(d)
#     index.add(filtered_states)
#     sq_distances, indices = index.search(filtered_states, k + 1)

#     distances = np.sqrt(np.maximum(sq_distances, 0.0))

#     return (
#         torch.tensor(distances, dtype=torch.float32, device='cpu'),
#         torch.tensor(indices, dtype=torch.int64, device='cpu'),
#     )

def compute_knn_policy_loss(policy, states, actions, heads, distances, k, d=2):
    """
    REINFORCE with Causal Rewards and k-NN State Entropy.
    """
    # states: [B, T, H, D], actions: [B, T, H, A_dim]
    B, T, H, _ = actions.shape
    device = states.device

    # 1. Get step-wise Log-Probabilities
    # Shape: [B*T*H] -> Reshape to [B, T, H]
    flat_states = states.reshape(-1, states.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])
    
    log_p = policy.get_log_p_select(flat_states, flat_actions, heads)
    log_p = log_p.view(B, T, H)
    
    # 2. Extract k-NN Radius and calculate Volumetric Reward
    # Using the first 2 dimensions (d=2) as requested
    dist_k = distances[:, k].view(B, T, H).to(device)

    # entropy reward
    intrinsic_reward = d * torch.log(dist_k + 1e-8)
    intrinsic_reward = intrinsic_reward.to(device)


    reward_mean = intrinsic_reward.mean()
    reward_std = intrinsic_reward.std()
    normalized_reward = (intrinsic_reward - reward_mean) / (reward_std + 1e-8)
    
    # reward-to-go
    # R = torch.flip(
    #     torch.cumsum(torch.flip(r, dims=[1]), dim=1),
    #     dims=[1]
    # )


    # REINFORCE
    loss = -torch.mean(torch.sum(
        log_p * normalized_reward.detach(),
        dim=1
    ))

    return loss

def compute_knn_policy_loss_chunk(policy, states, actions, heads, distances, indices, k, chunk_starts, chunk_length):
    # Get raw probability of the chunk
    # No log here as per your note
    prob = torch.exp(policy.get_log_p_select(states, actions, heads))
    
    # Calculate k-NN reward for the boundary states
    # We use the distance to the k-th neighbor as a proxy for entropy
    # Ensure distances are indexed to match the boundary_states
    boundary_indices = [] # Logic to map flattened boundary states to k-NN distance indices
    
    # Simple version: If distances were computed for all states, 
    # extract only the distances at chunk boundaries
    dist_k = distances[:, k].view(states.shape[0], -1, states.shape[2]) 
    boundary_dist = dist_k[:, chunk_starts // 1, :].reshape(-1) # Simplified mapping
    
    knn_reward = torch.log(boundary_dist + 1e-6)
    
    # Direction: grad(Pi) * Reward
    # Detach reward to ensure gradient only flows through policy probability
    loss = -torch.mean(prob * knn_reward.detach())
    
    return loss

def multihead_kl_divergence(means, log_stds):
    """
    means:    [B, H, A]
    log_stds: [B, H, A]
    Returns scalar diversity bonus (higher = more diverse heads)
    """
    B, H, A = means.shape
    stds = log_stds.exp()

    kl_total = 0.0
    count = 0

    for i in range(H):
        for j in range(H):
            if i == j:
                continue

            mu_i = means[:, i]
            mu_j = means[:, j]
            std_i = stds[:, i]
            std_j = stds[:, j]

            # KL(N_i || N_j)
            kl = (
                torch.log(std_j / std_i)
                + (std_i**2 + (mu_i - mu_j)**2) / (2.0 * std_j**2)
                - 0.5
            ).sum(dim=-1)  # sum over action dims

            kl_total += kl.mean()
            count += 1

    return kl_total / max(count, 1)

def head_state_mutual_information(states, env2agent, state_dims=[0,1], num_bins=32):
    device = states.device
    B, T1, E, D = states.shape
    H = int(env2agent.max().item()) + 1

    flat = states.reshape(-1, E, D)[:, :, state_dims]

    mins = flat.min(dim=0, keepdim=True)[0]
    maxs = flat.max(dim=0, keepdim=True)[0]
    norm = (flat - mins) / (maxs - mins + 1e-8)

    s_all = norm.reshape(-1, len(state_dims))
    head_ids = env2agent.repeat(B * T1)

    bins = torch.clamp((s_all * num_bins).long(), 0, num_bins - 1)

    joint = torch.zeros((H, num_bins, num_bins), device=device)

    joint.index_put_(
        (head_ids, bins[:, 0], bins[:, 1]),
        torch.ones(len(bins), device=device),
        accumulate=True
    )

    joint = joint / joint.sum().clamp_min(1e-12)

    p_head = joint.sum(dim=(1,2), keepdim=True)
    p_state = joint.sum(dim=0, keepdim=True)

    # Mask zero entries
    mask = joint > 0

    mi = joint[mask] * (
        torch.log(joint[mask])
        - torch.log(p_head.expand_as(joint)[mask])
        - torch.log(p_state.expand_as(joint)[mask])
    )

    return mi.sum()

def head_cosine_diversity(means):
    """
    means: [B, H, A]
    Penalizes similarity between heads.
    """
    m = F.normalize(means, dim=-1)  # normalize per head
    sim = torch.einsum("bha,bja->bhj", m, m)  # [B,H,H]

    H = sim.shape[-1]
    mask = ~torch.eye(H, device=sim.device, dtype=torch.bool)

    # penalize similarity across different heads only
    return sim[:, mask].mean()

def _symmetrize(A):
    return 0.5 * (A + A.transpose(-1, -2))

def _safe_cholesky_batched(
    Sigma: torch.Tensor,
    *,
    jitter0: float = 1e-6,
    max_tries: int = 6,
    shrink_to_I: bool = True,
    shrink_lambda: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust batched Cholesky with progressive jitter and optional shrinkage-to-identity.

    Args:
        Sigma: [B, D, D] SPD/PSD matrices
        jitter0: initial diagonal jitter
        max_tries: jitter multiplications by 10 up to this many attempts
        shrink_to_I: after jitter attempts, try shrinkage toward scaled identity
        shrink_lambda: shrinkage coefficient in [0,1]

    Returns:
        chol:     [B, D, D] (zeros where failed)
        ok_mask:  [B] bool, True where Cholesky succeeded
        Sigma_adj:[B, D, D] adjusted covariances actually factorized
    """
    B, D, _ = Sigma.shape
    Sigma_adj = _symmetrize(Sigma).clone()
    chol = torch.zeros_like(Sigma_adj)
    ok_mask = torch.zeros(B, dtype=torch.bool, device=Sigma.device)

    eye = torch.eye(D, device=Sigma.device, dtype=Sigma.dtype).expand(B, D, D)
    fail_mask = ~ok_mask
    jitter = jitter0

    # Progressive jitter
    for _ in range(max_tries):
        if not fail_mask.any():
            break
        ids = fail_mask.nonzero(as_tuple=True)[0]
        S_try = Sigma_adj[ids] + jitter * eye[ids]
        L, info = torch.linalg.cholesky_ex(S_try)
        success = (info == 0)
        if success.any():
            good = ids[success]
            chol[good] = L[success]
            ok_mask[good] = True
            fail_mask[good] = False
        jitter *= 10.0

    # Shrinkage toward identity (scaled by avg variance)
    if shrink_to_I and fail_mask.any():
        ids = fail_mask.nonzero(as_tuple=True)[0]
        S = Sigma_adj[ids]
        avg_var = S.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # [k,1]
        I_scaled = torch.eye(D, device=S.device, dtype=S.dtype).unsqueeze(0) * avg_var.unsqueeze(-1)
        S_shrunk = _symmetrize((1.0 - shrink_lambda) * S + shrink_lambda * I_scaled)
        L, info = torch.linalg.cholesky_ex(S_shrunk)
        success = (info == 0)
        if success.any():
            good = ids[success]
            chol[good] = L[success]
            ok_mask[good] = True
            Sigma_adj[good] = S_shrunk[success]

    return chol, ok_mask, Sigma_adj

def calculate_vector_kl_by_agent(
    states: torch.Tensor,
    env2agent,
    num_agents: int | None = None,
    eps: float = 1e-6,
    *,
    jitter0: float = 1e-6,
    max_tries: int = 6,
    shrink_lambda: float = 0.05,
    diagonal_fallback: bool = True,
) -> torch.Tensor:
    """
    KL for each agent's Gaussian N(mu_a, Σ_a) vs pooled others N(mu_-a, Σ_-a),
    computed from vectorized sufficient statistics and robust PD handling.

    Args:
        states:      [num_traj, T+1, num_envs, state_dim]
        env2agent:   1D LongTensor/List len=num_envs with agent ids in [0..A-1]
        num_agents:  optional; inferred from env2agent if None
        eps:         tiny diagonal regularizer
        jitter0:     initial jitter for robust Cholesky
        max_tries:   jitter attempts (×10 each time)
        shrink_lambda: shrinkage toward identity applied after jitter if needed
        diagonal_fallback: if Cholesky still fails, use diagonal KL for those batches

    Returns:
        kl: [num_agents] KL(N_a || N_-a); NaN where stats are ill-defined
    """
    device = states.device
    dtype = states.dtype
    _, _, E, D = states.shape

    # normalize env2agent
    if not isinstance(env2agent, torch.Tensor):
        env2agent = torch.as_tensor(env2agent, dtype=torch.long, device=device)
    else:
        env2agent = env2agent.to(device=device, dtype=torch.long)
    assert env2agent.shape == (E,), "env2agent must have shape [num_envs]"
    A = (int(env2agent.max().item()) + 1) if num_agents is None else int(num_agents)
    assert torch.all((0 <= env2agent) & (env2agent < A)), "env2agent has out-of-range ids"

    # samples per env (uniform)
    N_env = states.shape[0] * states.shape[1]
    if N_env <= 1 or E <= 1 or A <= 1:
        return torch.full((A,), float("nan"), device=device, dtype=dtype)

    # reshape to [N_env, E, D]
    X = states.reshape(N_env, E, D)

    # per-env sufficient stats
    S_env = X.sum(dim=0)                              # [E, D]
    XE = X.transpose(0, 1)                            # [E, N_env, D]
    Q_env = XE.transpose(1, 2) @ XE                   # [E, D, D] = Σ x x^T

    # aggregate per-agent via scatter_add
    S_agent = torch.zeros((A, D), device=device, dtype=dtype)
    Q_agent = torch.zeros((A, D, D), device=device, dtype=dtype)

    idx_SD = env2agent.view(E, 1).expand(E, D)
    
    #for e in range(E):
    #   a = env2agent[e]
    #   S_agent[a] += S_env[e]
    # GPU IMPLEMENTATION

    S_agent.scatter_add_(0, idx_SD, S_env)

    idx_Q = env2agent.view(E, 1, 1).expand(E, D, D)
    Q_agent.scatter_add_(0, idx_Q, Q_env)

    # counts
    env_count = torch.bincount(env2agent, minlength=A).to(device=device)
    N_agent = env_count * N_env  # [A]

    # means and unbiased covariances per agent
    mu_agent = torch.zeros((A, D), device=device, dtype=dtype)
    has_samples = N_agent > 0
    mu_agent[has_samples] = S_agent[has_samples] / N_agent[has_samples].unsqueeze(-1)

    mu_outer = mu_agent.unsqueeze(-1) @ mu_agent.unsqueeze(-2)           # [A, D, D]
    denom = (N_agent - 1).clamp(min=1).view(A, 1, 1)
    Sigma_agent = (Q_agent - N_agent.view(A, 1, 1) * mu_outer) / denom   # [A, D, D]

    # pooled "others"
    S_all = S_agent.sum(dim=0)                # [D]
    Q_all = Q_agent.sum(dim=0)                # [D, D]
    N_all = N_env * E

    S_others = S_all.unsqueeze(0) - S_agent   # [A, D]
    Q_others = Q_all.unsqueeze(0) - Q_agent   # [A, D, D]
    N_others = N_all - N_agent                # [A]

    mu_others = torch.zeros((A, D), device=device, dtype=dtype)
    has_other = N_others > 0
    mu_others[has_other] = S_others[has_other] / N_others[has_other].unsqueeze(-1)

    mu_others_outer = mu_others.unsqueeze(-1) @ mu_others.unsqueeze(-2)  # [A, D, D]
    denom_o = (N_others - 1).clamp(min=1).view(A, 1, 1)
    Sigma_others = (Q_others - N_others.view(A, 1, 1) * mu_others_outer) / denom_o

    # regularize + symmetrize
    eye = torch.eye(D, device=device, dtype=dtype).expand(A, D, D)
    Sigma_agent  = _symmetrize(Sigma_agent  + eps * eye)
    Sigma_others = _symmetrize(Sigma_others + eps * eye)

    kl = torch.full((A,), float("nan"), device=device, dtype=dtype)

    # need at least 2 samples for unbiased covariance on both sides
    valid_cov = (N_agent > 1) & (N_others > 1)
    if not valid_cov.any():
        return kl

    aidx = valid_cov.nonzero(as_tuple=True)[0]
    S_self   = Sigma_agent[aidx]
    S_other  = Sigma_others[aidx]
    mu_self  = mu_agent[aidx]
    mu_other = mu_others[aidx]

    # robust Cholesky
    chol_o, ok_o, S_other_adj = _safe_cholesky_batched(
        S_other, jitter0=jitter0, max_tries=max_tries, shrink_to_I=True, shrink_lambda=shrink_lambda
    )
    chol_s, ok_s, S_self_adj = _safe_cholesky_batched(
        S_self,  jitter0=jitter0, max_tries=max_tries, shrink_to_I=True, shrink_lambda=shrink_lambda
    )

    ok = ok_o & ok_s
    # exact KL where both decompositions succeeded
    if ok.any():
        ids = ok.nonzero(as_tuple=True)[0]
        L_o = chol_o[ids]
        L_s = chol_s[ids]
        So  = S_other_adj[ids]
        Ss  = S_self_adj[ids]
        muo = mu_other[ids]
        mus = mu_self[ids]

        # logdet via Cholesky
        ld_o = 2.0 * torch.log(torch.diagonal(L_o, dim1=-2, dim2=-1)).sum(-1)
        ld_s = 2.0 * torch.log(torch.diagonal(L_s, dim1=-2, dim2=-1)).sum(-1)

        # trace(So^{-1} Ss)
        A_mat = torch.cholesky_solve(Ss, L_o)                             # [K, D, D]
        tr_term = A_mat.diagonal(dim1=-2, dim2=-1).sum(-1)

        # (mu_o - mu_s)^T So^{-1} (mu_o - mu_s)
        diff = (muo - mus).unsqueeze(-1)                                   # [K, D, 1]
        v = torch.cholesky_solve(diff, L_o)                                # [K, D, 1]
        mahal = (diff.squeeze(-1) * v.squeeze(-1)).sum(-1)

        Ddim = D
        kl_vals = 0.5 * (tr_term + mahal - Ddim + (ld_o - ld_s))
        kl[aidx[ids]] = kl_vals.clamp_min(0.0)

    # diagonal fallback if still failing
    if diagonal_fallback:
        rem = (~ok).nonzero(as_tuple=True)[0]
        if rem.numel() > 0:
            So_diag = torch.clamp(torch.diagonal(S_other_adj[rem], dim1=-2, dim2=-1), min=1e-12)
            Ss_diag = torch.clamp(torch.diagonal(S_self_adj[rem],  dim1=-2, dim2=-1), min=1e-12)
            muo = mu_other[rem]
            mus = mu_self[rem]
            # KL for diagonal Gaussians
            tr = (Ss_diag / So_diag).sum(-1)
            diff2 = ((muo - mus) ** 2 / So_diag).sum(-1)
            logdet = torch.log(So_diag).sum(-1) - torch.log(Ss_diag).sum(-1)
            kl_diag = 0.5 * (tr + diff2 - D + logdet)
            kl[aidx[rem]] = kl_diag.clamp_min(0.0)

    return kl

def knn_kl_agents_subspace_scipy(
    states,                 # ndarray, shape [num_traj, T+1, num_envs, D]
    env2agent,              # 1D array-like, len=num_envs, values in [0..A-1]
    k=5,
    *,
    subspace_dims=None,     # list/array of dims to KEEP; if None -> keep all
    state_selector=None,    # alternative: length-D mask; keep dims where not np.nan
    standardize=False,      # z-score on the selected dims
    n_max=None,        # optional: cap #P samples per agent (subsample for speed) -- fixed 11 T * 1000 Envs
    m_max=None,        # optional: cap #Q samples per agent (subsample for speed) -- fixed 11 T * 1000 Envs
    rng=None,               # numpy Generator or seed for reproducible subsampling
    eps=1e-12,
    leafsize=40,            # cKDTree leaf size (tune for large data)
    workers=-1,             # threads for cKDTree.query (SciPy >= 1.6)
):
    """
    Nonparametric k-NN KL estimator per agent on a selected subspace using SciPy cKDTree.
    Estimates KL(P_a || P_-a) from samples via the Wang–Kulkarni–Verdú k-NN formula:

        KL ≈ (d' / n) * sum_i log( ν_k(x_i) / ρ_k(x_i) ) + log(m / (n - 1))

    where:
      - P_a: agent a's samples (n points) in the chosen subspace of dimension d'
      - P_-a: pooled others (m points)
      - ρ_k(x_i): distance to k-th NN in P_a \ {x_i} (self-excluded via k+1 trick)
      - ν_k(x_i): distance to k-th NN in P_-a

    Parameters
    ----------
    states : np.ndarray
        Shape [num_traj, T+1, num_envs, D].
    env2agent : array-like
        Length num_envs; maps environment -> agent id in [0 .. A-1].
    k : int
        k-th neighbor for the estimator (typically 3..10).
    subspace_dims : list[int] or np.ndarray, optional
        Exact indices of dims to keep.
    state_selector : np.ndarray, optional
        Alternative dimension selector: length-D array; keep dims where not np.nan.
    standardize : bool
        If True, z-score the selected dims globally before building trees.
    m_max : int, optional
        If given and m >> n, subsample up to m_max others per agent for speed.
    rng : np.random.Generator or int, optional
        RNG or seed for reproducible subsampling of Q when m_max is set.
    eps : float
        Small constant to stabilize logs/divisions.
    leafsize : int
        cKDTree leaf size.
    workers : int
        Threads for cKDTree.query. Use -1 for "all cores" (SciPy >= 1.6).

    Returns
    -------
    kl_per_agent : np.ndarray
        Length A; NaN where insufficient samples.

    Notes
    -----
    - Correctly aligns rows of the flattened state matrix to env ids using
      env_ids = np.tile(np.arange(E), N_env), matching the C-order reshape.
    - Returns NaN for an agent if n <= k or n <= 1 or m < k after subspace selection
      (and optional subsampling).
    """
    # ---------- inputs & shapes ----------
    X = np.asarray(states)
    assert X.ndim == 4, "states must be [num_traj, T+1, num_envs, D]"
    num_traj, T1, E, D = X.shape

    env2agent = np.asarray(env2agent, dtype=np.int64)
    assert env2agent.shape == (E,), "env2agent must have length num_envs"
    A = int(env2agent.max()) + 1
    if not np.all((0 <= env2agent) & (env2agent < A)):
        raise ValueError("env2agent has out-of-range agent ids")

    # ---------- choose subspace dims (D -> d') ----------
    if subspace_dims is not None:
        dims = np.asarray(subspace_dims, dtype=np.int64)
    elif state_selector is not None:
        sel = np.asarray(state_selector)
        if sel.shape != (D,):
            raise ValueError("state_selector must have shape (D,)")
        dims = np.nonzero(~np.isnan(sel))[0].astype(np.int64)
    else:
        dims = np.arange(D, dtype=np.int64)

    d_eff = int(dims.size)
    if d_eff == 0:
        return np.full((A,), np.nan, dtype=float)

    # ---------- flatten (time, traj) and project ----------
    N_env = num_traj * T1
    X_2d = X.reshape(N_env, E, D)  # [N_env, E, D]
    # Ensure C-contiguous before ravel; then project to subspace
    X_all = np.ascontiguousarray(X_2d).reshape(N_env * E, D)[:, dims].astype(np.float64, copy=False)  # [N_tot, d_eff]

    # Map each row in X_all to an env id, then to an agent id
    # Row order after reshape is (n=0,e=0..E-1), (n=1,e=0..E-1), ...
    env_ids = np.tile(np.arange(E, dtype=np.int64), N_env)    # CORRECT alignment
    agent_ids_per_row = env2agent[env_ids]                    # [N_tot]

    # ---------- global standardization on selected dims ----------
    if standardize:
        mean = X_all.mean(axis=0, keepdims=True)
        std = X_all.std(axis=0, ddof=0, keepdims=True)
        std = np.maximum(std, 1e-8)
        X_all = (X_all - mean) / std

    # ---------- RNG for optional subsampling ----------
    if m_max is not None and not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # ---------- output ----------
    kl = np.full((A,), np.nan, dtype=np.float64)

    # ---------- per-agent KL ----------
    for a in range(A):
        maskP = (agent_ids_per_row == a)
        maskQ = ~maskP
        n = int(maskP.sum())
        m = int(maskQ.sum())

        # Need at least k+1 in P (exclude self via k+1 trick) and k in Q
        if n <= k or n <= 1 or m < k:
            continue

        P = X_all[maskP, :]  # [n, d_eff]
        Q = X_all[maskQ, :]  # [m, d_eff]

        # Optional: subsample Q or P if huge
        if (m_max is not None) and (m > m_max):
            idx = rng.choice(m, size=m_max, replace=False)
            Q = Q[idx, :]
            m = Q.shape[0]
            if m < k:
                continue
        if (n_max is not None) and (n > n_max):
            idx = rng.choice(n, size=n_max, replace=False)
            P = P[idx, :]
            n = P.shape[0]
            if n < k:
                continue

        # Build trees
        tree_P = cKDTree(P, leafsize=leafsize)
        tree_Q = cKDTree(Q, leafsize=leafsize)

        # Distances to k-th NN within P (exclude self):
        # query k+1 neighbors (first is self at 0), take index k
        dists_P, _ = tree_P.query(P, k=k+1, workers=workers)  # [n, k+1] or [n] if k==0 (we don't allow)
        # ensure 2D
        if dists_P.ndim == 1:
            # this would imply k==0 requested, which we don't support
            raise ValueError("k must be >= 1")
        rho_k = dists_P[:, k]  # [n]

        # Distances to k-th NN in Q
        dists_Q, _ = tree_Q.query(P, k=k, workers=workers)    # [n, k] or [n] if k==1
        if k == 1 and dists_Q.ndim == 1:
            dists_Q = dists_Q[:, None]
        nu_k = dists_Q[:, -1]  # [n]

        # KL estimator
        term = np.log((nu_k + eps) / (rho_k + eps)).sum()
        kl_a = (d_eff / n) * term + np.log(m / (n - 1))
        kl[a] = kl_a

    return kl

def get_model_memory(model, optimizer=None):
    # Parameter memory
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())

    # Buffer memory (running stats, masks, etc.)
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers())

    # Gradient memory
    grad_mem = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters()
        if p.grad is not None
    )

    # Optimizer state memory
    opt_mem = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    opt_mem += v.numel() * v.element_size()

    total = param_mem + buffer_mem + grad_mem + opt_mem

    return {
        "parameters_MB": param_mem / 1024**2,
        "buffers_MB":    buffer_mem / 1024**2,
        "gradients_MB":  grad_mem / 1024**2,
        "optimizer_MB":  opt_mem / 1024**2,
        "total_MB":      total / 1024**2,
    }

def record_train_step(policy,loss,entropy_loss,current_lr,entropy,epoch,writer):
    
    writer.add_scalar(f'Loss/Agent', loss.item(), epoch)
    writer.add_scalar(f'Loss/Entropy_Component', entropy_loss.item(), epoch)

    writer.add_scalar(f'Learning_Rate/Agent', current_lr, epoch)

    try:
        if epoch % 10 == 0:  # Less frequent logging
            for name, param in policy.named_parameters():
                if param.grad is not None and param.grad.numel() > 0:
                    if param.grad.isfinite().all()== False:
                        print(f"Non-finite gradient detected in parameter {name}; skipping logging for this parameter.")
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    except Exception as e:
        print(f"Error logging gradients for Agent: {e}")

    #total_loss = torch.stack(losses).sum()
    writer.add_scalar('Loss/Total', loss.item(), epoch)
    writer.add_scalar('Entropy',  entropy.mean().item(), epoch)

    print(f"Loss: {loss.item()}, Entropy: {entropy.mean().item()}")
    return

def record_actions_step(num_agents, env2agent, actions, states, epoch, writer):
    for agent_id in range(num_agents):
        # select env indices that belong to this agent
        env_mask = (env2agent == agent_id).nonzero(as_tuple=True)[0]  # [num_envs_assigned]

        # actions: [num_traj, T, num_envs, action_dim]
        agent_actions = actions[:, :, env_mask, :]  # [num_traj, T, num_envs_assigned, action_dim]
        agent_states  = states[:, :, env_mask, :]   # [num_traj, T+1, num_envs_assigned, state_dim]

        # flatten over trajectories, time, and envs
        flat_actions = agent_actions.reshape(-1, agent_actions.shape[-1])  # [N, action_dim]
        flat_states  = agent_states.reshape(-1, agent_states.shape[-1])    # [N, state_dim]

        # per-action-dim histograms
        for action_dim in range(flat_actions.shape[-1]):
            writer.add_histogram(
                f'Action_Distributions/Agent_{agent_id}/Action_{action_dim}',
                flat_actions[:, action_dim],
                epoch
            )

        # state histogram (just first 2 dims as before)
        writer.add_histogram(
            f'State_Distributions/Agent_{agent_id}/States',
            flat_states[:, :2],
            epoch
        )
    return