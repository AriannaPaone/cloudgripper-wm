from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import math
from matplotlib.cm import get_cmap


def knn_entropy_estimation_scipy(states,state_filter,real_traj_lengths, k=500):
    """
    Compute entropy estimate using kNN with scipy, vectorized.

    Arguments:
    - states: NumPy array of shape [batch_size, N, d], where:
        - batch_size is the number of independent sets of samples.
        - N is the number of samples per set.
        - d is the dimensionality of each sample.
    - k: Number of neighbors to consider (default: 500).

    Returns:
    - entropies: NumPy array of shape [batch_size] containing entropy estimates for each batch.
    """
    # Detach once and move to CPU
    states = states.detach().cpu()

    # Optional: filter states before converting to numpy
    if state_filter is not None:
        states = states[:, :, :, state_filter]

    states = states.permute(2, 0, 1, 3)  # Shape: [batch_size, D, N, H]
    num_envs, num_trajs, _, feat_dim = states.shape

    # Convert to numpy float32 once
    states = states.numpy().astype(np.float32)
    # Pre-allocate list of slices
    sliced_states = []
    for env in range(num_envs):
        # Gather slices for this env into a list
        env_slices = [states[env, j, 1:real_traj_lengths[env,j]+1, :] for j in range(num_trajs)]
        sliced_states.extend(env_slices)  # Flat list of tensors

    states = np.vstack(sliced_states)  # Shape: [H, N, D]
    # If needed, permute and reshape as per your final goal

    N, d = states.shape
    entropies = 0.0
    eps = 1e-7
    B = np.log(k) - scipy.special.digamma(k)
    G = scipy.special.gamma(d/2 + 1)

    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean' ,algorithm='auto')
    nbrs.fit(states)
    distances, _ = nbrs.kneighbors(states)
    distances = torch.tensor(distances, dtype=torch.float32)  # Exclude the first column (self-distance)
    # compute volume for each particle
    volumes = (torch.pow(distances[:, k], d) * torch.pow(torch.tensor(np.pi), d/2)) / G
    # compute entropy
    entropies= - (1/N) * torch.sum( torch.log(((k/N) / (volumes + eps)) + eps)) + B

    return entropies

def knn_entropy_estimation_torch(states, state_filter,real_traj_lengths, k=500, eps=1e-7, dim_weights=None):
    """
    Compute entropy estimate using kNN entirely on GPU with torch.kthvalue.

    Arguments:
    - states: torch.Tensor [batch_size, N, d] or similar
    - state_filter: indices to filter features (or None)
    - real_traj_lengths: tensor [batch_size, N] giving valid trajectory lengths
    - k: number of neighbors

    - dim_weights: optional weights for each dimension to give different importance to features
    """
    device = states.device

    # Optional feature filter
    if state_filter is not None:
        states = states[..., state_filter]

    if dim_weights is not None:
        states = states * torch.as_tensor(
            dim_weights, dtype=states.dtype, device=states.device) # filter dimensions by weights

    # Permute same as original code [num_envs, num_trajs, H, feat_dim]
    states = states.permute(2, 0, 1, 3)
    num_envs, num_trajs, _, feat_dim = states.shape

    # Flatten states like your original code
    # sliced_states = []
    # for env in range(num_envs):
    #     env_slices = [
    #         [states[env, j, 1:real_traj_lengths[env,j]+1, :] for j in range(num_trajs)]
    #         for j in range(num_trajs)
    #     ]
    #     sliced_states.extend(env_slices)
    # states = torch.vstack(sliced_states).to(device)  # shape [N, d]


    mask = torch.zeros(states.shape[0], states.shape[1], states.shape[2], dtype=torch.bool, device=device)
    for env in range(num_envs):
        for j in range(num_trajs):
            mask[env, j, 1:real_traj_lengths[j,env]+1] = True
            #mask[env, j, 1:real_traj_lengths[env,j]+1] = True
    states = states[mask]  # directly gives you [N, d]

    N, d = states.shape

    # # ---- compute pairwise distances on GPU ----

    #subsample N if too large
    if N > 60000:
        indices = torch.randperm(N)[:60000]
        states = states[indices]
        N = states.shape[0]

    distances = torch.cdist(states, states, p=2)  # Euclidean norm

    # # ---- kthvalue to get distance to k-th nearest neighbor ----
    # # kthvalue sorts per row internally but faster than full sort
    # # Exclude diagonal (self-distance = 0)
    # # Replace diagonal with large number so it isn't chosen
    distances[range(N), range(N)] = float('inf')

    kth_distances, _ = torch.kthvalue(distances, k, dim=1)  # [N]

    # # ----- Alternative ------  
    # #CHUNCHED VERSION
    #kth_distances = compute_kth_distances_chunked(states, k=k, chunk_size=8192)  # [N]

    # ---- entropy calculation ----
    B = math.log(k) - torch.digamma(torch.tensor(k, device=device))
    G = torch.lgamma(torch.tensor(d/2 + 1.0, device=device)).exp()  # gamma(d/2+1)
    volumes = (kth_distances ** d) * (math.pi ** (d/2)) / G

    entropy = - (1.0/N) * torch.sum(torch.log((k/N) / (volumes + eps) + eps)) + B
    return entropy


def compute_kth_distances_chunked(x, k, chunk_size=1024):
    """
    Computes k-th nearest neighbor distances in chunks to avoid OOM.

    Args:
        x (Tensor): [N, D]
        k (int): Number of neighbors (k-th smallest)
        chunk_size (int): Number of rows to process at a time

    Returns:
        kth_distances: Tensor [N] with distance to k-th nearest neighbor
    """
    N, D = x.shape
    kth_distances = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x_chunk = x[start:end]  # shape [chunk_size, D]

        # Compute distances to all other points
        dist = torch.cdist(x_chunk, x, p=2)  # [chunk_size, N]

        # Set diagonal (self-distance) to large number
        for i in range(start, end):
            dist[i - start, i] = float('inf')

        # Get k-th smallest distance for each row in the chunk
        kth_dist, _ = torch.kthvalue(dist, k, dim=1)
        kth_distances.append(kth_dist)

    return torch.cat(kth_distances, dim=0)

def compute_entropy_single_batch(states_b, k, d, N, B, G, eps):
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto')
    nbrs.fit(states_b)
    distances, _ = nbrs.kneighbors(states_b)

    distances = torch.tensor(distances, dtype=torch.float32)  # shape: [N, k+1]
    volumes = (torch.pow(distances[:, k], d) * torch.pow(torch.tensor(np.pi), d/2)) / G
    entropy = - (1 / N) * torch.sum(torch.log(((k / N) / (volumes + eps)) + eps)) + B
    return entropy.item()


def knn_entropy_estimation_scipy_parallel(states, k=500, n_jobs=-1):
    """
    Parallel kNN entropy estimation over batches.

    Arguments:
    - states: torch.Tensor of shape [batch_size, N, d]
    - k: number of neighbors
    - n_jobs: number of parallel jobs (default: all CPUs)

    Returns:
    - entropies: NumPy array of shape [batch_size]
    """
    states = states.detach().cpu().numpy().astype(np.float32)
    batch_size, N, d = states.shape

    eps = 0
    B = np.log(k) - scipy.special.digamma(k)
    G = scipy.special.gamma(d / 2 + 1)

    entropies = Parallel(n_jobs=n_jobs)(
        delayed(compute_entropy_single_batch)(states[b], k, d, N, B, G, eps) for b in range(batch_size)
    )

    return np.array(entropies, dtype=np.float32)

def get_heatmap(states, discretizer):
    """
    Builds a log-probability state visitation heatmap by running
    the policy in env. The heatmap is built using the provided
    discretizer.

    Args:
        STATES: [NUM_TRAJECTORY,TRAJECTORY_LENGTH, STATE_DIM]

    """
    NUM_TRAJECTORY = states.shape[0]
    TRAJECTORY_LENGTH = states.shape[1]

    average_state_dist = discretizer.get_empty_mat()
    average_entropy = 0

    for _ in range(NUM_TRAJECTORY):
        s = states[_]
        state_dist = discretizer.get_empty_mat()

        for t in range(TRAJECTORY_LENGTH):
            state_dist[discretizer.discretize(s[t])] += 1

        state_dist /= t+1
        average_state_dist += state_dist
        average_entropy += scipy.stats.entropy(state_dist.cpu().ravel().numpy())

    average_state_dist /= NUM_TRAJECTORY
    average_entropy /= NUM_TRAJECTORY

    #plt.close()
    image_fig = plt.figure()

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('X')
    plt.ylabel('Y')

    avg_dist_np = average_state_dist.cpu().numpy()


    if len(average_state_dist.shape) == 2:
        log_p = np.ma.log(avg_dist_np)
        log_p_ravel = log_p.ravel()
        min_log_p_ravel = np.min(log_p_ravel)
        second_min_log_p_ravel = np.min(log_p_ravel[log_p_ravel != min_log_p_ravel])
        log_p_ravel[np.argmin(log_p_ravel)] = second_min_log_p_ravel
        plt.imshow(log_p.filled(min_log_p_ravel))
    else:
        plt.bar([i for i in range(discretizer.bins_sizes[0])], avg_dist_np)

    return average_state_dist, average_entropy, image_fig


def get_agent_heatmaps(states, discretizer, env2agent, num_agents):
    """
    Builds per-agent log-probability state visitation heatmaps.
    
    Args:
        states: [num_traj, T, num_envs, state_dim]
        discretizer: object with .get_empty_mat() and .discretize()
        env2agent: [num_envs] LongTensor mapping env -> agent head
        num_agents: int

    Returns:
        agent_dists: list of [bins_x, bins_y] tensors, one per agent
        agent_entropies: list of floats
        image_fig: matplotlib figure with colored overlays
    """
    num_traj, T, num_envs, _ = states.shape

    agent_dists = [discretizer.get_empty_mat() for _ in range(num_agents)]
    agent_entropies = [0.0 for _ in range(num_agents)]

    # Collect per-agent visitation distributions
    for traj in range(num_traj):
        for env in range(num_envs):
            agent_id = int(env2agent[env].item())
            s = states[traj, :, env]  # [T, state_dim]
            state_dist = discretizer.get_empty_mat()
            for t in range(T):
                state_dist[discretizer.discretize(s[t])] += 1
            state_dist /= T
            agent_dists[agent_id] += state_dist
            agent_entropies[agent_id] += scipy.stats.entropy(state_dist.cpu().ravel().numpy())

    # Normalize
    for i in range(num_agents):
        agent_dists[i] /= num_traj
        agent_entropies[i] /= num_traj

    # === Plot ===
    cmap = plt.cm.get_cmap("tab10") if num_agents <= 10 else plt.cm.get_cmap("turbo")
    colors = [cmap(i / max(num_agents - 1, 1)) for i in range(num_agents)]
    image_fig, ax = plt.subplots()
    white_bg = (1.0, 1.0, 1.0)
    image_fig.patch.set_facecolor(white_bg)
    ax.set_facecolor(white_bg)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(num_agents):
        dist_np = agent_dists[i].cpu().numpy()
        if dist_np.ndim == 2:
            base_color = np.array(colors[i][:3])  # ensure RGB
            # Normalize visit intensity to [0, 1] for alpha scaling
            heat = dist_np
            heat_min = np.nanmin(heat)
            heat_max = np.nanmax(heat)
            if np.isfinite(heat_max) and heat_max > heat_min:
                heat_norm = (heat - heat_min) / (heat_max - heat_min)
            else:
                heat_norm = np.zeros_like(heat)

            rgba = np.zeros((*heat_norm.shape, 4), dtype=np.float32)
            rgba[..., :3] = base_color[None, None, :]  # color per agent
            rgba[..., 3] = heat_norm     # transparency encodes intensity

            im = ax.imshow(rgba, origin='lower')
            im.set_label(f"Agent {i}")
        else:
            ax.bar(
                np.arange(discretizer.bins_sizes[0]),
                dist_np,
                color=colors[i],
                alpha=0.5,
                label=f"Agent {i}",
            )

    #ax.legend()

    return agent_dists, agent_entropies, image_fig


def get_heatmap_fast(states, discretizer):
    """
    Fast state visitation heatmap + average entropy (matches get_heatmap).

    Args:
        states: torch.Tensor [num_trajectories, trajectory_length, state_dim]
        discretizer: object with `bins_sizes` and `bins`

    Returns:
        average_state_dist: torch.Tensor of shape bins_sizes
        average_entropy: float
        image_fig: matplotlib figure
    """
    num_trajectories, trajectory_length, state_dim = states.shape
    device = states.device

    # Flatten states -> [N, D]  (N = num_trajectories * trajectory_length)
    flat_states = states.reshape(-1, state_dim)

    # --- Vectorized discretization ---

    if state_dim == 1:
        i0 = torch.bucketize(flat_states[:, 0], discretizer.bins[0].to(device))
        flat_indices = i0
        total_bins = discretizer.bins_sizes[0]

    elif state_dim == 2:
        i0 = torch.bucketize(flat_states[:, 0], discretizer.bins[0].to(device))
        i1 = torch.bucketize(flat_states[:, 1], discretizer.bins[1].to(device))
        flat_indices = i0 * discretizer.bins_sizes[1] + i1
        total_bins = discretizer.bins_sizes[0] * discretizer.bins_sizes[1]


    else:
        raise ValueError(f"get_heatmap_fast only supports 1D/2D states, got {state_dim}D")

    # Restore [num_trajectories, trajectory_length]
    flat_indices = flat_indices.view(num_trajectories, trajectory_length)

    # --- Count visitations per trajectory ---
    state_counts = torch.zeros((num_trajectories, total_bins), device=device)
    for i in range(num_trajectories):
        state_counts[i] = torch.bincount(flat_indices[i], minlength=total_bins)

    # Normalize each trajectory distribution
    state_dists = state_counts / trajectory_length  # [B, total_bins]

    # Average distribution across trajectories
    average_state_dist = state_dists.mean(0)

    # Reshape back to grid if 2D
    if state_dim == 2:
        average_state_dist = average_state_dist.view(discretizer.bins_sizes)

    # --- Entropy per trajectory ---
    entropies = [
        scipy.stats.entropy(state_dists[i].cpu().numpy())
        for i in range(num_trajectories)
    ]
    average_entropy = float(np.mean(entropies))

    # --- Plotting ---
    image_fig = plt.figure()
    plt.xticks([]); plt.yticks([])
    plt.xlabel('X'); plt.ylabel('Y')
    avg_dist_np = average_state_dist.cpu().numpy()

    if avg_dist_np.ndim == 2:
        log_p = np.ma.log(avg_dist_np)
        valid_log_p = log_p[log_p > -np.inf]
        if valid_log_p.size > 1:
            second_min = np.sort(valid_log_p)[1]
            log_p = log_p.filled(second_min)
        else:
            log_p = log_p.filled(-np.inf)
        plt.imshow(log_p)
    else:
        plt.bar(range(discretizer.bins_sizes[0]), avg_dist_np)

    return average_state_dist, average_entropy, image_fig




def get_heatmap_fast_colored(states, discretizer):
    """
    State visitation heatmap with distinct per-trajectory colors and soft overlap blending.

    Each trajectory is assigned a unique color. Overlapping visits are softly blended,
    while a grayscale density map in the background shows global visitation coverage.

    Args:
        states: torch.Tensor [num_trajectories, trajectory_length, state_dim]
        discretizer: object with attributes:
            - bins: list of torch tensors defining discretization bins per dimension
            - bins_sizes: list/tuple with number of bins per dimension

    Returns:
        average_state_dist: torch.Tensor of shape discretizer.bins_sizes
        average_entropy: float
        image_fig: matplotlib figure
    """

    num_trajectories, trajectory_length, state_dim = states.shape
    device = states.device

    # Flatten states -> [N, D]
    flat_states = states.reshape(-1, state_dim)

    # --- Vectorized discretization ---
    if state_dim == 1:
        i0 = torch.bucketize(flat_states[:, 0], discretizer.bins[0].to(device))
        flat_indices = i0
        total_bins = discretizer.bins_sizes[0]

    elif state_dim == 2:
        i0 = torch.bucketize(flat_states[:, 0], discretizer.bins[0].to(device))
        i1 = torch.bucketize(flat_states[:, 1], discretizer.bins[1].to(device))
        flat_indices = i0 * discretizer.bins_sizes[1] + i1
        total_bins = discretizer.bins_sizes[0] * discretizer.bins_sizes[1]

    else:
        raise ValueError(f"get_heatmap_fast_colored only supports 1D/2D states, got {state_dim}D")

    # Restore shape [num_trajectories, trajectory_length]
    flat_indices = flat_indices.view(num_trajectories, trajectory_length)

    # --- Count visitations per trajectory ---
    state_counts = torch.zeros((num_trajectories, total_bins), device=device)
    for i in range(num_trajectories):
        state_counts[i] = torch.bincount(flat_indices[i], minlength=total_bins)

    # Normalize each trajectory distribution
    state_dists = state_counts / trajectory_length
    average_state_dist = state_dists.mean(0)

    # Reshape to 2D grid if needed
    if state_dim == 2:
        average_state_dist = average_state_dist.view(discretizer.bins_sizes)

    # --- Compute entropy per trajectory ---
    entropies = [
        scipy.stats.entropy(state_dists[i].cpu().numpy())
        for i in range(num_trajectories)
    ]
    average_entropy = float(np.mean(entropies))

    # --- Plotting ---
    image_fig = plt.figure()
    plt.xticks([]); plt.yticks([])
    plt.xlabel('X'); plt.ylabel('Y')

    if state_dim == 2:
        avg_dist_np = average_state_dist.cpu().numpy()

        # Base grayscale heatmap (soft background)
        plt.imshow(np.log1p(avg_dist_np), cmap="gray", alpha=0.25)

        # Prepare color overlay
        color_map = np.zeros((*discretizer.bins_sizes, 3))  # RGB
        cmap = plt.get_cmap("hsv", num_trajectories)

        # Plot each trajectory with distinct color
        for i in range(num_trajectories):
            traj_states = states[i].cpu().numpy()
            x_idx = np.digitize(traj_states[:, 0], discretizer.bins[0].cpu().numpy()) - 1
            y_idx = np.digitize(traj_states[:, 1], discretizer.bins[1].cpu().numpy()) - 1

            x_idx = np.clip(x_idx, 0, discretizer.bins_sizes[0] - 1)
            y_idx = np.clip(y_idx, 0, discretizer.bins_sizes[1] - 1)

            color = np.array(cmap(i)[:3])  # unique color per trajectory

            for xi, yi in zip(x_idx, y_idx):
                # Soft blending (keeps overlap visible but not overmixed)
                color_map[xi, yi] = 0.7 * color_map[xi, yi] + 0.3 * color

        # Show colored overlay
        plt.imshow(color_map, alpha=0.9)

    else:
        # 1D case — color-coded lines for each trajectory
        cmap = plt.get_cmap("tab10", num_trajectories)
        for i in range(num_trajectories):
            plt.plot(states[i, :, 0].cpu(), np.ones(trajectory_length) * i, ".", color=cmap(i))
        plt.ylabel("Trajectory ID")
        plt.xlabel("Discretized State")

    return average_state_dist, average_entropy, image_fig


def get_heatmap_fast_xy(states, discretizer):
    """
    Fast state visitation heatmap + average entropy (matches get_heatmap).

    Args:
        states: torch.Tensor [num_trajectories, trajectory_length, state_dim]
        discretizer: object with `bins_sizes` and `bins`

    Returns:
        average_state_dist: torch.Tensor of shape bins_sizes
        average_entropy: float
        image_fig: matplotlib figure
    """
    num_trajectories, trajectory_length, state_dim = states.shape
    device = states.device

    # Flatten states -> [N, D]  (N = num_trajectories * trajectory_length)
    flat_states = states.reshape(-1, state_dim)

    # --- Vectorized discretization ---

    if state_dim == 1:
        i0 = torch.bucketize(flat_states[:, 0], discretizer.bins[0].to(device))
        flat_indices = i0
        total_bins = discretizer.bins_sizes[0]

    elif state_dim == 2:
        i0 = torch.bucketize(flat_states[:, 0], discretizer.bins[0].to(device))
        i1 = torch.bucketize(flat_states[:, 1], discretizer.bins[1].to(device))
        flat_indices = i0 * discretizer.bins_sizes[1] + i1
        total_bins = discretizer.bins_sizes[0] * discretizer.bins_sizes[1]


    else:
        raise ValueError(f"get_heatmap_fast only supports 1D/2D states, got {state_dim}D")

    # Restore [num_trajectories, trajectory_length]
    flat_indices = flat_indices.view(num_trajectories, trajectory_length)

    # --- Count visitations per trajectory ---
    state_counts = torch.zeros((num_trajectories, total_bins), device=device)
    for i in range(num_trajectories):
        state_counts[i] = torch.bincount(flat_indices[i], minlength=total_bins)

    # Normalize each trajectory distribution
    state_dists = state_counts / trajectory_length  # [B, total_bins]

    # Average distribution across trajectories
    average_state_dist = state_dists.mean(0)

    # Reshape back to grid if 2D
    if state_dim == 2:
        average_state_dist = average_state_dist.view(discretizer.bins_sizes)

    # --- Entropy per trajectory ---
    entropies = [
        scipy.stats.entropy(state_dists[i].cpu().numpy())
        for i in range(num_trajectories)
    ]
    average_entropy = float(np.mean(entropies))

    # --- Plotting ---
    image_fig = plt.figure()
    plt.xticks([]); plt.yticks([])
    plt.xlabel('X'); plt.ylabel('Y')
    avg_dist_np = average_state_dist.cpu().numpy()

    if avg_dist_np.ndim == 2:
        log_p = np.ma.log(avg_dist_np)
        valid_log_p = log_p[log_p > -np.inf]
        if valid_log_p.size > 1:
            second_min = np.sort(valid_log_p)[1]
            log_p = log_p.filled(second_min)
        else:
            log_p = log_p.filled(-np.inf)
        plt.imshow(log_p)
    else:
        plt.bar(range(discretizer.bins_sizes[0]), avg_dist_np)

    return average_state_dist, average_entropy, image_fig