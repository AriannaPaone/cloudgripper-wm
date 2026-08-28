import torch
import random
import numpy as np
import os
import sys
import time
from torch.utils.tensorboard import SummaryWriter
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


file_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one folder and then into "config"
CONFIG_PATH = os.path.join(file_dir, "..", "config")

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)# If using CUDA, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set the deterministic flag for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_writer(env_name = "Swimmer-v5",seed = 0 , pretrain = False,MaxEnt=False,goal_position = None,num_envs = 1,hidden_size = 256):

    # = "/isaac-lab/logs_new/MaxEnt/" if MaxEnt else "/isaac-lab/logs_new/GoalBased/"
    #prefix = prefix + "no_pretrain/" if not pretrain else prefix
    base = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/data"))
    prefix = os.path.join(base, "logs/MaxEnt/" if MaxEnt else "logs/GoalBased/")

    out_path = prefix + time.strftime("%Y%m%d-%H%M%S") + "_" + env_name + "_seed_" + str(seed) + "_pre_" + str(pretrain)  + "_envs_" + str(num_envs) + "_hidden_" + str(hidden_size) + "_goal_" + str(goal_position) + "/"
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    writer = SummaryWriter(out_path)
    return writer, out_path

def save_policy(model, exp_folder, num_env, step, seed):
    
    if not os.path.exists(os.path.join(exp_folder, "models")):
        os.makedirs(os.path.join(exp_folder, "models"), exist_ok=True)

    model_file = os.path.join(exp_folder, f"models/model_agent_{num_env}_step_{step}_seed_{seed}.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")
    return

def save_settings(exp_folder, name_env, seed, num_epochs, hidden_sizes, obs_dim, act_dim, num_envs, total_trajs):
    if not os.path.exists(os.path.join(exp_folder, "settings")):
        os.makedirs(os.path.join(exp_folder, "settings"), exist_ok=True)

    settings = {
        "env_name": name_env,
        "seed": seed,
        "num_epochs": num_epochs,
        "hidden_sizes": hidden_sizes,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "num_envs": num_envs,
        "total_trajs": total_trajs,
    }
    with open(os.path.join(exp_folder, "settings/config.json"), "w") as f:
        json.dump(settings, f, indent=4)
    return


def compute_trajectory_budget_from_kl_o(last_kl_divs, total_trajs, min_budget=1, writer=None, count=None):
    """
    Allocate per-agent trajectory budgets based on KL divergences, 
    ensuring each agent gets at least `min_budget` trajectories.
    
    Args:
        last_kl_divs: list or 1D tensor of KL divergences per agent
        total_trajs: total available trajectories (int)
        min_budget: minimum trajectories per agent (default=1)
        writer, count: optional TensorBoard logging
    
    Returns:
        trajectory_budget: list[int] of length num_agents
    """
    kl_tensor = torch.tensor(last_kl_divs, dtype=torch.float32)
    num_agents = len(kl_tensor)

    # Softmax over KLs
    softmax_kl = F.softmax(kl_tensor, dim=0)

    # Initial fractional allocation
    frac_alloc = softmax_kl * total_trajs
    trajectory_budget = frac_alloc.round().to(torch.int32)

    # Enforce minimum budget
    # If any agent has less than min_budget, raise it and subtract excess from others proportionally
    too_low = trajectory_budget < min_budget
    if too_low.any():
        deficit = int((min_budget - trajectory_budget[too_low]).sum().item())
        trajectory_budget[too_low] = min_budget

        if deficit > 0:
            # Reduce excess proportionally from agents with > min_budget
            eligible = trajectory_budget > min_budget
            if eligible.any():
                excess = trajectory_budget[eligible].float()
                excess_ratio = excess / excess.sum()
                reduce = torch.floor(excess_ratio * deficit).to(torch.int32)
                # ensure sum matches
                diff = deficit - reduce.sum().item()
                if diff > 0:
                    reduce[0] += diff
                trajectory_budget[eligible] -= reduce.clamp(max=trajectory_budget[eligible] - min_budget)

    # Fix rounding mismatch to make total match total_trajs
    diff = total_trajs - trajectory_budget.sum().item()
    if diff != 0:
        # Add/subtract 1 trajectory from random agents until total matches
        sign = 1 if diff > 0 else -1
        for _ in range(abs(diff)):
            idx = torch.randint(0, num_agents, (1,)).item()
            new_val = trajectory_budget[idx] + sign
            if new_val >= min_budget:
                trajectory_budget[idx] = new_val

    # Optional logging
    if writer is not None and count is not None:
        for idx, (kl, sm, tb) in enumerate(zip(kl_tensor, softmax_kl, trajectory_budget)):
            writer.add_scalar(f"KL_Divergence/Agent_{idx}", kl.item(), count)
            writer.add_scalar(f"Softmax_KL/Agent_{idx}", sm.item(), count)
            writer.add_scalar(f"Trajectory_Budget/Agent_{idx}", tb.item(), count)

    print("Trajectory Budget per Agent:", trajectory_budget.tolist())
    return trajectory_budget.tolist()

def compute_trajectory_budget_from_kl(
    last_kl_divs,
    total_trajs,
    min_budget=1,
    writer=None,
    count=None,
    *,
    prev_budget=None,      # previous allocation list/tensor or None
    alpha=0.3,             # smoothing strength
    temperature=1.0,       # softmax temperature (>1 = smoother, <1 = sharper)
    cap_step_abs=None,     # max absolute change per agent
    cap_step_frac=None     # max fractional change of previous
):
    """
    Smooth trajectory budget allocation based on KL divergences.
    Ensures sum == total_trajs and each agent >= min_budget.
    """
    # Convert to tensor and keep device consistent
    kl_tensor = torch.as_tensor(last_kl_divs, dtype=torch.float32)
    device = kl_tensor.device
    num_agents = kl_tensor.numel()

    # 1) Softmax weighting with temperature
    weights = F.softmax(kl_tensor / max(temperature, 1e-8), dim=0)

    # 2) Base allocation (on same device)
    base = torch.full((num_agents,), float(min_budget), device=device)
    remaining = max(0, int(total_trajs) - int(min_budget) * num_agents)
    target_float = base + remaining * weights  # [A]

    # 3) Previous allocation
    if prev_budget is None:
        prev_float = base + remaining / max(num_agents, 1)
    else:
        prev_float = torch.as_tensor(prev_budget, dtype=torch.float32, device=device)

    # 4) Exponential smoothing
    smoothed_float = (1.0 - alpha) * prev_float + alpha * target_float

    # 5) Optional per-step caps
    if cap_step_abs is not None or cap_step_frac is not None:
        delta = smoothed_float - prev_float
        if cap_step_abs is not None:
            delta = torch.clamp(delta, -abs(cap_step_abs), abs(cap_step_abs))
        if cap_step_frac is not None:
            lim = (prev_float.abs() * float(cap_step_frac)).clamp_min(1.0)
            delta = torch.max(torch.min(delta, lim), -lim)
        smoothed_float = prev_float + delta

    # 6) Convert to integers while keeping sum and min constraints
    ints = torch.floor(smoothed_float).to(torch.int64)
    ints = torch.max(ints, torch.as_tensor(min_budget, dtype=torch.int64, device=device))

    current_sum = int(ints.sum().item())
    diff = total_trajs - current_sum

    if diff != 0:
        frac = (smoothed_float - ints.float())
        if diff > 0:
            order = torch.argsort(frac, descending=True)
            for i in range(diff):
                ints[order[i % num_agents]] += 1
        else:
            order = torch.argsort(frac, descending=False)
            to_remove = -diff
            i = 0
            while to_remove > 0 and i < num_agents:
                idx = order[i].item()
                if ints[idx] > min_budget:
                    ints[idx] -= 1
                    to_remove -= 1
                else:
                    i += 1
            if to_remove > 0:
                order2 = torch.argsort(ints, descending=True)
                j = 0
                while to_remove > 0 and j < num_agents:
                    idx = order2[j].item()
                    if ints[idx] > min_budget:
                        ints[idx] -= 1
                        to_remove -= 1
                    else:
                        j += 1

    # 7) Optional TensorBoard logging
    if writer is not None and count is not None:
        softmax_kl = F.softmax(kl_tensor, dim=0)
        for idx, (kl, sm, tb) in enumerate(zip(kl_tensor, softmax_kl, ints)):
            writer.add_scalar(f"KL_Divergence/Agent_{idx}", kl.item(), count)
            writer.add_scalar(f"Softmax_KL/Agent_{idx}", sm.item(), count)
            writer.add_scalar(f"Trajectory_Budget/Agent_{idx}", tb.item(), count)

    out = ints.cpu().tolist()  # move to CPU for printing/safe usage
    print("Trajectory Budget per Agent (smoothed):", out)
    return out

def log_barplot(writer, tag, agents,values, step):
    labels = [f"Agent {i}" for i in range(agents[0])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="skyblue")
    ax.set_title(tag)
    ax.set_ylabel("Value")
    ax.set_xlabel("Agent")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    writer.add_figure(tag, fig, step)
    plt.close(fig)


def initilize_from_me(multihead_policy, me_policy,num_agents):

    # Copy shared trunk
    multihead_policy.net.load_state_dict(me_policy.net.state_dict())

    # Copy the adapter (Sequential: Linear + activation) from the single-head policy
    src_adapter_state = me_policy.head_adapters[0].state_dict()

    # Only copy as many heads as both policies actually have to avoid index issues
    heads_to_copy = min(num_agents, len(multihead_policy.head_adapters))
    for i in range(heads_to_copy):
        with torch.no_grad():
            multihead_policy.head_adapters[i].load_state_dict(src_adapter_state)

            # Replicate the single-head output layers across all heads
            multihead_policy.fc_mean.weight[i].copy_(me_policy.fc_mean.weight[0])
            multihead_policy.fc_mean.bias[i].copy_(me_policy.fc_mean.bias[0])
            multihead_policy.fc_log_std.weight[i].copy_(me_policy.fc_log_std.weight[0])
            multihead_policy.fc_log_std.bias[i].copy_(me_policy.fc_log_std.bias[0])

    return multihead_policy