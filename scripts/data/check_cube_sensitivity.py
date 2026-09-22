"""Does the trained policy condition on the cube's position?

Two tests:

1. Sweep fabricated observations: hold the arm fixed, put the cube
   in a few places, compare the action distributions.
2. Shuffle (with --dataset). Real observations from a collected dataset:
   shuffle the cube positions between rows and measure how much the action
   distributions change. Shuffling the arm instead gives a baseline, since
   the policy certainly uses its own state.

Test 2 is the more reliable one. Its observations are in-distribution, and it
includes states where the arm is near the cube, which the sweep does not.

The policy outputs a Gaussian per action dimension (pre-tanh mean and std),
sampled, squashed through tanh and rescaled to +-max_delta. So:
  - "executed" numbers are statistics of the actions the env actually
    receives, estimated by sampling. With a large std, tanh saturates and
    the executed action barely depends on the mean.
  - KL divergence compares two pre-tanh Gaussians, mean and std together,
    in one number (0 = identical distributions).

Observations are built with the env adapter's own _to_policy_obs, so layout,
normalisation and clipping match training exactly.

This tests sensitivity, not competence: a policy can respond to the cube and
respond wrongly.

Usage:
    uv run python scripts/data/check_cube_sensitivity.py <checkpoint.pt>
        [--dataset <path>.lance] [--fixed-std auto|true|false]
"""

import argparse
import os
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from scripts.data.KMyriad.policy_multihead import PolicyMultiheadNetwork
# Change this import to wherever MaxEntEnvAdapter lives in your repo:
from scripts.data.KMyriad.env_adapter import MaxEntEnvAdapter, OBJ_LOW, OBJ_HIGH

ACTION_NAMES = ["dx", "dy", "dz", "drot", "dgrip"]
STEP_THRESHOLD = 0.01    # set_control ignores smaller commands
CUBE_REST_Z = 0.0139     # metres
N_SAMPLES = 4000         # samples per distribution for the "executed" statistics
STATE_FILTER = None      # set by --state-filter: which observation features the policy takes

# Cube positions as fractions of the adapter's workspace (OBJ_LOW..OBJ_HIGH).
CUBE_FRACTIONS = [
    ("centre",     [0.50, 0.50]),
    ("near -x -y", [0.10, 0.10]),
    ("near +x +y", [0.90, 0.90]),
    ("near -x +y", [0.10, 0.90]),
    ("near +x -y", [0.90, 0.10]),
]

# Arm states in the env's normalised [0, 1] state (x, y, z, rot, grip).
ARM_STATES = [
    ("arm centred", [0.5, 0.5, 0.5, 0.5, 0.5]),
    ("arm low",     [0.5, 0.5, 0.9, 0.5, 0.5]),
    ("arm corner",  [0.1, 0.1, 0.5, 0.5, 0.5]),
]


# ---------------------------------------------------------------- network

def detect_fixed_std(sd):
    """Tell from the weights whether training used the fixed or the learned std.

    Only the branch that was used receives gradients, so the other one keeps
    its initial value: log_std_constant starts at -2.0, fc_log_std.bias at 0.
    """
    const_untouched = bool(torch.all(sd["log_std_constant"] == -2.0))
    bias_untouched = bool(torch.all(sd["fc_log_std.bias"] == 0))
    if const_untouched and not bias_untouched:
        return False
    if bias_untouched and not const_untouched:
        return True
    raise SystemExit("Cannot tell whether this checkpoint used a fixed std. "
                     "Pass --fixed-std true or --fixed-std false.")


def load_net(path, device, fixed_std):
    """Rebuild the network from the checkpoint's own shapes."""
    sd = torch.load(path, map_location=device)

    hidden = [sd["net.0.weight"].shape[0]]
    i = 2
    while f"net.{i}.weight" in sd:
        hidden.append(sd[f"net.{i}.weight"].shape[0])
        i += 2
    n_heads, action_dim = sd["fc_mean.bias"].shape
    state_dim = sd["net.0.weight"].shape[1]

    if fixed_std is None:
        fixed_std = detect_fixed_std(sd)

    # The constructor needs an action space for its scaling buffers;
    # the real values are in the checkpoint and get loaded below.
    scale = sd["action_scale"].cpu().numpy()
    bias = sd["action_bias"].cpu().numpy()
    space = SimpleNamespace(low=bias - scale, high=bias + scale, shape=(action_dim,))

    net = PolicyMultiheadNetwork(
        hidden_sizes=hidden,
        adapter_hidden=sd["head_adapters.0.0.weight"].shape[0],
        activation=torch.nn.ReLU,
        num_envs=n_heads, num_agents=n_heads,
        state_dim=state_dim, action_dim=action_dim, action_space=space,
        latent_proj_dim=sd["latent_proj.weight"].shape[0],
        use_fixed_std=fixed_std,
    ).to(device)
    net.load_state_dict(sd)
    net.eval()

    print(f"hidden {hidden}   heads {n_heads}   state_dim {state_dim}   action_dim {action_dim}")
    print(f"std: {'fixed per head' if fixed_std else 'learned, state-dependent'}"
          f"   pre-tanh std range [{np.exp(net.LOG_STD_MIN):.4f}, {np.exp(net.LOG_STD_MAX):.2f}]")
    print(f"action scale (max_delta) {scale}")
    return net, state_dim, n_heads, action_dim


def action_names(action_dim):
    """dx, dy, ... and, for chunked policies, dx1, dy1, ... for later chunks."""
    names = []
    for i in range(action_dim):
        chunk = i // len(ACTION_NAMES)
        names.append(ACTION_NAMES[i % len(ACTION_NAMES)] + (str(chunk) if chunk else ""))
    return names


# ---------------------------------------------------------------- building blocks

def make_obs(state, obj_metres, device):
    """Policy observations, built exactly as the env adapter builds them."""
    stub = SimpleNamespace(normalise_object=True, device=device)
    raw = {"state": np.asarray(state, dtype=np.float32),
           "object_position": np.asarray(obj_metres, dtype=np.float32)}
    obs = MaxEntEnvAdapter._to_policy_obs(stub, raw)["policy"]
    if STATE_FILTER is not None:      # policy trained on a subset of the features
        obs = obs[:, STATE_FILTER]
    return obs


def gaussians(net, obs, head):
    """Pre-tanh mean and std for every row, using one head."""
    idx = torch.full((len(obs),), head, dtype=torch.long, device=obs.device)
    with torch.no_grad():
        mean, log_std = net.forward_select(obs, idx) # takes 1 index per row and returns the corresponding head's output
    return mean, log_std.exp()


def executed(net, mean, std, n=N_SAMPLES):
    """Mean and std of the actions the env actually receives, by sampling."""
    u = mean + std * torch.randn((n,) + tuple(mean.shape), device=mean.device) # torch.randn(4000, 5, 5) random samples from a normal distribution with mean 0 and std 1, shape [n_samples, n_rows, action_dim], rescaled to the correct mean and std for each row
    a = torch.tanh(u) * net.action_scale + net.action_bias # rescale to the env's action space
    return a.mean(0), a.std(0) # mean and std over the samples, shape [n_rows, action_dim]


def kl(m1, s1, m2, s2):
    """KL( N(m1, s1) || N(m2, s2) ) per action dimension.
    closed form formula for 2 univariate gaussians"""
    return torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5


def print_row(label, values, fmt="{:10.4f}"):
    print("    " + f"{label:18s}" + "".join(fmt.format(float(v)) for v in values))


# ---------------------------------------------------------------- test 1: sweep

def sweep(net, head, names, device):
    low, high = np.asarray(OBJ_LOW), np.asarray(OBJ_HIGH)
    cubes = []
    for _, (fx, fy) in CUBE_FRACTIONS:
        xy = low[:2] + np.array([fx, fy]) * (high[:2] - low[:2])
        cubes.append([xy[0], xy[1], CUBE_REST_Z]) # metres, not fractions

    for arm_label, arm in ARM_STATES:
        print(f"\n  --- {arm_label}: {arm} ---")
        obs = make_obs([arm] * len(cubes), cubes, device) # builds a tensor of shape [n_cubes, state_dim] with the arm state repeated for each cube position
        mean, std = gaussians(net, obs, head)
        act_mean, act_std = executed(net, mean, std)
        kl_to_centre = kl(mean, std, mean[:1], std[:1]).sum(-1)   # centre is row 0
        # mean and std are [n_cubes, action_dim], mean[:1] and std[:1] are [1, action_dim], so broadcasting gives [n_cubes, action_dim] and sum(-1) gives [n_cubes], 
        # the kl is summed over the action dimensions, so we get one number per cube position

        print("    executed action mean")
        print("    " + f"{'cube':18s}" + "".join(f"{n:>10s}" for n in names) + f"{'KL vs centre':>14s}")
        for (label, _), row, k in zip(CUBE_FRACTIONS, act_mean, kl_to_centre):
            print("    " + f"{label:18s}" + "".join(f"{float(v):10.4f}" for v in row) + f"{float(k):14.4f}")

        spread = act_mean.max(0).values - act_mean.min(0).values # per-dim spread of the executed action means, shape [action_dim]
        print_row("spread", spread)
        print("    " + f"{'spread > 0.01?':18s}"
              + "".join(f"{'YES' if v > STEP_THRESHOLD else 'no':>10s}" for v in spread))
        print_row("noise (exec std)", act_std.mean(0)) # mean of the executed action stds over the cube positions, shape [action_dim]

        signs = torch.sign(act_mean)
        flips = (signs != signs[:1]).any(0)
        print("    " + f"{'sign flips':18s}" + "".join(f"{'YES' if f else 'no':>10s}" for f in flips))

        print("    pre-tanh std")
        for (label, _), row in zip(CUBE_FRACTIONS, std):
            print_row(label, row) # pre-tanh std for each cube position, shape [action_dim]


# ---------------------------------------------------------------- test 2: shuffle

def shuffle_test(net, head, names, dataset, device, n_rows=4096):
    import lance
    t = lance.dataset(dataset).to_table(columns=["state", "object_position"]).to_pydict()
    state = np.asarray(t["state"], dtype=np.float32) # shape [n_rows, state_dim]
    obj = np.asarray(t["object_position"], dtype=np.float32) # shape [n_rows, 3]

    rng = np.random.default_rng(0) # seeded for reproducibility
    rows = rng.choice(len(state), size=min(n_rows, len(state)), replace=False) # randomly select a subset of rows to use for the shuffle test
    state, obj = state[rows], obj[rows] # shape [n_rows, state_dim] and [n_rows, 3], the selected rows of the original dataset
    perm = rng.permutation(len(rows)) # a random permutation of the indices of the selected rows, used to shuffle the cube positions and arm states independently

    real = gaussians(net, make_obs(state, obj, device), head) # shape [n_rows, action_dim] for mean and std, the original observations
    cube_shuffled = gaussians(net, make_obs(state, obj[perm], device), head) # shape [n_rows, action_dim] for mean and std, the observations with shuffled cube positions
    arm_shuffled = gaussians(net, make_obs(state[perm], obj, device), head) # shape [n_rows, action_dim] for mean and std, the observations with shuffled arm states

    real_act, _ = executed(net, *real, n=500) # shape [n_rows, action_dim], the mean of the executed actions for the original observations
    results = {}
    for label, other in [("cube shuffled", cube_shuffled), ("arm shuffled", arm_shuffled)]:
        other_act, _ = executed(net, *other, n=500) # shape [n_rows, action_dim], the mean of the executed actions for the shuffled observations
        results[label] = {
            "kl": kl(*real, *other).mean(0),                       # average KL divergence per action dimension, averaged over rows
            "dact": (real_act - other_act).abs().mean(0),          # average absolute difference in executed actions, per dim, averaged over rows
        }

    print(f"\n  --- shuffle test on {len(rows)} dataset rows ---")
    print("    " + f"{'':18s}" + "".join(f"{n:>10s}" for n in names) + f"{'total':>10s}")
    for label, r in results.items():
        print_row(f"KL {label}", list(r["kl"]) + [r["kl"].sum()])
    for label, r in results.items():
        print_row(f"|dAct| {label}", list(r["dact"]) + [r["dact"].sum()])

    arm_kl = float(results["arm shuffled"]["kl"].sum())
    if arm_kl > 0:
        ratio = float(results["cube shuffled"]["kl"].sum()) / arm_kl
        print(f"\n    cube/arm KL ratio: {ratio:.3f}"
              "   (~0: ignores the cube; ~1 or more: the cube matters as much as the arm)")
    else:
        print("\n    arm shuffling changed nothing, so there is no baseline to compare against")
    if STATE_FILTER is not None and not set(range(5)) & set(STATE_FILTER):
        print("    note: the arm state is not part of this policy's input, so the arm "
              "baseline only acts through the relative-position features")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--dataset", help="Lance dataset for the shuffle test")
    ap.add_argument("--fixed-std", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--state-filter", default=None,
                    help="comma-separated feature indices the policy was trained on, "
                         "e.g. 5,6,7,8,9,10 (default: all 11 adapter features)")
    args = ap.parse_args()

    global STATE_FILTER
    if args.state_filter:
        STATE_FILTER = [int(i) for i in args.state_filter.split(",")]

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_std = {"auto": None, "true": True, "false": False}[args.fixed_std]
    net, state_dim, n_heads, action_dim = load_net(args.checkpoint, device, fixed_std)
    names = action_names(action_dim)

    probe = make_obs([ARM_STATES[0][1]], [[0.0, 0.0, CUBE_REST_Z]], device)
    assert probe.shape[1] == state_dim, (
        f"observation has {probe.shape[1]} features, network expects {state_dim}. "
        "If the policy was trained on filtered features, pass --state-filter, "
        "e.g. --state-filter 5,6,7,8,9,10")

    print(f"{os.path.basename(args.checkpoint)}")
    for head in range(n_heads):
        print(f"\n=================== head {head} ===================")
        sweep(net, head, names, device)
        if args.dataset:
            shuffle_test(net, head, names, args.dataset, device)


if __name__ == "__main__":
    main()