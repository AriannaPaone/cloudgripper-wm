"""Does the trained policy condition on the cube's position?

Feeds the network fabricated observations that hold the arm state fixed and
vary only the cube position, then reports how the action distribution changes.

The policy outputs a Gaussian per action dimension, not an action: a mean and
a state-dependent log-std, sampled and squashed through tanh. So there are two
channels through which the cube could influence behaviour:

  - the MEAN: where the policy aims
  - the STD:  how much it explores around that aim

Checking only the mean is misleading. A policy can condition entirely through
the std (being near-deterministic in one dimension and near-uniform in
another) and still be responding to the object.

Note this tests sensitivity, not competence. A policy could respond to the
cube and respond wrongly.

Usage:
    uv run python scripts/data/check_cube_sensitivity.py <checkpoint.pt>
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from scripts.data.KMyriad.policy_multihead import PolicyMultiheadNetwork


# Action dimensions, in the order of the env's joint_names. All are deltas in
# normalised [0,1] space, bounded by +-max_delta (0.05):
#   dx     Rail_joint      translate along x
#   dy     Slider_joint    translate along y
#   dz     Linear_joint    raise / lower  (higher normalised value = lower arm)
#   drot   Rotation_joint  rotate the gripper
#   dgrip  RightSpur_joint open / close the fingers
#                          (normalised 0 = open, 1 = closed — inverted
#                           relative to the real robot's move_gripper)
ACTION_NAMES = ["dx", "dy", "dz", "drot", "dgrip"]

# Cube positions in the normalised [0,1] space the policy sees, i.e. after
# env_adapter divides by OBJ_LOW/OBJ_HIGH. 0.5 is mid-workspace.
CUBE_POSITIONS = [
    ("centre",     [0.50, 0.50]),
    ("near -x -y", [0.10, 0.10]),
    ("near +x +y", [0.90, 0.90]),
    ("near -x +y", [0.10, 0.90]),
    ("near +x -y", [0.90, 0.10]),
]

# Arm states to test at. A policy might attend to the cube only from some
# configurations, so more than one is worth checking.
ARM_STATES = [
    ("arm centred", [0.5, 0.5, 0.5, 0.5, 0.5]),
    ("arm low",     [0.5, 0.5, 0.9, 0.5, 0.5]),
    ("arm corner",  [0.1, 0.1, 0.5, 0.5, 0.5]),
]

CUBE_Z = 0.278      # normalised resting height: 0.0139 m / 0.05 m

# set_control ignores commands below this, mirroring the real robot's
# behaviour of not actuating for tiny moves. Useful as a reference: variation smaller than this can never reach an actuator.
STEP_THRESHOLD = 0.01


def load_net(path, device):

    """Rebuild the network from the checkpoint's own shapes.
    """

    sd = torch.load(path, map_location=device)

    hidden = [sd["net.0.weight"].shape[0]]
    i = 2
    while f"net.{i}.weight" in sd:
        hidden.append(sd[f"net.{i}.weight"].shape[0])
        i += 2

    latent = sd["latent_proj.weight"].shape[0] if "latent_proj.weight" in sd else 2
    n_agents = sd["fc_mean.bias"].shape[0]
    state_dim = sd["net.0.weight"].shape[1]
    action_dim = sd["fc_mean.bias"].shape[1]

    # PolicyMultiheadNetwork needs an action_space for its rescaling buffers.
    # The values are in the checkpoint, so no env is required.
    scale = sd["action_scale"].cpu().numpy()
    bias = sd["action_bias"].cpu().numpy()

    class _Space:
        low = bias - scale
        high = bias + scale
        shape = (action_dim,)

    net = PolicyMultiheadNetwork(
        hidden_sizes=hidden, adapter_hidden=hidden[-1],
        activation=torch.nn.ReLU, num_envs=1, num_agents=n_agents,
        state_dim=state_dim, action_dim=action_dim,
        action_space=_Space(), latent_proj_dim=latent,
    ).to(device) #the constructor builds an empty network with the right shapes, then we load the weights from the checkpoint
    net.load_state_dict(sd) # load the weights from the checkpoint
    net.eval() #switch to evaluation mode

    fixed = getattr(net, "use_fixed_std", None) # check if the network uses a fixed standard deviation
    print(f"hidden {hidden}   agents {n_agents}   state_dim {state_dim}   "
          f"action_dim {action_dim}   latent {latent}") # print the network architecture
    print(f"use_fixed_std={fixed}  "
          f"(False means the std is a learned function of the state)")
    print(f"log_std bounds [{net.LOG_STD_MIN}, {net.LOG_STD_MAX}]  ->  "
          f"std in [{np.exp(net.LOG_STD_MIN):.4f}, {np.exp(net.LOG_STD_MAX):.2f}]") # prints the scale for reading the std values in the output table
    return net, action_dim


def sweep(net, arm, device, action_dim):

    """Mean action and std for each cube position, arm state held fixed."""

    head = torch.zeros(1, dtype=torch.long, device=device) #we only have one head, so this is always 0
    means, stds = [], []
    for _, cube in CUBE_POSITIONS:
        x = torch.tensor(np.concatenate([arm, cube, [CUBE_Z]])[None],
                         dtype=torch.float32, device=device) #builds a tensor of shape [1, state_dim] with the arm state, cube position, and cube height
        with torch.no_grad():
            # forward_select gives the raw distribution parameters (means, log_stds), pre tanh;
            # sample() gives (action, log_prob, mean_action), the mean already squashed and rescaled, comparable to an action.
            _, log_std = net.forward_select(x, head)
            _, _, mean_action = net.sample(x)
        means.append(mean_action[0, 0].cpu().numpy())
        stds.append(log_std.exp()[0].cpu().numpy())
    return np.array(means), np.array(stds)


def table(title, rows, spread, names, fmt="{:10.4f}"):
    print(f"  {title}")
    print("    " + f"{'cube':14s}" + "".join(f"{n:>10s}" for n in names))
    for (label, _), row in zip(CUBE_POSITIONS, rows):
        print("    " + f"{label:14s}" + "".join(fmt.format(v) for v in row))
    print("    " + f"{'spread':14s}" + "".join(fmt.format(v) for v in spread))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, action_dim = load_net(args.checkpoint, device)
    print(f"{os.path.basename(args.checkpoint)}\n")

    names = ACTION_NAMES[:action_dim]
    mean_spreads, std_rel_spreads = [], []

    for arm_label, arm in ARM_STATES: #The arm configurations
        print(f"=== {arm_label}: {arm} ===")
        means, stds = sweep(net, arm, device, action_dim) #This gives a matrix of means and stds for each cube position

        #How much did this dimention change as teh cube moves?
        m_spread = means.max(axis=0) - means.min(axis=0) #Collapses the position axis, giving the spread of the mean action for each action dimension. 
        mean_spreads.append(m_spread)
        table("mean action (after tanh and rescaling)", means, m_spread, names)

        # How much did the std change as the cube moves?
        s_spread = stds.max(axis=0) - stds.min(axis=0)
        # Relative, because an absolute change of 0.5 means something very
        # different at std 0.03 than at std 5.0.
        s_rel = s_spread / np.maximum(stds.mean(axis=0), 1e-9) #Expressed as a fraction of its own magnitude, because a change of 0.5 is very different if the std is 0.03 or 5.0
        std_rel_spreads.append(s_rel)
        print()
        table("std (pre-tanh, per dimension)", stds, s_spread, names)
        print("    " + f"{'spread %':14s}"
              + "".join(f"{100*v:9.1f}%" for v in s_rel))
        print()

    # Check for sign changes across cube positions
    signs = np.sign(means).astype(int)
    flips = (signs != signs[0]).any(axis=0)
    print("  sign changes across cube positions:",
          {n: bool(f) for n, f in zip(names, flips)})

    # mean against a sample
    # If sampled actions are far larger than the mean, behaviour is dominated
    # by exploration noise rather than by where the policy aims.
    head = torch.zeros(1, dtype=torch.long, device=device)
    x = torch.tensor(np.concatenate([ARM_STATES[0][1], [0.5, 0.5], [CUBE_Z]])[None],
                     dtype=torch.float32, device=device)
    print("\n=== mean against three samples (arm centred, cube centred) ===")
    with torch.no_grad():
        _, _, m = net.sample(x)
    print("  " + f"{'mean':10s}" + "".join(f"{v:10.4f}" for v in m[0, 0].cpu().numpy()))
    for i in range(3):
        with torch.no_grad():
            a, _, _ = net.sample(x)
        print("  " + f"{'sample ' + str(i):10s}"
              + "".join(f"{v:10.4f}" for v in a[0, 0].cpu().numpy()))


if __name__ == "__main__":
    main()