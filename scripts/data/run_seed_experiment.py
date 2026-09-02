"""Train MaxEnt across several seeds and compare against sticky-random.

For each training seed: train a policy, collect a dataset with it, collect a
sticky-random dataset under the same collection seed, then report confidence
intervals and write plots.

The comparison is paired: both policies use the same collection seed, so they
face the same sequence of starting configurations and the difference between
them is not confounded by which conditions came up.

Run from the project root:
    uv run python scripts/data/run_seed_experiment.py
"""

import cmd
import json
import os
import pathlib
import subprocess

import matplotlib
matplotlib.use("Agg")

import lance
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.data.KMyriad.MaxEnt_Automatic import train_maxent_policy


# config

SEEDS = [0,1,1234]

# Training. num_epochs=100 is where the working result came from; 10 is only useful for checking the pipeline runs end to end.
NUM_EPOCHS = 100 #10
HIDDEN = [512, 256]
STATE_FILTER = [0, 1, 5, 6]     # arm x,y + cube x,y
TRAJ_LEN = 300
TOTAL_TRAJS = 32                # for 1 agent, must equal NUM_ENVS: the heatmap reshape in
NUM_ENVS = 32                   # pl_agent assumes num_trajectories*num_agents = num_envs
K = 5                           
TRUNK_LR = 0.0005
HEAD_LR = 0.0002
MILESTONES = [80]

# Evaluation
EPISODES = 200
STEPS = 300
COLLECT_ENVS = 2

# Analysis
CONTACT_MM = 2.0                # threshold: below this is physics jitter, not contact
N_BOOT = 10_000
Z_LIMITS = (0.0, 0.10)          # metres; below 0 it fell through the floor, above 0.1 it was launched. 

OUTDIR = pathlib.Path("results/seed_experiment_fixedobjects") # where to write plots and results.json
BOOT_RNG = np.random.default_rng(0)
STICKY_C, MAXENT_C = "#8a8a8a", "#2f6fb0"

RANDOM_OBJECT_POS = False
NAMEDATASET = "fixedstart" if not RANDOM_OBJECT_POS else "randomstart"
# collection

def collect(output_name, seed, checkpoint=None):

    """Run one collection through the project's Hydra script.
    Subprocess (own memory, own imports and environment) rather than an import because collect_cloudgripper_mujoco.run
    is wrapped in @hydra.main, which takes over sys.argv and the working
    directory.
    """

    cmd = ["uv", "run", "python", "scripts/data/collect_cloudgripper_mujoco.py"]
    if checkpoint: #passing no checkpoint is the sticky-random baseline, which uses the default policy in the Hydra config.
        cmd += ["--config-name", "mj_maxent", f"policy.checkpoint='{checkpoint}'"]
    cmd += [
        f"episodes={EPISODES}",
        f"num_envs={COLLECT_ENVS}",
        f"world.max_episode_steps={STEPS}",
        f"seed={seed}",
        f"output_name={output_name}",
    ]
    if RANDOM_OBJECT_POS:
        subprocess.run(cmd, env={**os.environ, "MUJOCO_GL": "egl", "CG_VARY_OBJECT": "1"}, check=True) #for randomstart
    else:
        subprocess.run(cmd, env={**os.environ, "MUJOCO_GL": "egl"}, check=True) #for no randomstart
    #subprocess.run asks the os to run the command in a new process
    #This is because we don't want to change the current process's environment or imports, which would happen if we just imported and called the function directly.
    #Each subprocess gets its own environment variable, which also allows to set CG_VARY_OBJECT=1 for the collection, which is needed for the object to vary in position.

# analysis
def episode_displacements(name):

    """Per-episode maximum cube displacement in mm, and a dropped count.
    Episodes where the cube left the workspace are excluded: one escaped cube
    can shift a mean by orders of magnitude, and it reflects a physics failure
    rather than behaviour.
    """

    ds = lance.dataset(f".stable_worldmodel/datasets/{name}.lance") #opens the dataset
    t = ds.to_table(columns=["episode_idx", "object_position"]).to_pydict() #pull the columns we care about into a dict of lists
    ep, obj = np.array(t["episode_idx"]), np.array(t["object_position"]) # convert the lists to numpy arrays for easier manipulation

    kept, dropped = [], 0 # kept will hold the max displacements for episodes that are valid, dropped counts how many episodes were dropped due to the cube leaving the workspace
    for e in np.unique(ep): #Iterate per episode 
        p = obj[ep == e] # Boolean mask, selecting the object positions for the current episode
        if p[:, 2].min() < Z_LIMITS[0] or p[:, 2].max() > Z_LIMITS[1]: # If the cube left the workspace on the z axis
            dropped += 1
            continue
        # max over timesteps and axes: how far the cube ever got from its start
        kept.append(np.abs(p - p[0]).max() * 1000) 
        #3 operations: 
        # p - p[0] subtracts the starting pos from all timesteps, giving the displacement from start
        # np.abs(...) takes the absolute value of the displacement, so we only care about distance
        # .max() takes the maximum displacement over all timesteps and axes, giving the max over both dimentions
        # * 1000 converts from meters to millimeters, since the original positions are in meters
    return np.array(kept), dropped # 1 number per surviving episode, and the count of dropped episodes


def wilson(k, n, z=1.96):

    """
    Wilson score, it's a confidence interval for a proportion.
    k = number of successes (e.g., episodes with contact)
    n = number of trials (e.g., total episodes)
    z = standard normal quantile for 95% confidence (95% of a normal distribution is within 1.96 standard deviations of the mean)
    Wilson score is preferred since my results sit at the extremes (around 0.9)
    """

    if n == 0: #If every episode was dropped due to failures, nothing to estimate
        return (float("nan"), float("nan"))
    p= k / n # observed proportion of successes
    d = 1 + z * z / n # denominator of the Wilson score interval formula
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def boot_ci(x, median=False, alpha=0.05):

    """Percentile bootstrap interval.
    Resamples with replacement, recomputes the statistic, takes percentiles.
    Assumes nothing about the distribution.
    """

    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = BOOT_RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    r = x[idx]
    vals = np.median(r, axis=1) if median else r.mean(axis=1)
    return tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def summarise(name):
    d, dropped = episode_displacements(name)
    n = len(d) # number of episodes that were kept (not dropped)
    k = int((d > CONTACT_MM).sum()) # number of episodes where the max displacement was greater than the contact threshold
    return {
        "name": name, "n": n, "dropped": dropped, "disp": d,
        "contact": k / n if n else float("nan"), # proportion of episodes with contact
        "contact_ci": wilson(k, n), # confidence interval for the contact proportion
        "mean": float(d.mean()) if n else float("nan"), # mean of the max displacements
        "mean_ci": boot_ci(d), # confidence interval for the mean of the max displacements
        "median": float(np.median(d)) if n else float("nan"), # median of the max displacements
        "median_ci": boot_ci(d, median=True), # confidence interval for the median of the max displacements
    }


# report
def report(results):
    print("\n" + "=" * 78)
    print(f"per seed   (contact threshold {CONTACT_MM}mm, 95% intervals, "
          f"n={EPISODES} episodes)")
    print("=" * 78)
    for i, seed in enumerate(SEEDS):
        for pol in ("sticky", "maxent"):
            s = results[pol][i]
            drop = f"  [{s['dropped']} dropped]" if s["dropped"] else ""
            print(f"  seed {seed:<5} {pol:7s} "
                  f"contact {s['contact']:.3f} "
                  f"({s['contact_ci'][0]:.3f}-{s['contact_ci'][1]:.3f})   "
                  f"mean {s['mean']:6.1f}mm "
                  f"({s['mean_ci'][0]:.0f}-{s['mean_ci'][1]:.0f})   "
                  f"median {s['median']:6.1f}mm{drop}")
        print()

    print("=" * 78)
    print("across seeds")
    print("=" * 78)
    print("  sticky's spread is evaluation noise only — it has no training seed.")
    print("  maxent's spread includes training variability, which is the thing")
    print("  that says whether the method works or one checkpoint got lucky.\n")
    for pol in ("sticky", "maxent"):
        for key in ("contact", "mean", "median"):
            v = np.array([s[key] for s in results[pol]])
            lo, hi = boot_ci(v)
            unit = "" if key == "contact" else "mm"
            print(f"  {pol:7s} {key:7s} {v.mean():7.3f}{unit}  ({lo:.3f}-{hi:.3f})"
                  f"   [min {v.min():.3f}, max {v.max():.3f}]")
        print()

    print("=" * 78)
    print("paired difference")
    print("=" * 78)
    print("  Same collection seed for both policies, so the difference per seed")
    print("  is not confounded by which starting configurations came up.\n")
    diffs = np.array([results["maxent"][i]["contact"] - results["sticky"][i]["contact"]
                      for i in range(len(SEEDS))])
    # subtracts within each seed
    lo, hi = boot_ci(diffs)
    print(f"  contact rate: {diffs.mean():+.3f}   ({lo:+.3f} to {hi:+.3f})")
    print(f"  per seed:     {np.round(diffs, 3)}")


#plots
def make_plots(results):
    x = np.arange(len(SEEDS)) # x positions for the bars
    w = 0.35 # width of the bars
    labels = [str(s) for s in SEEDS] # x-axis labels for the bars

    # contact rate with Wilson intervals
    fig, ax = plt.subplots(figsize=(1.8 * len(SEEDS) + 3, 4))
    for off, pol, colour in ((-w / 2, "sticky", STICKY_C),
                             (w / 2, "maxent", MAXENT_C)):
        v = np.array([s["contact"] for s in results[pol]])
        lo = np.array([s["contact_ci"][0] for s in results[pol]])
        hi = np.array([s["contact_ci"][1] for s in results[pol]])
        # yerr wants distances from the bar top, not absolute positions
        ax.bar(x + off, v, w, yerr=[v - lo, hi - v], capsize=4,
               label=pol, color=colour)
    ax.set_xticks(x, labels)
    ax.set_xlabel("seed")
    ax.set_ylabel(f"episodes with contact (>{CONTACT_MM}mm)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Contact rate, 95% Wilson intervals")
    fig.tight_layout()
    fig.savefig(OUTDIR / "contact_rate.png", dpi=150)
    plt.close(fig)

    # displacement distributions
    # Box plots rather than bars: the distribution is bimodal, with a spike at
    # zero for episodes that never touched the cube. A mean alone hides that.
    fig, ax = plt.subplots(figsize=(1.8 * len(SEEDS) + 3, 4))
    data, colours, ticks = [], [], []
    for i in range(len(SEEDS)):
        for pol, colour in (("sticky", STICKY_C), ("maxent", MAXENT_C)):
            d = results[pol][i]["disp"]
            data.append(d if len(d) else np.array([0.0]))
            colours.append(colour)
        ticks.append(2 * i + 0.5)
    bp = ax.boxplot(data, positions=np.arange(len(data)), widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("seed")
    ax.set_ylabel("max cube displacement (mm)")
    ax.set_title("Displacement (grey sticky, blue MaxEnt)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "displacement.png", dpi=150)
    plt.close(fig)

    # paired view, one line per seed connecting the two policies' contact rates. 
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for i, seed in enumerate(SEEDS):
        ax.plot([0, 1],
                [results["sticky"][i]["contact"], results["maxent"][i]["contact"]],
                marker="o", label=f"seed {seed}")
    ax.set_xticks([0, 1], ["sticky", "maxent"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("contact rate")
    ax.legend(fontsize=8)
    ax.set_title("Paired by collection seed")
    fig.tight_layout()
    fig.savefig(OUTDIR / "paired.png", dpi=150)
    plt.close(fig)


# main

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    checkpoints = {}

    for seed in SEEDS:
        print(f"\n{'#' * 78}\n# seed {seed}\n{'#' * 78}")
        np.random.seed(seed)
        torch.manual_seed(seed)

        if RANDOM_OBJECT_POS:
            ckpt = train_maxent_policy(
                object_pos=None, agent_start_pos=None,
                num_agents=1, multihead=True,
                num_epochs=NUM_EPOCHS, name_env="cloudgripper_mujoco",
                seed=seed, k=K, hidden_sizes=HIDDEN,
                traj_len=TRAJ_LEN, total_trajs=TOTAL_TRAJS, num_envs=NUM_ENVS,
                env=None, chunk_size=1, log_entropy=40,
                trunk_lr=TRUNK_LR, head_lr=HEAD_LR,
                milestones=MILESTONES, state_filtering=STATE_FILTER,
                automatic_budget=False,
            ) #train_maxent_policy is a function that trains a MaxEnt policy using the specified parameters. It returns the path to the checkpoint of the trained policy.
        else:
            ckpt = train_maxent_policy(
                            object_pos=None, agent_start_pos=None,
                            num_agents=1, multihead=True,
                            num_epochs=NUM_EPOCHS, name_env="cloudgripper_mujoco",
                            seed=seed, k=K, hidden_sizes=HIDDEN,
                            traj_len=TRAJ_LEN, total_trajs=TOTAL_TRAJS, num_envs=NUM_ENVS,
                            env=None, chunk_size=1, log_entropy=40,
                            trunk_lr=TRUNK_LR, head_lr=HEAD_LR,
                            milestones=MILESTONES, state_filtering=STATE_FILTER,
                            automatic_budget=False, randomize_object_pos = False,
                        ) #train_maxent_policy is a function that trains a MaxEnt policy using the specified parameters. It returns the path to the checkpoint of the trained policy.

        checkpoints[seed] = ckpt # store the checkpoint path for this seed in a dictionary, so we can save it later in the results.json file
        print(f"\ncheckpoint: {ckpt}")

        # Same collection seed for both, so they see identical conditions.
        collect(f"maxent_{NAMEDATASET}{seed}", seed, checkpoint=ckpt)
        collect(f"sticky_{NAMEDATASET}{seed}", seed)

    results = {
        "sticky": [summarise(f"sticky_{NAMEDATASET}{s}") for s in SEEDS],
        "maxent": [summarise(f"maxent_{NAMEDATASET}{s}") for s in SEEDS],
    }

    report(results)
    make_plots(results)

    with open(OUTDIR / "results.json", "w") as f:
        json.dump({
            "seeds": SEEDS,
            "checkpoints": {str(k): str(v) for k, v in checkpoints.items()},
            "config": {
                "num_epochs": NUM_EPOCHS, "hidden": HIDDEN,
                "state_filter": STATE_FILTER, "traj_len": TRAJ_LEN,
                "num_envs": NUM_ENVS, "k": K,
                "episodes": EPISODES, "steps": STEPS,
            },
            "results": {
                pol: [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in s.items()}
                      for s in results[pol]]
                for pol in results
            },
        }, f, indent=2)

    print(f"\nplots and results.json in {OUTDIR}/")


if __name__ == "__main__":
    main()