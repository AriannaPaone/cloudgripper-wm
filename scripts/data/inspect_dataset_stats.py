"""Summary statistics + figure for a collected Lance dataset.

Usage:
    uv run python scripts/data/inspect_dataset_stats.py <path>.lance
        [--out-dir results/dataset_stats] [--no-plot]

Writes to <out-dir>/<dataset name>/:
    report.txt     the printed report
    summary.json   headline numbers (compare datasets with these)
    episodes.csv   one row per episode
    stats.png      figure

Cube metrics come in two kinds:
  absolute  where the cube was. Inflated by spawn randomization.
  relative  how far and which way the cube moved from its own spawn.
            Independent of where it spawned.
"""

import argparse
import csv
import json
from pathlib import Path

import lance
import numpy as np

from scripts.data.coverage import GridCoverage

# Grids. Arm state is normalized to [0, 1]; cube positions are in metres.
ARM_XY = GridCoverage(bounds=[(0, 1), (0, 1)], bins=[40, 40])
ARM_XYZ = GridCoverage(bounds=[(0, 1), (0, 1), (0, 1)], bins=[20, 20, 20])
CUBE_XY = GridCoverage(bounds=[(-0.1, 0.1), (-0.08, 0.08)], bins=[40, 40])
# Displacement grids: centred on 0, 5 mm cells
DISP_XY = GridCoverage(bounds=[(-0.1, 0.1), (-0.1, 0.1)], bins=[40, 40])
DISP_XYZ = GridCoverage(bounds=[(-0.1, 0.1), (-0.1, 0.1), (-0.02, 0.05)], bins=[40, 40, 14])

CONTACT_MM = 2.0   # cube moved this far in the plane -> it was contacted
LIFT_MM = 5.0      # cube rose this much -> lifted or tipped (edge-tip is ~4 mm)
N_DIR_BINS = 16    # sectors for the push-direction histogram


# ---------------------------------------------------------------- loading

def load(path):
    ds = lance.dataset(path)
    columns = ["episode_idx", "step_idx", "state", "object_position", "action"]
    has_head = "head_idx" in ds.schema.names          # multi-head runs record this
    if has_head:
        columns.append("head_idx")
    t = ds.to_table(columns=columns).to_pydict()
    ep = np.asarray(t["episode_idx"])
    step = np.asarray(t["step_idx"])
    state = np.asarray(t["state"], dtype=float)
    obj = np.asarray(t["object_position"], dtype=float)
    act = np.asarray(t["action"], dtype=float)

    head = np.asarray(t["head_idx"]) if has_head else None
    episode_ids = np.unique(ep)
    episodes = []   # one (T, 3) array of cube positions per episode, in time order
    heads = []      # which policy head collected each episode
    bad_eps = []    # episodes whose steps are not exactly 0, 1, 2, ...
    for e in episode_ids:
        rows = np.flatnonzero(ep == e)
        rows = rows[np.argsort(step[rows])]
        episodes.append(obj[rows])
        heads.append(int(head[rows[0]]) if has_head else 0)
        if not np.array_equal(step[rows], np.arange(len(rows))):
            bad_eps.append(int(e))

    return {"path": str(path), "state": state, "obj": obj, "act": act,
            "episode_ids": episode_ids, "episodes": episodes, "bad_eps": bad_eps,
            "heads": np.array(heads), "has_head": has_head}


# ---------------------------------------------------------------- per episode

def episode_stats(o):
    """Stats for one episode. o: (T, 3) cube positions in metres, in time order."""
    d = (o - o[0]) * 1000                              # displacement from spawn, mm
    dist = np.linalg.norm(d[:, :2], axis=1)            # distance from spawn in the plane
    path = np.linalg.norm(np.diff(d[:, :2], axis=0), axis=1).sum()
    moved = np.flatnonzero(dist > CONTACT_MM)

    return {
        "steps": len(o),
        "spawn_x_mm": o[0, 0] * 1000,
        "spawn_y_mm": o[0, 1] * 1000,
        "max_xy_mm": dist.max(),
        "final_xy_mm": dist[-1],
        "max_abs_x_mm": np.abs(d[:, 0]).max(),
        "max_abs_y_mm": np.abs(d[:, 1]).max(),
        "max_abs_z_mm": np.abs(d[:, 2]).max(),
        "lift_mm": d[:, 2].max(),
        "path_mm": path,
        "straightness": dist[-1] / path if path > 0 else 0.0,
        "first_contact_step": int(moved[0]) if len(moved) > 0 else -1,
        "push_angle_deg": np.degrees(np.arctan2(d[-1, 1], d[-1, 0])),
        "contact": bool(dist.max() > CONTACT_MM),
        "lifted": bool(d[:, 2].max() > LIFT_MM),
    }


def direction_entropy(angles_deg):
    """0 = every push in the same direction, 1 = spread evenly over N_DIR_BINS sectors."""
    if len(angles_deg) < 2:
        return 0.0
    counts, _ = np.histogram(angles_deg, bins=N_DIR_BINS, range=(-180, 180))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(N_DIR_BINS))


def describe(a):
    if len(a) == 0:
        return "n/a"
    return (f"mean {a.mean():6.1f}  median {np.median(a):6.1f}  "
            f"p90 {np.percentile(a, 90):6.1f}  max {a.max():6.1f}")


def displacements(data, dims):
    """Each episode's cube trajectory minus its spawn, first `dims` axes, in metres."""
    return [o[:, :dims] - o[0, :dims] for o in data["episodes"]]


# ---------------------------------------------------------------- report

def report(data, m):
    """Build the text report as a list of lines."""
    state, obj, act = data["state"], data["obj"], data["act"]
    n = len(data["episodes"])
    c = m["contact"]
    lines = []

    lines.append(data["path"])
    lines.append(f"{n} episodes, {len(obj)} rows, steps/episode "
                 f"min {m['steps'].min()} median {int(np.median(m['steps']))} "
                 f"max {m['steps'].max()}")

    lines.append("")
    lines.append("cube interaction  [relative to spawn]")
    lines.append(f"  contact (planar >{CONTACT_MM}mm):   {c.sum()}/{n}  ({c.mean():.0%})")
    lines.append(f"  lifted/tipped (z >+{LIFT_MM}mm):  {m['lifted'].sum()}/{n}"
                 f"  ({m['lifted'].mean():.0%})")
    lines.append(f"  max planar disp (mm)    {describe(m['max_xy_mm'])}")
    lines.append(f"  final planar disp (mm)  {describe(m['final_xy_mm'])}")
    for axis in "xyz":
        lines.append(f"    max |{axis}| (mm)          {describe(m[f'max_abs_{axis}_mm'])}")
    if c.any():
        lines.append(f"  contacted episodes only ({c.sum()}):")
        lines.append(f"    first contact step    median {np.median(m['first_contact_step'][c]):.0f}")
        lines.append(f"    straightness          median {np.median(m['straightness'][c]):.2f}"
                     "   (net/path, 1 = clean push)")
        lines.append(f"    direction entropy     {direction_entropy(m['push_angle_deg'][c]):.2f}"
                     f"   (0 = one direction, 1 = uniform over {N_DIR_BINS} sectors)")

    disp_xy = displacements(data, 2)
    disp_xyz = displacements(data, 3)
    lines.append("")
    lines.append("coverage  [relative to spawn, spawn-independent]")
    lines.append(f"  cube disp xy  pooled       {DISP_XY.coverage_relative(disp_xy):.4f}"
                 f"  of {DISP_XY.total_cells} cells")
    lines.append(f"  cube disp xy  per-episode  "
                 f"{DISP_XY.coverage_relative(disp_xy, pooled=False):.4f}")
    lines.append(f"  cube disp xyz pooled       {DISP_XYZ.coverage_relative(disp_xyz):.4f}"
                 f"  of {DISP_XYZ.total_cells} cells")
    lines.append(f"  outside disp grid          "
                 f"{DISP_XY.outside_fraction(np.concatenate(disp_xy)):.1%}")

    spawns = np.array([o[0, :2] for o in data["episodes"]])
    lines.append("")
    lines.append("coverage  [absolute, inflated by spawn randomization]")
    lines.append(f"  arm xy   {ARM_XY.coverage(state[:, :2]):.3f} of {ARM_XY.total_cells}"
                 f"   entropy {ARM_XY.entropy(state[:, :2]):.2f}")
    lines.append(f"  arm xyz  {ARM_XYZ.coverage(state[:, :3]):.3f} of {ARM_XYZ.total_cells}"
                 f"   entropy {ARM_XYZ.entropy(state[:, :3]):.2f}")
    lines.append(f"  cube xy  {CUBE_XY.coverage(obj[:, :2]):.3f} of {CUBE_XY.total_cells}"
                 f"   (spawns alone {CUBE_XY.coverage(spawns):.3f}, gained by motion "
                 f"{CUBE_XY.excess_coverage([o[:, :2] for o in data['episodes']]):.3f})")
    lines.append(f"  outside grid: arm {ARM_XY.outside_fraction(state[:, :2]):.1%}, "
                 f"cube {CUBE_XY.outside_fraction(obj[:, :2]):.1%}")

    spawns_mm = spawns * 1000
    n_distinct = len(np.unique(spawns_mm.round(1), axis=0))
    lines.append("")
    lines.append(f"spawn positions: {n_distinct}/{n} distinct, "
                 f"std x {spawns_mm[:, 0].std():.1f}mm y {spawns_mm[:, 1].std():.1f}mm")
    if n_distinct == 1:
        lines.append("  (fixed spawn: absolute and relative coverage differ only by a shift)")

    if data["has_head"] and len(np.unique(data["heads"])) > 1:
        lines.append("")
        lines.append("per head  (specialisation: do the heads do different things?)")
        lines.append("  head   eps  contact   median push   lift   push direction (mean +- circular std)")
        for h in np.unique(data["heads"]):
            sel = data["heads"] == h
            ang = np.radians(m["push_angle_deg"][sel & c])
            if len(ang):
                mean_dir = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
                R = np.hypot(np.sin(ang).mean(), np.cos(ang).mean())
                spread = np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9))))
                direction = f"{mean_dir:+6.0f} deg +- {spread:3.0f}"
            else:
                direction = "n/a"
            lines.append(f"  {h:4d} {sel.sum():5d} {c[sel].mean():7.0%} "
                         f"{np.median(m['max_xy_mm'][sel]):12.1f} mm "
                         f"{m['lifted'][sel].mean():6.0%}   {direction}")
        lines.append("  (different mean directions with small spreads = the heads specialised;"
                     " similar directions = they converged)")

    lines.append("")
    lines.append("actions (mean |a|, max |a| per dim; max hitting the cap = saturation)")
    for i, name in enumerate(["dx", "dy", "dz", "drot", "dgrip"][:act.shape[1]]):
        a = np.abs(act[:, i])
        lines.append(f"  {name:6s} {a.mean():.4f}  {a.max():.4f}")

    lines.append("")
    lines.append("state ranges")
    for i, name in enumerate(["x", "y", "z", "rot", "grip"][:state.shape[1]]):
        lo, hi = state[:, i].min(), state[:, i].max()
        flag = "  <-- narrow" if hi - lo < 0.3 else ""
        lines.append(f"  {name:5s} {lo:.2f} .. {hi:.2f}{flag}")

    lines.append("")
    lines.append("sanity")
    bad = data["bad_eps"]
    lines.append(f"  episodes with missing/duplicate steps: {len(bad)}"
                 + (f"  e.g. {bad[:10]}" if bad else "  (should be 0)"))
    n_nonfinite = sum((~np.isfinite(a)).sum() for a in (state, obj, act))
    lines.append(f"  non-finite values: {n_nonfinite}")
    z = obj[:, 2] * 1000
    lines.append(f"  cube z: {z.min():.1f} .. {z.max():.1f} mm "
                 "(expect ~13.9, large = lifted or flipped)")
    lines.append(f"  time at arm low z (state z > 0.95): {(state[:, 2] > 0.95).mean():.3f}")
    return lines


def summary(data, m):
    """Headline numbers for summary.json."""
    c = m["contact"]
    disp_xy = displacements(data, 2)
    state = data["state"]
    spawns = np.array([o[0, :2] for o in data["episodes"]])
    s = {
        "dataset": data["path"],
        "episodes": len(data["episodes"]),
        "rows": len(data["obj"]),
        "contact_rate": c.mean(),
        "lift_rate": m["lifted"].mean(),
        "max_xy_mm_median": np.median(m["max_xy_mm"]),
        "max_xy_mm_p90": np.percentile(m["max_xy_mm"], 90),
        "final_xy_mm_mean": m["final_xy_mm"].mean(),
        "direction_entropy": direction_entropy(m["push_angle_deg"][c]),
        "straightness_median": np.median(m["straightness"][c]) if c.any() else None,
        "first_contact_step_median": np.median(m["first_contact_step"][c]) if c.any() else None,
        "disp_xy_coverage_pooled": DISP_XY.coverage_relative(disp_xy),
        "disp_xy_coverage_per_episode": DISP_XY.coverage_relative(disp_xy, pooled=False),
        "disp_xyz_coverage_pooled": DISP_XYZ.coverage_relative(displacements(data, 3)),
        "arm_xy_coverage": ARM_XY.coverage(state[:, :2]),
        "arm_xyz_coverage": ARM_XYZ.coverage(state[:, :3]),
        "cube_xy_coverage_abs": CUBE_XY.coverage(data["obj"][:, :2]),
        "cube_xy_coverage_spawns_only": CUBE_XY.coverage(spawns),
    }
    # plain Python floats rounded to 4 decimals, so json can write them
    for key, value in s.items():
        if value is not None and not isinstance(value, str):
            s[key] = round(float(value), 4)
    s["episodes"], s["rows"] = int(s["episodes"]), int(s["rows"])
    s["thresholds"] = {"contact_mm": CONTACT_MM, "lift_mm": LIFT_MM}
    return s


# ---------------------------------------------------------------- figure

def heatmap(ax, grid, points, title, scale=1.0):
    """Log-scale visit counts of a 2D grid. scale multiplies the axis labels (1000 = m -> mm)."""
    from matplotlib.colors import LogNorm
    counts = grid.counts(points)
    image = np.ma.masked_equal(counts, 0).T      # hide empty cells; transpose so rows = y
    x0, x1 = grid.bounds[0] * scale
    y0, y1 = grid.bounds[1] * scale
    im = ax.imshow(image, origin="lower", extent=[x0, x1, y0, y1], cmap="magma",
                   norm=LogNorm(vmin=1, vmax=max(counts.max(), 2)), aspect="equal")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label="visits (log)")


def plot(data, m, out):
    import matplotlib
    matplotlib.use("Agg")               # draw to a file, no screen needed
    import matplotlib.pyplot as plt

    episodes = data["episodes"]
    contacted = m["contact"]
    disp_mm = [(o - o[0]) * 1000 for o in episodes]
    fig = plt.figure(figsize=(22, 10.5))

    # 1. absolute top-down: spawns + paths of contacted episodes
    ax = fig.add_subplot(2, 4, 1)
    for o, hit in zip(episodes, contacted):
        if hit:
            ax.plot(o[:, 0] * 1000, o[:, 1] * 1000, lw=0.6, alpha=0.6)
    spawns_mm = np.array([o[0] for o in episodes]) * 1000
    colors = ["tab:red" if hit else "0.6" for hit in contacted]
    ax.scatter(spawns_mm[:, 0], spawns_mm[:, 1], s=8, c=colors, zorder=3)
    x0, x1 = CUBE_XY.bounds[0] * 1000
    y0, y1 = CUBE_XY.bounds[1] * 1000
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--"))
    ax.set_aspect("equal")
    ax.set(title="absolute: spawns (red = contacted) + paths", xlabel="x (mm)", ylabel="y (mm)")

    # 2. displacement paths: every contacted episode starts at the origin
    ax = fig.add_subplot(2, 4, 2)
    for d, hit in zip(disp_mm, contacted):
        if hit:
            ax.plot(d[:, 0], d[:, 1], lw=0.6, alpha=0.6)
            ax.plot(d[-1, 0], d[-1, 1], "k.", ms=3)
    ax.add_patch(plt.Circle((0, 0), CONTACT_MM, fill=False, color="tab:red"))
    ax.set_aspect("equal")
    ax.set(title="relative: displacement paths (contacted)", xlabel="dx (mm)", ylabel="dy (mm)")

    # 3. displacement heatmap, all episodes, all steps
    ax = fig.add_subplot(2, 4, 3)
    heatmap(ax, DISP_XY, np.concatenate(displacements(data, 2)),
            "relative: displacement visits (all)", scale=1000)
    ax.set(xlabel="dx (mm)", ylabel="dy (mm)")

    # 4. push directions (rose plot)
    ax = fig.add_subplot(2, 4, 4, projection="polar")
    angles = m["push_angle_deg"][contacted]
    counts, edges = np.histogram(angles, bins=N_DIR_BINS, range=(-180, 180))
    centres = np.radians(edges[:-1] + 180 / N_DIR_BINS)
    ax.bar(centres, counts, width=2 * np.pi / N_DIR_BINS, alpha=0.7, edgecolor="k")
    ax.set_title(f"final push direction (entropy {direction_entropy(angles):.2f})")

    # 5. max displacement per episode (log x, since untouched episodes are ~0)
    ax = fig.add_subplot(2, 4, 5)
    largest = max(m["max_xy_mm"].max(), m["max_abs_z_mm"].max(), 1.0)
    bins = np.logspace(-2, np.log10(largest) + 0.1, 40)
    ax.hist(np.clip(m["max_xy_mm"], 0.01, None), bins=bins, histtype="step", lw=1.5, label="planar")
    ax.hist(np.clip(m["max_abs_z_mm"], 0.01, None), bins=bins, histtype="step", lw=1.5, label="|z|")
    ax.axvline(CONTACT_MM, color="tab:red", ls="--", label="contact")
    ax.axvline(LIFT_MM, color="tab:purple", ls=":", label="lift")
    ax.set_xscale("log")
    ax.set(title="max displacement per episode", xlabel="mm", ylabel="episodes")
    ax.legend()

    # 6. 3D displacement paths
    ax = fig.add_subplot(2, 4, 6, projection="3d")
    for d, hit in zip(disp_mm, contacted):
        if hit:
            ax.plot(d[:, 0], d[:, 1], d[:, 2], lw=0.6, alpha=0.6)
    ax.set(title="relative: 3D displacement (contacted)", xlabel="dx", ylabel="dy", zlabel="dz (mm)")

    # 7. arm top-down heatmap
    ax = fig.add_subplot(2, 4, 7)
    heatmap(ax, ARM_XY, data["state"][:, :2], "arm xy visits (normalized)")
    ax.set(xlabel="x", ylabel="y")

    # 8. distance from spawn over time, and fraction of episodes contacted so far
    ax = fig.add_subplot(2, 4, 8)
    T = m["steps"].max()
    dist = []
    for d in disp_mm:
        r = np.linalg.norm(d[:, :2], axis=1)
        dist.append(np.pad(r, (0, T - len(r)), mode="edge"))   # repeat last value if shorter
    dist = np.array(dist)                                        # (episodes, T)
    t = np.arange(T)
    ax.fill_between(t, np.percentile(dist, 25, axis=0), np.percentile(dist, 75, axis=0),
                    alpha=0.3, label="p25-p75")
    ax.plot(t, np.median(dist, axis=0), label="median")
    ax.plot(t, np.percentile(dist, 90, axis=0), ls="--", label="p90")
    ax.set(title="planar displacement over time", xlabel="step", ylabel="mm")
    ax.legend(loc="upper left")

    first = m["first_contact_step"]
    frac_contacted = [np.mean((first >= 0) & (first <= step)) for step in t]
    ax2 = ax.twinx()
    ax2.plot(t, frac_contacted, color="tab:red")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("fraction contacted", color="tab:red")

    coverage = DISP_XY.coverage_relative(displacements(data, 2))
    fig.suptitle(f"{data['path']}   |   {len(episodes)} episodes   |   "
                 f"contact {contacted.mean():.0%}   |   disp-xy coverage (pooled) {coverage:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main

def save_csv(episode_ids, per_episode, path, heads=None):
    with open(path, "w", newline="") as f:
        fields = ["episode_idx"] + (["head_idx"] if heads is not None else []) + list(per_episode[0])
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i, (e, row) in enumerate(zip(episode_ids, per_episode)):
            row = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
            extra = {"head_idx": int(heads[i])} if heads is not None else {}
            writer.writerow({"episode_idx": int(e), **extra, **row})


def main(path, out_root="results/dataset_stats", make_plot=True):
    name = Path(str(path).rstrip("/")).stem          # "runs/push_v2.lance/" -> "push_v2"
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load(path)
    per_episode = [episode_stats(o) for o in data["episodes"]]
    # same numbers, one array per stat: m["max_xy_mm"] has one value per episode
    m = {key: np.array([row[key] for row in per_episode]) for key in per_episode[0]}

    text = "\n".join(report(data, m))
    print(text)
    (out_dir / "report.txt").write_text(text + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary(data, m), indent=2))
    save_csv(data["episode_ids"], per_episode, out_dir / "episodes.csv",
             heads=data["heads"] if data["has_head"] else None)
    if make_plot:
        plot(data, m, out_dir / "stats.png")
    print(f"\nresults -> {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out-dir", default="results/dataset_stats")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    main(args.dataset, args.out_dir, not args.no_plot)