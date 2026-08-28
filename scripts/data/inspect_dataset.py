"""Summary statistics for a collected Lance dataset.

Usage:
    uv run python scripts/data/inspect_dataset_stats.py <path>.lance
"""

import argparse
import lance
import numpy as np

from scripts.data.coverage import GridCoverage

ARM = GridCoverage(bounds=[(0.0, 1.0), (0.0, 1.0)], bins=[20, 20])
CUBE = GridCoverage(bounds=[(-0.1, 0.1), (-0.08, 0.08)], bins=[20, 20])
CONTACT_MM = 2.0 #Threshold of movement to consider the cube "contacted" (in mm).  The cube is 20mm wide, so this is 10% of its width.


def main(path):
    ds = lance.dataset(path)
    t = ds.to_table(columns=["episode_idx", "step_idx", "state",
                             "object_position", "action"]).to_pydict()
    ep = np.array(t["episode_idx"]) 
    print(ep)
    state = np.array(t["state"])
    obj = np.array(t["object_position"])
    act = np.array(t["action"])
    eps = np.unique(ep)

    print(f"{path}")
    print(f"{len(eps)} episodes, {len(ep)} rows, "
          f"{len(ep) // max(len(eps), 1)} steps/episode\n")

    # --- contact ---
    moved = np.array([np.abs(obj[ep == e] - obj[ep == e][0]).max() * 1000
                      for e in eps]) #array of the maximum displacement of the cube in each episode, in mm
    print("cube displacement")
    print(f"  contact (>{CONTACT_MM}mm): {(moved > CONTACT_MM).sum()}/{len(eps)}"
          f"  ({(moved > CONTACT_MM).mean():.0%})")
    print(f"  mean {moved.mean():.1f}mm   median {np.median(moved):.1f}mm"
          f"   max {moved.max():.1f}mm\n")

    # --- coverage ---
    print("coverage (fraction of 400 cells)")
    print(f"  arm  {ARM.coverage(state[:, :2]):.3f}")
    print(f"  cube {CUBE.coverage(obj[:, :2]):.3f}\n")

    # --- spawn variety: are episodes actually different? ---
    starts = np.array([obj[ep == e][0][:2] for e in eps])
    print(f"spawn positions: {len(np.unique(starts.round(4), axis=0))}/{len(eps)} distinct")

    # --- action usage ---
    names = ["dx", "dy", "dz", "drot", "dgrip"]
    print("\nactions (mean |a| per dim, cap is max_delta)")
    print("  " + "  ".join(f"{n} {np.abs(act[:, i]).mean():.4f}"
                           for i, n in enumerate(names)))

    # --- state ranges: did any axis stay stuck? ---
    print("\nstate ranges")
    for i, n in enumerate(["x", "y", "z", "rot", "grip"]):
        lo, hi = state[:, i].min(), state[:, i].max()
        flag = "  <-- narrow" if hi - lo < 0.3 else ""
        print(f"  {n:5s} {lo:.2f} .. {hi:.2f}{flag}")

    # --- sanity ---
    print("\nsanity")
    print(f"  non-finite values: {(~np.isfinite(state)).sum() + (~np.isfinite(act)).sum()}")
    z = obj[:, 2]
    print(f"  cube z: {z.min()*1000:.1f} .. {z.max()*1000:.1f} mm "
          f"(expect ~13.9, large = lifted or flipped)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    main(p.parse_args().dataset)