"""Grid-based coverage metrics for N-dimensional workspaces."""

import numpy as np


class GridCoverage:
    """Discretize an N-d box into cells and measure how much of it is visited.

    bounds: [(lo, hi), ...]  one pair per dimension
    bins:   [n0, n1, ...]    cells per dimension, total = prod(bins)

    Works for any number of dimensions: 2 pairs for a top-down (x, y) grid,
    3 for (x, y, z).

    Example:
        cube = GridCoverage(bounds=[(-0.1, 0.1), (-0.08, 0.08)], bins=[40, 40])
        frac = cube.coverage(points)          # points: (N, 2)

    Points outside the bounds (or non-finite) are dropped rather than clipped
    onto the edge cells; use outside_fraction() to see how many there were.
    """

    def __init__(self, bounds, bins, verbose=False):
        self.bounds = np.asarray(bounds, dtype=float)       # (n_dims, 2)
        self.bins = np.asarray(bins, dtype=int)             # (n_dims,)
        if self.bounds.shape != (len(self.bins), 2):
            raise ValueError("need one (lo, hi) pair per entry in bins")
        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError(f"every bound needs hi > lo, got {self.bounds.tolist()}")
        if np.any(self.bins < 1):
            raise ValueError(f"bins must be >= 1, got {self.bins.tolist()}")
        self.n_dims = len(self.bins)
        self.total_cells = int(np.prod(self.bins))
        if verbose:
            print(f"GridCoverage: bounds {self.bounds.tolist()}, bins {self.bins.tolist()}")

    # ---- internals ---------------------------------------------------------

    def _check(self, points):
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return points.reshape(0, self.n_dims)
        if points.ndim != 2 or points.shape[1] != self.n_dims:
            raise ValueError(f"expected (N, {self.n_dims}) points, got {points.shape}")
        return points

    def _in_bounds(self, points):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        # NaN compares False, so non-finite points are excluded too
        return np.all((points >= lo) & (points <= hi), axis=1)

    def _to_cells(self, points):
        """(N, n_dims) in-bounds points -> (N,) flat cell indices."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        idx = np.floor((points - lo) / (hi - lo) * self.bins).astype(int)
        idx = np.clip(idx, 0, self.bins - 1)     # only moves points exactly on `hi`
        return np.ravel_multi_index(idx.T, self.bins)

    # ---- public ------------------------------------------------------------

    def cells(self, points):
        """Flat cell index of every in-bounds point."""
        points = self._check(points)
        return self._to_cells(points[self._in_bounds(points)])

    def outside_fraction(self, points):
        """Fraction of points that fell outside the grid (bounds too small?)."""
        points = self._check(points)
        return float(1.0 - self._in_bounds(points).mean()) if len(points) else 0.0

    def counts(self, points):
        """Visit count per cell, shaped like the grid: (bins[0], bins[1], ...)."""
        c = np.bincount(self.cells(points), minlength=self.total_cells)
        return c.reshape(self.bins)

    def coverage(self, points):
        """Fraction of cells visited at least once."""
        return len(np.unique(self.cells(points))) / self.total_cells

    def entropy(self, points):
        """Normalized Shannon entropy of visits, in [0, 1].

        1 = time spread uniformly over all cells, 0 = everything in one cell.
        Unlike coverage, this penalizes a cell visited once as much as it
        rewards it: it says whether visits are balanced.
        """
        c = self.counts(points).ravel().astype(float)
        if c.sum() == 0 or self.total_cells == 1:
            return 0.0
        p = c[c > 0] / c.sum()
        return float(-(p * np.log(p)).sum() / np.log(self.total_cells)) + 0.0  # +0.0 turns -0.0 into 0.0

    def coverage_relative(self, trajectories, pooled=True):
        """Coverage of displacements from each trajectory's own start.

        trajectories: iterable of (T_i, n_dims) arrays; lengths may differ.
        Bounds should be centred on 0, since these are displacements.

        pooled=True  -> union of cells over all trajectories: the variety of
                        outcomes in the dataset.
        pooled=False -> mean per-trajectory coverage: how much a typical
                        episode moves. A trajectory that never moves scores
                        exactly 1 / total_cells.

        Either way the spawn position cancels out.
        """
        if not self._in_bounds(np.zeros((1, self.n_dims)))[0]:
            raise ValueError("relative grid bounds must contain 0 (the start of every trajectory)")
        shifted = []
        for t in trajectories:
            t = self._check(t)
            if len(t) == 0:
                raise ValueError("empty trajectory")
            shifted.append(t - t[0])
        if not shifted:
            return 0.0
        if pooled:
            return self.coverage(np.concatenate(shifted)) # stacks all trajectories into one big set of displacements and asks how many cells were visited in total
        return float(np.mean([self.coverage(s) for s in shifted])) # asks how many cells were visited on average per trajectory

    def excess_coverage(self, trajectories):
        """Absolute coverage minus what the spawn positions alone already cover.

        The part of absolute coverage that was reached by moving the object.
        It is a lower bound, since cells that some other episode spawned in are
        not credited.
        """
        trajectories = [self._check(t) for t in trajectories]
        if not trajectories:
            return 0.0
        if any(len(t) == 0 for t in trajectories):
            raise ValueError("empty trajectory")
        # Get the starting positions of all trajectories
        starts = np.stack([t[0] for t in trajectories])
        return self.coverage(np.concatenate(trajectories)) - self.coverage(starts)