import numpy as np

class GridCoverage: 
    """ 
    Measures what fraction of a discretized workspace is visited
    Usage example: 
    pusher = GridCoverage(bounds=[(-0.1,0.1), (-0.08,0.08)], bins=[20,20])
    frac = pusher.coverage(points)

    bounds: define the region we care about, ex the arm's reachable area
    bins: how many bins to discretize each dimension into, ex [20, 20] means 20 bins in x and 20 bins in y, for a total of 400 cells
    """

    def __init__(self, bounds, bins):
        self.bounds = np.asarray(bounds) #becomes 2x2 array
        self.bins = list(bins)
        self.n_dims = len(self.bins)
        self.total_cells = int(np.prod(self.bins))

    def _to_cells(self, points):
        """convert points to cell indices, flattening the multi-dimensional grid into a 1D array of cell indices
           points: Nxn_dims array of points"""
        idx = []
        for d in range(self.n_dims):
            lo, hi = self.bounds[d] #get the lower and upper bounds for dimension d
            idx_d = np.floor((points[:, d] - lo) / (hi - lo) * self.bins[d]) #shifts, scales to [0,1] and then scales to [0, bins[d]]
            idx.append(np.clip(idx_d, 0, self.bins[d] - 1).astype(int))
        flat = np.zeros(len(points), dtype=int) 
        for d in range(self.n_dims): #every (x,y) point is mapped to a single cell index in the flattened grid
            flat *= self.bins[d]
            flat += idx[d]
        return flat

    def coverage(self, points):
        """Fraction of cells visited at least once"""
        return len(np.unique(self._to_cells(points))) / self.total_cells

    
    def coverage_relative(self, trajectories):
        """Mean per-trajectory coverage of displacement from each start.

        trajectories: (n_traj, T, n_dims) — one array per rollout.

        Each trajectory is shifted so its first point sits at the origin,
        so the result measures how far the policy moved things, not where
        they happened to start. A trajectory that never moves scores
        1/total_cells regardless of its starting position.
        """
        fracs = []
        for traj in trajectories: #each traj is a (T, n_dims) array
            shifted = traj - traj[0] #every position becomes a displacement from the starting position
            fracs.append(len(np.unique(self._to_cells(shifted))) / self.total_cells)
        return float(np.mean(fracs)) #average over all trajectories
            
