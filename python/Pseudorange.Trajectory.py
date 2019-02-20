import numpy as np
from plotRawGPS import plotRawGPS, LOG_WIDTH

data = np.reshape(np.fromfile('/tmp/ceres_sandbox/Pseudorange.Trajectory.log', dtype=np.float64), (-1, LOG_WIDTH))

plotRawGPS(data, "Prange, Traj")