import numpy as np
from plotRawGPS import plotRawGPS, LOG_WIDTH

data = np.reshape(np.fromfile('/tmp/ceres_sandbox/CarrierPhase.Trajectory.log', dtype=np.float64), (-1, LOG_WIDTH))

plotRawGPS(data, "CPhase, Traj")