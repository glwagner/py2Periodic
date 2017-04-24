import sys, time
import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.append('../../')
from py2Periodic.physics import hydrostaticWaveEqn_xy
from numpy import pi

# Parameters:
Lx = 1600e3
(alpha, f0) = (1, 1e-4)
sigma = f0*np.sqrt(1+alpha)
kappa = 32.0*pi / (Lx*np.sqrt(alpha))

dt = 0.05 * 2.0*pi/f0
(waveVisc, waveViscOrder) = (1e24, 8.0)

# Add new parameters
params = {
    'waveVisc'      : waveVisc,
    'waveViscOrder' : waveViscOrder,
    'f0'          : f0,
    'sigma'       : sigma,
    'kappa'       : kappa, 
    'dt'          : dt, 
    'timeStepper' : 'ETDRK4', 
}

hwe = hydrostaticWaveEqn_xy.init_from_turb_endpoint(
        'strongTurbData.hdf5', 'ic_06', **params)

plt.figure('Initial condition'), plt.clf()
plt.imshow(hwe.q)
plt.show()

stopTime = 256.0*pi/f0
nSteps = int(np.ceil(stopTime / dt))

hwe.run(nSteps=nSteps, nPlots=128, nLogs=128)
