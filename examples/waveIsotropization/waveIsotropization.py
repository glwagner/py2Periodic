import sys, time
import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.append('../../')
from py2Periodic.physics import hydrostaticWaveEqn_xy
from numpy import pi

# Load data
fileName = 'strongTurbulentInitialConditions.hdf5'
runName = 'ic_00'

dataFile = h5py.File(fileName, 'r')
turbData = dataFile[runName]

# Extract initial condition
q0 = turbData['q_snapshots']['q'][:, :, -1]

# Generate dictionary of parameters
turbParams = { param:value for param, value in turbData.attrs.iteritems() }

# Parameters:
Lx = turbParams['Lx']
alpha = 1
f0 = 1e-4
sigma = f0*np.sqrt(1+alpha)
kappa = 32.0*pi / (Lx*np.sqrt(alpha))
dt = 0.05 * 2.0*pi/f0
(waveVisc, waveViscOrder) = (1e24, 8.0)

# Add new parameters
addParams = {
    'meanVisc'      : turbParams['visc'], 
    'meanViscOrder' : turbParams['viscOrder'], 
    'waveVisc'      : waveVisc,
    'waveViscOrder' : waveViscOrder,
    'f0'          : f0,
    'sigma'       : sigma,
    'kappa'       : kappa, 
    'dt'          : dt, 
    'timeStepper' : 'ETDRK4', 
}

del turbParams['visc']
del turbParams['viscOrder']

params = turbParams.copy()
params.update(addParams)
    
# Initialize hydrostatic wave model
hwe = hydrostaticWaveEqn_xy.model(**params)
hwe.set_q(q0)

plt.figure('Initial condition'), plt.clf()
plt.imshow(hwe.q)
plt.show()

stopTime = 256.0*pi/f0
nSteps = int(np.ceil(stopTime / dt))

hwe.run(nSteps=nSteps, nPlots=128, nLogs=128)
