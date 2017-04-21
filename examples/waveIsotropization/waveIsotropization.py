import sys, time
import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.append('../py2Periodic/')
from py2Periodic.physics import hydrostaticWaveEqn_xy
from numpy import pi

# Parameters:
# * Physical
f0 = 1e-4
alpha = 1
sigma = f0*np.sqrt(1+alpha)
kappa = 32.0*pi / (Lx*np.sqrt(alpha))
# * Numerical
dt = 0.05 * 2.0*pi/f0
# * Frictional
(waveVisc, waveViscOrder) = (1e24, 8.0)

# Load data
fileName = 'strongTurbulentInitialConditions.hdf5'
runName = 'ic_00'

dataFile = h5py.File(fileName, 'r')
run = dataFile[runName]

# Generate dictionary of parameters
params = { param:value for param, value in run.attrs.iteritems() }

# Add new parameters
addParams = {
    'meanVisc' : params['visc'], 
    'meanViscOrder' : params['viscOrder'], 
    'waveVisc' : waveVisc,
    'waveViscOrder' : waveViscOrder,
    'f0' : f0,
    'sigma' : sigma,
    'kappa' = kappa, 
    'dt' : dt, 
    'timeStepper', 'ETDRK4', 
}

del params['visc']
del params['viscOrder']

params.update(addParams)
    
# Initialize hydrostatic wave model
hwe = hydrostaticWaveEqn_xy.model(**params)

hwe.set_q(q0)

# Run turbulence model
stopTime = 400.0*pi/f0
nSteps = int(np.ceil(stopTime / dt))
nPlots = 10
nSubsteps = int(np.ceil(nSteps/nPlots))

print("Iterating {:d} times and making {:d} "
        "plots:".format(nSteps, nPlots))
for i in xrange(nPlots):

    turb.run(nSteps=nSubsteps, nLogs=10)
    turb.update_state_variables()

    # Plot the result
    plt.figure('Flow', figsize=(12, 8)); plt.clf()
    plt.imshow(turb.q)
    plt.pause(0.01)

print("The root-mean-square vorticity is " + \
        "{:0.3f}.\n".format(np.sqrt((turb.q**2.0).mean())))

print("Finished the preliminary turbulence integration! " \
        "Now let's put a wave field in it...")

# Give final turbulence field to wave model, and run
hwe.set_q(turb.q)

stopTime = 200.0*pi/f0
nSteps = int(np.ceil(stopTime / dt))
nPlots = 100
nSubsteps = int(np.ceil(nSteps/nPlots))

print("Iterating {:d} times and making {:d} plots:".format(nSteps, nPlots))
for i in xrange(nPlots):
    
    hwe.run(nSteps=nSubsteps, nLogs=2)
    hwe.update_state_variables()
    
    plt.figure('Waves and flow', figsize=(12, 8)); plt.clf()
    plt.subplot(121); plt.imshow(hwe.q)
    plt.subplot(122); plt.imshow(np.sqrt(hwe.u**2.0 + hwe.v**2.0))
    plt.pause(0.01)
