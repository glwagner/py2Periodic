import sys, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from py2Periodic.physics import twoDimTurbulence
from numpy import pi

# Initialize model
f0 = 1e-4
Lx = 1e6
dt = 0.1 * 2.0*pi/f0

params = {
    'nx' : 256,
    'Lx' : Lx,
    'dt' : dt,
    'visc' : 3e8,
    'viscOrder' : 4.0,
    'timeStepper' : 'RK4',
    'nThreads' : 2,
    'name' : 'turbulentInitialConditions'
}

turb = twoDimTurbulence.model(**params)

# Generate initial conditions
initialConditions = {
    'strong_1' : turb.random_energy_spectrum(q0rms=0.20*f0, kPeak=32.0),
    'medium_1' : turb.random_energy_spectrum(q0rms=0.10*f0, kPeak=32.0),
    'weak_1'   : turb.random_energy_spectrum(q0rms=0.07*f0, kPeak=32.0),
}

stopTimes = {
    'strong_1' : 200.0*2.0*pi/f0, 
    'medium_1' : 400.0*2.0*pi/f0, 
    'weak_1'   : 600.0*2.0*pi/f0, 
}

# Run turbulence models
for runName, q0 in initialConditions.iteritems():
    turb.set_q(q0)

    turb.run(nSteps=np.ceil(stopTimes[runName]/dt), 
        nLogs=10, nPlots=10, nSnaps=10, runName=runName)

    turb.update_state_variables()
    maxRo = np.abs(turb.q).max() / f0
    print("\nMaximum Rossby number: {:3f}".format(maxRo))
