import sys, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from py2Periodic.physics import twoDimTurbulence
from numpy import pi

# Initialize model
f0 = 1e-4
Lx = 1600e3
dt = 0.1 * 2.0*pi/f0

stopTime = 200.0*2.0*pi/f0
itemsToSave = {'q' : stopTime}

params = {
    'nx' : 256,
    'Lx' : Lx,
    'dt' : dt,
    'visc' : 3e8,
    'viscOrder' : 4.0,
    'timeStepper' : 'RK4',
    'nThreads' : 8,
    'name' : 'strongTurbData'
}

turb = twoDimTurbulence.model(**params)

for i in xrange(3):

    strongIC = turb.random_energy_spectrum(q0rms=0.18*f0, kPeak=64.0)
    turb.set_q(strongIC)

    turb.run(nSteps=np.ceil(stopTime/dt), nLogs=10, 
        runName='ic_{:02d}'.format(i), 
        saveEndpoint=True, saveEndpointVars='q')

    turb.update_state_variables()
    maxRo = np.abs(turb.q).max() / f0
    print("\nMaximum Rossby number: {:3f}".format(maxRo))
