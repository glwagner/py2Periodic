import time, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from py2Periodic.physics import hydrostaticWaveEqn_xy
from numpy import pi

f0 = 1e-4
alpha = 3
sigma = f0*np.sqrt(1+alpha)
Lx = 1e6
k1 = 2.0*pi/Lx
kappa = 16.0*k1 / np.sqrt(alpha)

params = { 
    'nx'            : 512,
    'dt'            : 0.1*2.0*pi/f0,
    'f0'            : f0,
    'sigma'         : sigma,
    'Lx'            : Lx,
    'kappa'         : kappa,
    'meanVisc'      : 1e8, 
    'meanViscOrder' : 4.0, 
    'waveVisc'      : 1e4, 
    'waveViscOrder' : 4.0, 
    'nThreads'      : 2,
}

# Instantiate a model for hydrostatic waves in two-dimensional turbulence.
mNe = hydrostaticWaveEqn_xy.model(timeStepper='ETDRK4_ne', **params)
mOr = hydrostaticWaveEqn_xy.model(timeStepper='ETDRK4', **params)

mNe.describe_model()
mNe.run(nSteps=1e2, nLogs=1e1)

mOr.describe_model()
mOr.run(nSteps=1e2, nLogs=1e1)

print("\nClose the figure to end the program.")
plt.show()
