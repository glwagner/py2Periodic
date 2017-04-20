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
    'nx'            : 128,
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
m = hydrostaticWaveEqn_xy.model(**params)
m.describe_model()

m.run(nSteps=1e3, nLogs=1e1, nPlots=1e1)

print("The root-mean-square vorticity is " + \
        "{:0.3f}".format(np.sqrt((m.q**2.0).mean())))

print("\nClose the figure to end the program.")
plt.show()
