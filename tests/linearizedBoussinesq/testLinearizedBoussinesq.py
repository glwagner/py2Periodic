import time, sys; sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt

from py2Periodic.physics import linearizedBoussinesq_xy
from numpy import pi

f0 = 1e-4
alpha = 3
sigma = f0*np.sqrt(1+alpha)
Lx = 1e6
k1 = 2.0*pi/Lx
kappa = 16.0*k1 / np.sqrt(alpha)

params = { 
    'nx'            : 128,
    'dt'            : 0.02*2.0*pi/f0,
    'f0'            : f0,
    'kappa'         : kappa,
    'Lx'            : Lx,
    'meanVisc'      : 1e8, 
    'meanViscOrder' : 4.0, 
    'timeStepper'   : 'RK4',
    'nThreads'      : 2,
}

# Instantiate a model for hydrostatic waves in two-dimensional turbulence.
m = linearizedBoussinesq_xy.model(**params)
m.describe_model()

# Step the model forward in time with default initial conditions.
nSteps = int(100*2.0*pi/(m.dt*m.f0))
m.run(nSteps=nSteps, nLogs=100, nPlots=100)

print("The root-mean-square vorticity is " + \
        "{:0.3f}".format(np.sqrt((m.q**2.0).mean())))

print("\nClose the figure to end the program.")
plt.show()
