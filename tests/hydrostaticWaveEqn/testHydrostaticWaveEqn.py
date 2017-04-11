import sys; sys.path.append('../../py2Periodic/')
import hydrostaticWaveEqn_xy
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

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

# Initialize plot
fig, axArr = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
fig.canvas.set_window_title("Waves and flow")

# Step the model forward in time with default initial conditions.
for i in xrange(100):

    m.step_nSteps(nSteps=1e1, dnLog=1e1)
    m.update_state_variables()

    axArr[0].pcolormesh(m.q)
    axArr[1].pcolormesh(np.sqrt(m.u**2.0+m.v**2.0))

    plt.pause(0.01)

print("The root-mean-square vorticity is " + \
        "{:0.3f}".format(np.sqrt((m.q**2.0).mean())))

print("\nClose the figure to end the program.")
plt.show()
