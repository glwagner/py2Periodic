import doublyPeriodic
import twoDimensionalTurbulence, hydrostaticWavesInXY, nearInertialWavesInXY
import time
import numpy as np
from numpy import pi, sin, cos, exp, sqrt
import matplotlib.pyplot as plt

## Create 2D turbulence model
#turbModel = twoDimensionalTurbulence.model(
#    nx = 128, 
#    Lx = 2.0*pi, 
#    dt = 5.0e-2,
#    nThreads = 4, 
#    timeStepper = "RKW3",
#)
#
#for ii in np.arange(0, 10):
#    turbModel.run_nSteps(nSteps=2e2, dnLog=1e2)
#
#    fig = plt.figure('vorticity', figsize=(6, 6))
#    plt.pcolormesh(turbModel.xx, turbModel.yy, turbModel.q, cmap='RdBu_r')
#    plt.axis('square')
#    plt.xlabel('$x$', labelpad=5.0)
#    plt.ylabel('$y$', labelpad=12.0)
#    plt.pause(0.1)
#waveModel.set_q(turbModel.q)

waveModel = nearInertialWavesInXY.model(
    nx = 128, 
    Lx = 2.0*pi, 
    dt = 1.0e-1,
    nThreads = 4, 
)

waveModel.set_A(exp(4j*waveModel.XX))

fig = plt.figure('waves', figsize=(6, 6))
plt.pcolormesh(waveModel.xx, waveModel.yy, np.real(waveModel.A), cmap='RdBu_r')

plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)

plt.pause(5)

# Run a loop
for ii in np.arange(0, 1e2):

    waveModel.run_nSteps(nSteps=1e1, dnLog=1e1)
    waveModel.update_state_variables()

    plt.pcolormesh(waveModel.xx, waveModel.yy, np.real(waveModel.A), cmap='YlGnBu_r')
    plt.pause(0.1)

print("Close the plot to end the program")
plt.show()
