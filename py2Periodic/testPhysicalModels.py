import doublyPeriodic
import twoDimensionalTurbulence, hydrostaticWavesInXY
import time
import numpy as np
from numpy import pi, sin, cos, exp, sqrt
import matplotlib.pyplot as plt

# Create 2D turbulence model
turbModel = twoDimensionalTurbulence.model(
    nThreads = 4, 
    timeStepper = "RKW3",
)

waveModel = hydrostaticWavesInXY.model(
    nThreads = 4, 
)

waveModel.describe_model()

fig = plt.figure('vorticity', figsize=(6, 6))
plt.pcolormesh(waveModel.xx, waveModel.yy, waveModel.uu, cmap='RdBu_r')

plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)

plt.pause(1)

# Run a loop
for ii in np.arange(0, 20):

    waveModel.run_nSteps(nSteps=1e2, dnLog=1e1)
    waveModel.update_state_variables()

    #plt.pcolormesh(waveModel.xx, waveModel.yy, waveModel.uu, cmap='RdBu_r')
    plt.pcolormesh(waveModel.xx, waveModel.yy, waveModel.sp, cmap='YlGnBu_r')
    plt.pause(0.1)

print("Close the plot to end the program")
plt.show()
