import doublyPeriodic
import twoLayerQuasigeostrophic
import time
import numpy as np
from numpy import pi, sin, cos, exp, sqrt
import matplotlib.pyplot as plt

# f0 is either 4.176e-3 or 1.744e-5...
# Create the two-layer model
#f0 = 4.176e-3
f0 = 1.744e-5
dt = 8000
qg = twoLayerQuasigeostrophic.model(
    Lx = 1.0e6,
    nx = 128, 
    dt = dt,
    nThreads = 4,
    timeStepper = 'RKW3',
    f0 = f0,
    beta = 1.5e-11,
    g = 9.81,
    H1 = 500,
    H2 = 2000,
    U1 = 2.5e-2,
    U2 = 0.0,
    drag = 5.787e-7,
    visc = 1.0e-6, 
    viscOrder = 4.0,
)
qg.describe_model()

# Initial condition: zeros in q2, random weak-global, strong-zonal
# perturbations in q1 
q1i = 1.0e-7*np.random.rand(qg.ny, qg.nx) \
    + 1.0e-6*( np.ones((qg.ny, 1))*np.random.rand(1, qg.nx) )
q2i = np.zeros(qg.physVarShape)

qg.set_q1(q1i)
qg.set_q2(q2i)

fig = plt.figure('vorticity', figsize=(6, 6)); plt.clf() 
plt.imshow(qg.q1 + qg.Qy1*qg.YY)
plt.clim([0, qg.Qy1*qg.Ly])
plt.pause(0.01), plt.draw()

# Run a loop
nt = 1e3
for ii in np.arange(0, 1e3):

    qg.run_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    plt.clf()
    plt.imshow(qg.q1 + qg.Qy1*qg.YY)
    plt.clim([0, qg.Qy1*qg.Ly])
    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
