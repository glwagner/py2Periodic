import sys; sys.path.append('../../py2Periodic/')
import twoDimTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

m = twoDimTurbulence.model(
    nx = 128, 
    Lx = 2.0*pi, 
    dt = 1.0e-1,
    nThreads = 1, 
    timeStepper = 'AB3',
    visc = 1.0e-7, 
    viscOrder = 4.0, 
)

m.set_q(np.random.standard_normal((m.ny, m.nx)))

m.run(nSteps=1e3, logInterval=1e2, nSaves=100)
m.update_state_variables()

# Plot the result
fig = plt.figure('vorticity', figsize=(6, 6)); plt.clf()

plt.pcolormesh(m.x, m.y, m.q, cmap='YlGnBu_r'); plt.axis('square') 

plt.xlabel('$x$', labelpad=5.0); 
plt.ylabel('$y$', labelpad=12.0)
plt.colorbar()
