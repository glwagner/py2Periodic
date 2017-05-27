import time, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from py2Periodic.physics import twoDimTurbulence
from py2Periodic.physics import twoDimTurbulence_ne1
from py2Periodic.physics import twoDimTurbulence_ne2
from numpy import pi

nSteps = 1e3
dnLog = 1e2

params = {
    'nx'          : 512, 
    'Lx'          : 2.0*pi, 
    'dt'          : 1e-2,
    'nThreads'    : 8, 
    'timeStepper' : 'RK4',
    'visc'        : 1e-4, 
    'viscOrder'   : 2.0, 
}

# Instantiate a model for two-dimensional turbulence.
turb0 = twoDimTurbulence.model(**params)
turb1 = twoDimTurbulence_ne1.model(**params)
turb2 = twoDimTurbulence_ne2.model(**params)

# Set an initial random vorticity field.
q0 = np.random.standard_normal((turb0.ny, turb0.nx))
turb0.set_q(q0)
turb1.set_q(q0)

# Step the model forward in time
turb2.step_nSteps(nSteps=nSteps, dnLog=dnLog)
turb2.update_state_variables()

# Plot the result
fig = plt.figure('vorticity'); plt.clf()

plt.pcolormesh(turb2.x, turb2.y, turb2.q, cmap='YlGnBu_r')
plt.axis('square') 

plt.xlabel('$x$')
plt.ylabel('$y$')

print("\nClose the figure to end the program.")
plt.show()

# Step the model forward in time
turb0.step_nSteps(nSteps=nSteps, dnLog=dnLog)
turb0.update_state_variables()

# Plot the result
fig = plt.figure('vorticity'); plt.clf()

plt.pcolormesh(turb0.x, turb0.y, turb0.q, cmap='YlGnBu_r')
plt.axis('square') 

plt.xlabel('$x$')
plt.ylabel('$y$')

print("\nClose the figure to end the program.")
plt.show()
