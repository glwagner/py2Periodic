import time, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from py2Periodic.physics import twoDimTurbulence
from numpy import pi

# Instantiate a model for two-dimensional turbulence.
turb = twoDimTurbulence.model(
    nx = 128, 
    Lx = 2.0*pi, 
    dt = 1e-2,
    nThreads = 1, 
    timeStepper = 'RK4',
    visc = 1e-4, 
    viscOrder = 2.0, 
)

turb.describe_model()

# Set an initial random vorticity field.
q0 = np.random.standard_normal((turb.ny, turb.nx))
turb.set_q(q0)

# Step the model forward in time
turb.step_nSteps(nSteps=1e4, dnLog=1e3)

# Update variables like vorticity, velocity, etc
turb.update_state_variables()

print("The root-mean-square vorticity is " + \
        "{:0.3f}".format(np.sqrt((turb.q**2.0).mean())))

# Plot the result
fig = plt.figure('vorticity'); plt.clf()

plt.pcolormesh(turb.x, turb.y, turb.q, cmap='YlGnBu_r')
plt.axis('square') 

plt.xlabel('$x$')
plt.ylabel('$y$')

print("\nClose the figure to end the program.")
plt.show()
