import sys; sys.path.append('../py2Periodic/')
import twoDimTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

# Instantiate a model for two-dimensional turbulence.
turb = twoDimTurbulence.model(
    nx = 128, 
    Lx = 2.0*pi, 
    dt = 1.0e-1,
    nThreads = 1, 
    timeStepper = 'AB3',
    visc = 1.0e-4, 
    viscOrder = 4.0, 
)

turb.describe_model()

# Set an initial random vorticity field. The model attribute "physSolnShape"
# is a tuple that gives the shape of the "soln" variable used internally
# by the model.
q0 = np.random.standard_normal((turb.ny, turb.nx))
turb.set_q(q0)

# Step the model forward in time
turb.step_nSteps(nSteps=1e3, dnLog=1e2)

# Update variables like vorticity, speed, etc
turb.update_state_variables()

print("The root-mean-square vorticity is " + \
        "{:0.3f}".format(np.sqrt((turb.q**2.0).mean())))

# Plot the result
fig = plt.figure('vorticity', figsize=(6, 6)); plt.clf()
plt.pcolormesh(turb.x, turb.y, turb.q, cmap='YlGnBu_r'); plt.axis('square') 
plt.xlabel('$x$', labelpad=5.0); plt.ylabel('$y$', labelpad=12.0)

print("\nClose the figure to end the program.")
plt.show()
