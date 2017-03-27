import sys; sys.path.append('../py2Periodic/')
import twoDimensionalTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

# Generate a model for two-dimensional turbulence
m = twoDimensionalTurbulence.model(
    nx = 512, 
    Lx = 2.0*pi, 
    dt = 1.0e-2,
    nThreads = 8, 
    timeStepper = 'forwardEuler',
    visc = 1.0e-5, 
    viscOrder = 4.0, 
)

m.describe_model()

# Define initial condition
q0 = np.random.standard_normal(m.physSolnShape)
m.set_physical_soln(q0)

# Run the model
m.run_nSteps(nSteps=1e4, dnLog=1e3)

# Update variables like vorticity, speed, etc
m.update_state_variables()

print("Root-mean-square vorticity = " + \
        "{:0.3f}".format(np.sqrt((m.q**2.0).mean())))

# Plot the result
fig = plt.figure('vorticity', figsize=(6, 6)); plt.clf()
plt.pcolormesh(m.xx, m.yy, m.q, cmap='YlGnBu_r'); plt.axis('square') 
plt.xlabel('$x$', labelpad=5.0); plt.ylabel('$y$', labelpad=12.0)

print("\nClose the figure to end the program")
plt.show()
