import time, sys
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append('../../')
from py2Periodic.physics import twoDimTurbulence
from py2Periodic.physics import twoDimTurbulence_ne
from numpy import pi

nSteps = 2e2

for nx in [128, 256, 512, 1024]:
    for nThreads in [2]:

        print(  "Initializing and solving models with "
                "nThreads = {:d} and nx = {:d}".format(nThreads, nx))


        # Instantiate a model for two-dimensional turbulence.
        turb = twoDimTurbulence.model(
                nThreads=nThreads, nx=nx, timeStepper='RK4')
                
        turb_ne = twoDimTurbulence_ne.model(
                nThreads=nThreads, nx=nx, timeStepper='RK4_ne')


        # Set an initial random vorticity field.
        q0 = np.random.standard_normal((turb.ny, turb.nx))
        turb.set_q(q0)
        turb_ne.set_q(q0)

        # Time model *without* numexpr
        start0 = time.time()
        turb.step_nSteps(nSteps=nSteps)
        time0 = time.time() - start0

        # Time model *with* numexpr
        start_ne = time.time()
        turb_ne.step_nSteps(nSteps=nSteps)
        time_ne = time.time() - start_ne


        print("Without numexpr: {:.3e} secs".format(time0))
        print("With numexpr:    {:.3e} secs\n".format(time_ne))



# Plot the result
#turb2.update_state_variables()
#turb0.update_state_variables()
#
#fig = plt.figure('vorticity'); plt.clf()
#
#plt.pcolormesh(turb2.x, turb2.y, turb2.q, cmap='YlGnBu_r')
#plt.axis('square') 
#
#plt.xlabel('$x$')
#plt.ylabel('$y$')
#
#print("\nClose the figure to end the program.")
#plt.show()
#
## Plot the result
#fig = plt.figure('vorticity'); plt.clf()
#
#plt.pcolormesh(turb0.x, turb0.y, turb0.q, cmap='YlGnBu_r')
#plt.axis('square') 
#
#plt.xlabel('$x$')
#plt.ylabel('$y$')
#
#print("\nClose the figure to end the program.")
#plt.show()
