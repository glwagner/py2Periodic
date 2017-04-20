import time, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from py2Periodic.physics import twoDimTurbulence
from numpy import pi
from exampleModuleInput import paramsInFile

# Parameters can be defined in a dictionary variable to be passed
# as input when the model is instantiated, or passed directly on 
# on instantiation. The dictionary variable is convenient as it allows
# sets of parameters to be swapped and stored easily.
paramsInScript = {
    'nx'         : 256, 
    'Lx'         : 2.0*pi, 
    'dt'         : 5.0e-2,
    'visc'       : 1.0e-7, 
    'viscOrder'  : 4.0, 
    'nThreads'   : 2, 
    'timeStepper': 'RK4',
}

# Instantiate the model. There are two possible inputs, the keyword
# argument dictionary "paramsInScript" defined above, or the keyword
# argument dictionary "paramsInFile" which was loaded as a module in the
# script's header.
m = twoDimTurbulence.model(**paramsInScript)
#m = twoDimTurbulence.model(**paramsInFile)
m.describe_model()

# Define initial condition
q0 = np.random.standard_normal(m.physSolnShape)
m.set_physical_soln(q0)

# Run the model
m.step_nSteps(nSteps=4e3, dnLog=1e2)

# Update variables like vorticity, u and v, etc
m.update_state_variables()

print("The root-mean-square vorticity is " + \
        "{:0.3f}".format(np.sqrt((m.q**2.0).mean())))

# Plot the result
fig = plt.figure('vorticity'); plt.clf()
plt.pcolormesh(m.x, m.y, m.q, cmap='YlGnBu_r'); plt.axis('square') 
plt.xlabel('$x$', labelpad=5.0); plt.ylabel('$y$', labelpad=12.0)

print("\nClose plot to end the problem.")
plt.show()
