import sys; sys.path.append('../py2Periodic/')
import twoDimensionalTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt
from exampleModuleInput import paramsInFile

# Parameters can be defined in a dictionary variable to be passed
# as input when the model is instantiated, or passed directly on 
# on instantiation. The dictionary variable is convenient as it allows
# sets of parametes to be swapped and stored easily.
paramsInScript = {
    'nx'         : 256, 
    'Lx'         : 2.0*pi, 
    'dt'         : 1.0e-2,
    'visc'       : 1.0e-4, 
    'viscOrder'  : 4.0, 
    'nThreads'   : 4, 
    'timeStepper': 'forwardEuler',
}

# Instantiated the model. There are two possible inputs, the keyword
# argument dictionary "paramsInScript" defined above, or the keyword
# argument dictionary "paramsInFile" which was loaded as a module in the
# script's header.
m = twoDimensionalTurbulence.model(**paramsInScript)

m.describe_model()
raw_input("Press enter to continue.")

# Define initial condition
q0 = np.random.standard_normal(m.physSolnShape)
m.set_physical_soln(q0)

# Run the model
m.run_nSteps(nSteps=4e3, dnLog=1e2)

# Update variables like vorticity, u and v, etc
m.update_state_variables()

print("Root-mean-square vorticity = " + \
        "{:0.3f}".format(np.sqrt((m.q**2.0).mean())))

# Plot the result
fig = plt.figure('vorticity', figsize=(6, 6)); plt.clf()
plt.pcolormesh(m.xx, m.yy, m.q, cmap='YlGnBu_r'); plt.axis('square') 
plt.xlabel('$x$', labelpad=5.0); plt.ylabel('$y$', labelpad=12.0)

plt.pause(0.01)
