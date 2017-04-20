import time, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from py2Periodic.physics import twoLayerQG
from numpy import pi

params = { 
    'f0'         : 1.0e-4,
    'Lx'         : 1.0e6, 
    'beta'       : 1.5e-11, 
    'defRadius'  : 1.5e4, 
    'H1'         : 500.0, 
    'H2'         : 2000.0, 
    'U1'         : 2.5e-2, 
    'U2'         : 0.0,
    'bottomDrag' : 1.0e-7,
    'nx'         : 128,
    'dt'         : 1.0e3, 
    'visc'       : 2.0e8, 
    'viscOrder'  : 4.0, 
    'timeStepper': 'AB3', 
    'nThreads'   : 4,
    'useFilter'  : False,
}

# Create the two-layer model
qg = twoLayerQG.model(**params)
qg.describe_model()

# Initial condition: 
Ro = 1.0e-3
f0 = 1.0e-4

q1 = Ro*f0*np.random.standard_normal(qg.physVarShape)
q2 = Ro*f0*np.random.standard_normal(qg.physVarShape)

qg.set_q1_and_q2(q1, q2)

# Run a loop
nt = 1e3
for ii in np.arange(0, 1e3):

    qg.step_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    fig = plt.figure('Perturbation vorticity', figsize=(8, 8)); plt.clf()

    plt.subplot(221); plt.imshow(qg.q1)
    plt.subplot(222); plt.imshow(qg.q2)

    plt.subplot(223); plt.imshow(np.abs(qg.soln[0:qg.ny//2, :, 0]))
    plt.subplot(224); plt.imshow(np.abs(qg.soln[0:qg.ny//2, :, 1]))

    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
