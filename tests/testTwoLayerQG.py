import sys; sys.path.append('../py2Periodic/')
import twoLayerQuasigeostrophic
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

params = { 
    'Lx'         : 1.0e6, 
    'beta'       : 1.0e-11,
    'defRadius'  : 1.5e4, 
    'H1'         : 500, 
    'H2'         : 2000, 
    'U1'         : 1.0e-1, 
    'U2'         : 0.0, 
    'drag'       : 1.0e-6, 
    'visc'       : 1.0e6, 
    'viscOrder'  : 4.0, 
    'nx'         : 256, 
    'dt'         : 1.0e4, 
    'timeStepper': 'ETDRK4', 
    'nThreads'   : 8, 
}

myParams = { 
    'Lx'         : 1.0e6, 
    'beta'       : 1.5e-11, 
    'defRadius'  : 1.5e4, 
    'H1'         : 500.0, 
    'H2'         : 2000.0, 
    'U1'         : 2.5e-2, 
    'U2'         : 0.0,
    'drag'       : 1.0e-7,
    'nx'         : 256,
    'dt'         : 1.0e4, 
    'visc'       : 1.0e6, 
    'viscOrder'  : 4.0, 
    'timeStepper': 'RK4', 
    'nThreads'   : 8,
}


# Create the two-layer model
qg = twoLayerQuasigeostrophic.model(**myParams)
qg.describe_model()

# Initial condition: 
Ro = 1.0e-2
f0 = 1.0e-4
q1 = Ro*f0*np.random.standard_normal(qg.physVarShape)
q2 = Ro*f0*np.random.standard_normal(qg.physVarShape)

qg.set_q1_and_q2(q1, q2)

# Run a loop
nt = 1e3
for ii in np.arange(0, 1e3):

    qg.run_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    fig = plt.figure('Perturbation vorticity', figsize=(8, 4)); plt.clf()
    plt.subplot(121); plt.imshow(qg.q1)
    plt.subplot(122); plt.imshow(qg.q2)
    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
