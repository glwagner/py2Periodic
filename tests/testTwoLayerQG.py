import sys; sys.path.append('../py2Periodic/')
import twoLayerQuasigeostrophic
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

params = { 
    'Lx'         : 1.0e7, 
    'f0'         : 1.4e-4, 
    'beta'       : 1.0e-11,
    'g'          : 9.81, 
    'H1'         : 200, 
    'H2'         : 200, 
    'U1'         : 1.0e0, 
    'U2'         : 1.0e-1, 
    'drag'       : 0.0e-6, 
    'visc'       : 1.0e10, 
    'viscOrder'  : 4.0, 
    'nx'         : 512, 
    'dt'         : 1.0e4, 
    'timeStepper': 'ETDRK4', 
    'nThreads'   : 8, 
}

# Create the two-layer model
qg = twoLayerQuasigeostrophic.model(**params)
qg.describe_model()

# Initial condition: 
# Zeros in q2, random weak-global, strong-zonal perturbations in q1 
Ro = 1.0e-1
q1 = Ro*params['f0']*np.random.standard_normal(qg.physVarShape)
q2 = Ro*params['f0']*np.random.standard_normal(qg.physVarShape)

qg.set_q1_q2(q1, q2)

# Run a loop
nt = 1e2
for ii in np.arange(0, 1e3):

    qg.run_nSteps(nSteps=nt)
    qg.update_state_variables()

    fig = plt.figure('Perturbation vorticity', figsize=(12, 8)); plt.clf()
    plt.subplot(121); plt.imshow(qg.q1)
    plt.subplot(122); plt.imshow(qg.q2)
    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
