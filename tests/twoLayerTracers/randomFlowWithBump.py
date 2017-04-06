import sys; sys.path.append('../../py2Periodic/')
import twoLayerTracers
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

params = { 
    'dt'          : 2e3,
    'nx'          : 128,
    'Lx'          : 1e6, 
    'f0'          : 1e-4,
    'beta'        : 1.5e-11, 
    'defRadius'   : 1.5e4, 
    'H1'          : 500.0, 
    'H2'          : 2000.0, 
    'U1'          : 0.0,
    'U2'          : 0.0,
    'bottomDrag'  : 1e-7,
    'visc'        : 1e9, 
    'viscOrder'   : 4.0, 
    'hDiff'       : 1e6, 
    'hDiffOrder'  : 4.0,
    'nThreads'    : 4,
    'useFilter'   : False,
    'timeStepper' : 'RK4',
}

# Create the two-layer model
qg = twoLayerTracers.model(**params)
qg.describe_model()

# Initial condition: 
Ro = 1.0e-1
f0 = 1.0e-4

q1 = Ro*f0*np.random.standard_normal(qg.physVarShape)
q2 = Ro*f0*np.random.standard_normal(qg.physVarShape)

qg.set_q1_and_q2(q1, q2)

# Gaussian hill topography
(x0, y0, r) = (qg.Lx/2.0, qg.Ly/2.0, qg.Lx/20.0)
h = 0.1*qg.H2*np.exp( -( (qg.x-x0)**2.0 + (qg.y-y0)**2.0 )/(2.0*r**2.0) ) 

qg.set_topography(h)

# Run a loop
nt = 1e3
for ii in np.arange(0, 1e2):

    qg.step_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    fig = plt.figure('Perturbation vorticity and tracers', 
        figsize=(8, 8)); plt.clf()

    plt.subplot(221); plt.imshow(qg.q1)
    plt.subplot(222); plt.imshow(qg.q2)

    plt.subplot(223); plt.imshow(qg.c1)
    plt.subplot(224); plt.imshow(qg.c2)

    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
