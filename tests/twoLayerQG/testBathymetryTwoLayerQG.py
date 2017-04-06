import sys; sys.path.append('../py2Periodic/')
import twoLayerQG
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

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
    'ny'         : 128,
    'dt'         : 1.0e3, 
    'visc'       : 0.0e8, 
    'viscOrder'  : 4.0, 
    'timeStepper': 'RK4', 
    'nThreads'   : 4,
    'useFilter'  : True,
    'flatBottom' : True, 
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

# Bathymetry
R = qg.Lx/20
(x0, y0) = (qg.Lx/2.0, qg.Ly/2.0)
h = 0.1*qg.H2*np.exp( (-(qg.x-x0)**2.0 - (qg.y-y0)**2.0)/(2.0*R**2.0) )
qg.set_bathymetry(h)

# Run a loop
nt = 1e3
for ii in np.arange(0, 1e3):

    qg.step_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    fig = plt.figure('Perturbation vorticity', figsize=(8, 8)); plt.clf()

    plt.subplot(221); plt.imshow(qg.q1)
    plt.subplot(222); plt.imshow(qg.q2)

    plt.subplot(223); plt.imshow(np.abs(qg.soln[0:qg.ny//2-1, :, 0]))
    plt.subplot(224); plt.imshow(np.abs(qg.soln[0:qg.ny//2-1, :, 1]))

    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
