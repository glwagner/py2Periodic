import sys; sys.path.append('../py2Periodic/')
import twoLayerTracers
import twoLayerQG
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

qgParams = { 
    'f0'         : 1e-4,
    'Lx'         : 1e6, 
    'beta'       : 1.5e-11, 
    'defRadius'  : 1.5e4, 
    'H1'         : 500.0, 
    'H2'         : 2000.0, 
    'U1'         : 2.5e-2, 
    'U2'         : 0.0,
    'bottomDrag' : 1e-7,
    'nx'         : 128,
    'dt'         : 1e4, 
    'visc'       : 1e9, 
    'viscOrder'  : 4.0, 
    'timeStepper': 'AB3', 
    'nThreads'   : 4,
    'useFilter'  : False,
    'topography' : True
}

tracerParams = {
    'dt'          : 2e3,
    'timeStepper' : 'RK4',
    'hDiff'       : 1e6, 
    'hDiffOrder'  : 4.0,
    'forcing'    : True
    'sponge'     : True
}

allParams = qgParams.copy()
allParams.update(tracerParams)

# Create the two-layer model
qg = twoLayerQG.model(**qgParams)
qgTracers = twoLayerTracers.model(**allParams)
qg.describe_model()

# Initial condition: 
Ro = 1.0e-3
f0 = 1.0e-4

q1 = Ro*f0*np.random.standard_normal(qg.physVarShape)
q2 = Ro*f0*np.random.standard_normal(qg.physVarShape)

qg.set_q1_and_q2(q1, q2)

# Gaussian hill topography
(x0, y0) = (qg.Lx/2.0, qg.Ly/2.0)
rTop = qg.Lx/20.0
h = 0.1*qg.H2*np.exp( -( (qg.x-x0)**2.0 + (qg.y-y0)**2.0 )/(2.0*rTop**2.0) ) 
qg.set_topography(h)
qgTracers.set_topography(h)

# Run a loop
nt = 1e3
for ii in np.arange(0, 1e1):
    qg.step_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()
    fig = plt.figure('Perturbation vorticity', figsize=(8, 8)); plt.clf()
    plt.subplot(121); plt.imshow(qg.q1)
    plt.subplot(122); plt.imshow(qg.q2)
    plt.pause(0.01), plt.draw()

qgTracers.set_q1_and_q2(qg.q1, qg.q2)

# Run a loop
nt = 1e3
for ii in np.arange(0, 1e2):

    qgTracers.step_nSteps(nSteps=nt, dnLog=nt)
    qgTracers.update_state_variables()

    fig = plt.figure('Perturbation vorticity and tracers', 
        figsize=(8, 8)); plt.clf()

    plt.subplot(221); plt.imshow(qgTracers.q1)
    plt.subplot(222); plt.imshow(qgTracers.q2)

    plt.subplot(223); plt.imshow(qgTracers.c1)
    plt.subplot(224); plt.imshow(qgTracers.c2)

    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
