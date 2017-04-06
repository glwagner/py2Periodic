import sys; sys.path.append('../../py2Periodic/')
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
}

# Create the two-layer model
qg = twoLayerQG.model(**qgParams)
qg.describe_model()

# Initial condition to seed baroclinic instability
Ro = 1.0e-3
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
for ii in np.arange(0, 1e1):

    qg.step_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    fig = plt.figure('Perturbation vorticity', figsize=(8, 6)); plt.clf()

    plt.subplot(121); plt.imshow(qg.q1)
    plt.subplot(122); plt.imshow(qg.q2)
    plt.pause(0.01), plt.draw()

# Save the result in an npz archive
np.savez('sampleQGTurb.npz', q1=qg.q1, q2=qg.q2, qgParams=qgParams,
    h=h)
