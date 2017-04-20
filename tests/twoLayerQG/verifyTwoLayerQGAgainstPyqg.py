import time, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from py2Periodic.physics import twoLayerQG
from numpy import pi

pyqgParams = { 
    'dt'        : 8000,
    'nx'        : 128,
    'L'         : 1.0e6,
    'beta'      : 1.5e-11,
    'rd'        : 1.5e4,
    'rek'       : 1.0e-7,
    'delta'     : 0.25, 
    'H1'        : 500, 
    'U1'        : 0.025, 
    'U2'        : 0.0, 
    'tavestart' : 0.0, 
    'taveint'   : 86400.0, 
    'twrite'    : 1000.0, 
    'useAB2'    : False,
    'ntd'       : 2,
}

myParams = { 
    'Lx'         : pyqgParams['L'],
    'beta'       : pyqgParams['beta'], 
    'defRadius'  : pyqgParams['rd'], 
    'H1'         : pyqgParams['H1'],
    'H2'         : pyqgParams['H1']/pyqgParams['delta'],
    'U1'         : pyqgParams['U1'],
    'U2'         : pyqgParams['U2'],
    'bottomDrag' : pyqgParams['rek'],
    'nx'         : pyqgParams['nx'],
    'dt'         : pyqgParams['dt'],
    'visc'       : 0e6, 
    'viscOrder'  : 4.0, 
    'timeStepper': 'AB3', 
    'nThreads'   : pyqgParams['ntd'],
    'useFilter'  : True,
}

m = pyqg.QGModel(**pyqgParams)
qg = twoLayerQG.model(**myParams)
qg.describe_model()

# Initial condition
(f0, Ro) = (1.0e-4, 1.0e-3)
q1 = Ro*f0*np.random.standard_normal((m.ny, m.nx))
q2 = Ro*f0*np.random.standard_normal((m.ny, m.nx))

m.set_q1q2(q1, q2) 
qg.set_q1_and_q2(q1, q2)

# Run models, interleaved within one another.
nt = 1e3

for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=nt*m.dt):

    # Run qg
    qg.step_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    fig = plt.figure('Physical', figsize=(12, 6)); plt.clf()

    # Plot qg
    plt.subplot(231); plt.imshow(qg.q1) # + qg.Q1y*qg.YY)
    plt.subplot(234); plt.imshow(qg.q2) # + qg.Q2y*qg.YY)

    # Plot pyqg
    plt.subplot(232); plt.imshow(m.q[0]) # + m.Qy1*m.y)
    plt.subplot(235); plt.imshow(m.q[1]) # + m.Qy2*m.y)

    # Plot difference
    norm = np.sqrt( (m.q[0]**2.0).mean() )
    plt.subplot(233); plt.imshow(np.abs(m.q[0]-qg.q1)/norm, vmin=0, vmax=1) # + m.Qy1*m.y)
    plt.subplot(236); plt.imshow(np.abs(m.q[1]-qg.q2)/norm, vmin=0, vmax=1) # + m.Qy2*m.y)

    plt.pause(0.01), plt.draw()

    fig = plt.figure('Spectral', figsize=(12, 6)); plt.clf()
    q1h = qg.soln[:, :, 0]
    q2h = qg.soln[:, :, 1]

    # Plot qg
    plt.subplot(231); plt.imshow(np.abs(q1h[0:qg.ny//2+1, :]))
    plt.subplot(234); plt.imshow(np.abs(q2h[0:qg.ny//2+1, :]))

    # Plot pyqg
    plt.subplot(232); plt.imshow(np.abs(m.qh[0][0:m.ny//2+1, :])) # + m.Qy1*m.y)
    plt.subplot(235); plt.imshow(np.abs(m.qh[1][0:m.ny//2+1, :])) # + m.Qy2*m.y)

    # Plot difference
    norm = np.sqrt( (np.abs(m.qh[0])**2.0).mean() )
    del1 = m.qh[0] - q1h
    del2 = m.qh[1] - q2h
    plt.subplot(233); plt.imshow(np.abs(del1[0:m.ny//2+1, :])/norm, vmin=0, vmax=1) # + m.Qy1*m.y)
    plt.subplot(236); plt.imshow(np.abs(del2[0:m.ny//2+1, :])/norm, vmin=0, vmax=1) # + m.Qy2*m.y)

    plt.pause(0.01), plt.draw()

    
print("Close the plot to end the program")
plt.show()
