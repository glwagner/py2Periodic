import sys; sys.path.append('../py2Periodic/')
import twoLayerQuasigeostrophic
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt
import pyqg

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
    'ntd'       : 4,
}

myParams = { 
    'Lx'         : pyqgParams['L'],
    'beta'       : pyqgParams['beta'], 
    'defRadius'  : pyqgParams['rd'], 
    'H1'         : pyqgParams['H1'],
    'H2'         : pyqgParams['H1']/pyqgParams['delta'],
    'U1'         : pyqgParams['U1'],
    'U2'         : pyqgParams['U2'],
    'drag'       : pyqgParams['rek'],
    'nx'         : pyqgParams['nx'],
    'dt'         : pyqgParams['dt'],
    'visc'       : 1.0e6, 
    'viscOrder'  : 4.0, 
    'timeStepper': 'AB3', 
    'nThreads'   : pyqgParams['ntd'],
    'useFilter'  : True,
}

m = pyqg.QGModel(**pyqgParams)
qg = twoLayerQuasigeostrophic.model(**myParams)
qg.describe_model()

# Initial condition
(f0, Ro) = (1.0e-4, 1.0e-2)
q1 = Ro*f0*np.random.standard_normal((m.ny, m.nx))
q2 = Ro*f0*np.random.standard_normal((m.ny, m.nx))

m.set_q1q2(q1, q2) 
qg.set_q1_and_q2(q1, q2)

# Run models, interleaved within one another.
nt = 1e3

for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=nt*m.dt):

    # Run qg
    qg.run_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    # Plot qg
    fig = plt.figure('Perturbation vorticity', figsize=(8, 4)); plt.clf()
    plt.subplot(121); plt.imshow(qg.q1) # + qg.Q1y*qg.YY)
    plt.subplot(122); plt.imshow(qg.q2) # + qg.Q2y*qg.YY)
    plt.pause(0.01), plt.draw()

    # Plot pyqg
    plt.figure('pyqg', figsize=(8, 4)); plt.clf()
    plt.subplot(121); plt.imshow(m.q[0]) # + m.Qy1*m.y)
    plt.subplot(122); plt.imshow(m.q[1]) # + m.Qy2*m.y)
    plt.pause(0.01); plt.draw()
    
print("Close the plot to end the program")
plt.show()
