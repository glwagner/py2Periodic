import sys; sys.path.append('../py2Periodic/')
import twoLayerQuasigeostrophic
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt
import twoLayerQGVerificationParams as p

params = { 
    'Lx'         : 1.0e6, 
    'f0'         : 1.0e-4, 
    'beta'       : 1.0e-11,
    'defRadius'  : 1.5e4, 
    'H1'         : 500, 
    'H2'         : 2000, 
    'U1'         : 1.0e0, 
    'U2'         : 1.0e-1, 
    'drag'       : 1.0e-6, 
    'visc'       : 1.0e8, 
    'viscOrder'  : 4.0, 
    'nx'         : 256, 
    'dt'         : 1.0e4, 
    'timeStepper': 'ETDRK4', 
    'nThreads'   : 4, 
}

# Create the two-layer model
qg = twoLayerQuasigeostrophic.model(**p.glennsParams)
qg.describe_model()

# Initial condition: 
# Zeros in q2, random weak-global, strong-zonal perturbations in q1 
#Ro = 1.0e-2
#q1 = Ro*qg.f0*np.random.standard_normal(qg.physVarShape)
#q2 = Ro*qg.f0*np.random.standard_normal(qg.physVarShape)

q1 = np.zeros(qg.physVarShape); q2 = np.zeros(qg.physVarShape)
nnz = [1.0, 2.0, 5.0, 8.0]
mmz = [-3.0, -1.0, 0.0, 1.0, 3.0]
for nn in nnz:
    for mm in mmz:
        q1 += 1.0e-2*np.random.standard_normal()*np.cos( \
            2.0*nn*pi/qg.Lx*qg.XX + 2.0*mm*pi/qg.Ly*qg.YY \
            + 2.0*pi*np.random.standard_normal() \
        ) 

        q2 += 1.0e-2*np.random.standard_normal()*np.cos( \
            2.0*nn*pi/qg.Lx*qg.XX + 2.0*mm*pi/qg.Ly*qg.YY \
            + 2.0*pi*np.random.standard_normal() \
        )

qg.set_q1_q2(q1, q2)

# Run a loop
nt = 1e3
for ii in np.arange(0, 1e3):

    qg.run_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    fig = plt.figure('Perturbation vorticity', figsize=(6, 3)); plt.clf()
    plt.subplot(121); plt.imshow(qg.q1)
    plt.subplot(122); plt.imshow(qg.q2)
    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
