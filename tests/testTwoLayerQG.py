import sys; sys.path.append('../py2Periodic/')
import twoLayerQuasigeostrophic
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt
import twoLayerQGInputParams as params

# Create the two-layer model
qg = twoLayerQuasigeostrophic.model(**params.simpleParams)
qg.describe_model()

# Initial condition: 
# Zeros in q2, random weak-global, strong-zonal perturbations in q1 
q1i = 2.0e-7*np.random.rand(qg.ny, qg.nx) \
    + 2.0e-6*( np.ones((qg.ny, 1))*np.random.rand(1, qg.nx) )
q2i = np.zeros(qg.physVarShape)

qg.set_q1(q1i)
qg.set_q2(q2i)

fig = plt.figure('vorticity', figsize=(3, 3)); plt.clf() 
plt.imshow(qg.q1 + qg.Qy1*qg.YY)
plt.clim([0, qg.Qy1*qg.Ly])
plt.pause(0.01), plt.draw()

# Run a loop
nt = 1e2
for ii in np.arange(0, 1e3):

    qg.run_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    plt.clf()
    plt.imshow(qg.q1 + qg.Qy1*qg.YY)
    plt.clim([0, qg.Qy1*qg.Ly])
    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
