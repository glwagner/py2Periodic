import spectralModels, twoDimTurbulence  
import time
import numpy as np
from numpy import pi, sin, cos, exp, sqrt
import matplotlib.pyplot as plt

# Initialize colorbar dictionary
colorbarProperties = {
    'orientation' : 'horizontal',
    'shrink'      : 1.0,
    'extend'      : 'both',
}

# Create the model
m = twoDimTurbulence.model(
    nx = 256, 
    Lx = 2.0*pi,
    dt = 1.0e-1, 
    nu = 1.0e-2,
    nThreads = 4, 
)

m.describe_model()

kz = 2*pi/m.Lx*np.arange(2, 5)
px = 2*pi*np.array((0.0, 1/12, 5/6))
py = 2*pi*np.array((1/3, 3/7, 6/7))
aa = np.array((0.7, 0.2, 0.1))

q0x, q0y = np.zeros_like(m.XX), np.zeros_like(m.XX)
for ii in np.arange(0, 3):
    q0x += aa[ii]*cos(kz[ii]*m.XX+px[ii])
    q0y += aa[ii]*cos(kz[ii]*m.YY+py[ii])

q0 = q0x*q0y
q0 = q0 / q0.max()
        
soln = np.zeros(m.physicalShape)
soln[:, :, 0] = q0
m.set_physical_sol(soln)

fig = plt.figure('vorticity', figsize=(6, 6))
plt.pcolormesh(m.xx, m.yy, m.q, cmap='RdBu_r')

plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)

plt.pause(1)

# Run a loop
for ii in np.arange(0, 20):

    m.run_nSteps(nSteps=1e2, dnLog=1e2)

    plt.pcolormesh(m.xx, m.yy, m.q, cmap='RdBu_r')
    plt.pause(0.1)

plt.show()
