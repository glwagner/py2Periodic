import doublyPeriodic
import twoDimensionalTurbulence
import hydrostaticWavesInXY
import nearInertialWavesInXY
import linearizedBoussinesqInXY
import time
import numpy as np
from numpy import pi, sin, cos, exp, sqrt
import matplotlib.pyplot as plt

# Create a bunch of models
turb = twoDimensionalTurbulence.model(
    nx = 128, 
    Lx = 2.0*pi, 
    dt = 1.0e-1,
    nThreads = 4, 
    timeStepper = "RK4",
)
turb.describe_model()

for ii in np.arange(0, 1e1):

    turb.run_nSteps(nSteps=1e2, dnLog=1e2)
    turb.update_state_variables()

    fig = plt.figure('vorticity', figsize=(6, 6))
    plt.pcolormesh(turb.xx, turb.yy, turb.q, cmap='YlGnBu_r') 
    plt.axis('square')
    plt.xlabel('$x$', labelpad=5.0)
    plt.ylabel('$y$', labelpad=12.0)
    plt.pause(0.1)

dt = 1.0e-2
niw = nearInertialWavesInXY.model(
    nx = 128, 
    Lx = 2.0*pi, 
    dt = dt,
    kappa = 64.0,
    timeStepper = 'RK4', 
    nThreads = 4, 
    meanVisc = 1.0e-4, 
    meanViscOrder = 2.0,
    waveVisc = 1.0e-4, 
    waveViscOrder = 2.0,
)
niw.describe_model()
niw.set_q(turb.q)

# Not working right now
bouss = linearizedBoussinesqInXY.model(
    nx = 128, 
    Lx = 2.0*pi, 
    dt = dt,
    timeStepper = 'RK4', 
    kappa = 64.0,
    nThreads = 4, 
    meanVisc = 1.0e-4, 
    meanViscOrder = 2.0,
    waveVisc = 1.0e-4, 
    waveViscOrder = 2.0,
    waveDiff = 0.0, 
)
bouss.describe_model()
bouss.set_q(turb.q)

u = np.ones(bouss.physVarShape)
v = np.zeros(bouss.physVarShape)
p = np.zeros(bouss.physVarShape)
bouss.set_uvp(u, v, p)

niw.speed = np.abs(niw.A)
bouss.speed = sqrt(bouss.u**2.0 + bouss.v**2.0)

# Plot initial state
(vmin, vmax) = (0.25, 1.25)

fig = plt.figure('bousswaves', figsize=(8, 8))

plt.subplot(221)
plt.pcolormesh(niw.xx, niw.yy, niw.q, cmap='YlGnBu_r')
plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)

plt.subplot(222)
plt.pcolormesh(bouss.xx, bouss.yy, bouss.q, cmap='YlGnBu_r')
plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)

plt.subplot(223)
plt.pcolormesh(niw.xx, niw.yy, niw.speed, cmap='YlGnBu_r') 
    #vmin=vmin, vmax=vmax)
plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)

plt.subplot(224)
plt.pcolormesh(bouss.xx, bouss.yy, bouss.speed, cmap='YlGnBu_r')
    #vmin=vmin, vmax=vmax)
plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)

plt.pause(10)

plt.pause(0.1)

# Run a loop
for ii in np.arange(0, 1e3):

    niw.run_nSteps(nSteps=1e2, dnLog=1e2)
    niw.update_state_variables()
    niw.speed = np.abs(niw.A)

    bouss.run_nSteps(nSteps=1e2, dnLog=1e2)
    bouss.update_state_variables()
    bouss.speed = sqrt(bouss.u**2.0 + bouss.v**2.0)

    plt.subplot(221)
    plt.pcolormesh(niw.xx, niw.yy, niw.q, cmap='YlGnBu_r') 

    plt.subplot(222)
    plt.pcolormesh(bouss.xx, bouss.yy, bouss.q, cmap='YlGnBu_r') 

    plt.subplot(223)
    plt.pcolormesh(niw.xx, niw.yy, niw.speed, cmap='YlGnBu_r')
        #vmin=vmin, vmax=vmax)

    plt.subplot(224)
    plt.pcolormesh(bouss.xx, bouss.yy, bouss.speed, cmap='YlGnBu_r')
        #vmin=vmin, vmax=vmax)

    plt.pause(0.1)

print("Close the plot to end the program")
plt.show()
