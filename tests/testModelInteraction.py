import doublyPeriodic
import twoDimensionalTurbulence
import hydrostaticWavesInXY
import nearInertialWavesInXY
import linearizedBoussinesqInXY
import time
import numpy as np
from numpy import pi, sin, cos, exp, sqrt
import matplotlib.pyplot as plt

nx = 128
dt = 1.0e-3
nt = 1e1
kappa = 1.0
f0 = 1.0e0
nThreads = 8
meanVisc = 1.0e-4
meanViscOrder = 2.0
waveVisc = 1.0e-4
waveViscOrder = 2.0
timeStepper = "forwardEuler"

# Create a bunch of models
turb = twoDimensionalTurbulence.model(
    nx = nx, 
    Lx = 2.0*pi, 
    dt = 1.0e-2,
    nThreads = nThreads, 
    timeStepper = timeStepper,
    visc = meanVisc, 
    viscOrder = meanViscOrder, 
)
turb.describe_model()
turb.run_nSteps(nSteps=8e3, dnLog=2e2)
turb.update_state_variables()

fig = plt.figure('vorticity', figsize=(6, 6))
plt.pcolormesh(turb.xx, turb.yy, turb.q, cmap='YlGnBu_r') 
plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)
plt.pause(0.1)

print("Root-mean-square Rossby number = " + \
        "{:0.3f}".format(sqrt((turb.q**2.0/f0**2.0).mean())))

niw = nearInertialWavesInXY.model(
    nx = nx, 
    Lx = 2.0*pi, 
    dt = dt,
    f0 = f0, 
    kappa = kappa,
    timeStepper = 'RK4', 
    nThreads = nThreads, 
    meanVisc = meanVisc,
    meanViscOrder = meanViscOrder,
    waveVisc = waveVisc,
    waveViscOrder = waveViscOrder,
)
niw.describe_model()
niw.set_q(turb.q)

# Not working right now
bouss = linearizedBoussinesqInXY.model(
    nx = nx, 
    Lx = 2.0*pi, 
    dt = dt,
    f0 = f0, 
    timeStepper = 'RK4', 
    kappa = kappa,
    nThreads = nThreads, 
    meanVisc = meanVisc,
    meanViscOrder = meanViscOrder,
    waveVisc = waveVisc,
    waveViscOrder = waveViscOrder,
    waveDiff = 0.0, 
)
bouss.describe_model()
bouss.set_q(turb.q)

#u = np.ones(bouss.physVarShape)
#v = np.zeros(bouss.physVarShape)
#p = np.zeros(bouss.physVarShape)

# Plane wave with Gaussian envelope
u, v, p = bouss.make_plane_wave(16)
env = exp( \
    - ( (bouss.XX-bouss.Lx/2.0)**2.0 + (bouss.YY-bouss.Ly/2.0)**2.0) \
    / (8.0*pi**2.0/400.0)
)
u = u*env
v = v*env
p = p*env

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

    niw.run_nSteps(nSteps=nt, dnLog=nt)
    niw.update_state_variables()
    niw.speed = np.abs(niw.A)

    bouss.run_nSteps(nSteps=nt, dnLog=nt)
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
