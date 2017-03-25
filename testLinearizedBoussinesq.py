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
dt = 1.0e-1
nt = 1e2
kappa = 64.0
f0 = 2.0*pi
nThreads = 8
meanVisc = 1.0e-4
meanViscOrder = 2.0
waveVisc = 1.0e-8
waveViscOrder = 4.0
timeStepper = "ETDRK4"

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
#turb.run_nSteps(nSteps=8e3, dnLog=2e2)
#turb.update_state_variables()

#fig = plt.figure('vorticity', figsize=(6, 6))
#plt.pcolormesh(turb.xx, turb.yy, turb.q, cmap='YlGnBu_r') 
#plt.axis('square')
#plt.xlabel('$x$', labelpad=5.0)
#plt.ylabel('$y$', labelpad=12.0)
#plt.pause(0.1)

print("Root-mean-square Rossby number = " + \
        "{:0.3f}".format(sqrt((turb.q**2.0/f0**2.0).mean())))

bouss = linearizedBoussinesqInXY.model(
    nx = nx, 
    Lx = 2.0*pi, 
    dt = dt,
    f0 = f0, 
    timeStepper = timeStepper,
    kappa = kappa,
    nThreads = nThreads, 
    meanVisc = meanVisc,
    meanViscOrder = meanViscOrder,
    waveVisc = waveVisc,
    waveViscOrder = waveViscOrder,
    waveDiff = 0.0, 
)
bouss.describe_model()
#bouss.set_q(turb.q)
#bouss.set_q(np.zeros(bouss.physVarShape))

q0 = 0.01 * exp( \
    - ( (bouss.XX-2.0*bouss.Lx/4.0)**2.0 + (bouss.YY-bouss.Ly/2.0)**2.0) \
    / (8.0*pi**2.0/200.0) \
)
bouss.set_q(q0)

u = np.ones(bouss.physVarShape)
v = np.zeros(bouss.physVarShape)
p = np.zeros(bouss.physVarShape)

# Plane wave with Gaussian envelope
#u, v, p = bouss.make_plane_wave(32)
#(x0, y0) = (bouss.Lx/4.0, bouss.Ly/2.0)
#env = exp( \
#    - ( (bouss.XX-x0)**2.0 + (bouss.YY-y0)**2.0) \
#    / (8.0*pi**2.0/200.0)
#)
#u = u*env
#v = v*env
#p = p*env

bouss.set_uvp(u, v, p)

# Plot initial state
(vmin, vmax) = (0.25, 1.25)

fig = plt.figure('bousswaves', figsize=(16, 8))

plt.subplot(121)
plt.pcolormesh(bouss.xx, bouss.yy, bouss.q, cmap='YlGnBu_r')
plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)

plt.subplot(122)
plt.pcolormesh(bouss.xx, bouss.yy, bouss.u, cmap='YlGnBu_r')
plt.axis('square')
plt.xlabel('$x$', labelpad=5.0)
plt.ylabel('$y$', labelpad=12.0)

plt.pause(0.1)

# Run a loop
for ii in np.arange(0, 1e3):

    bouss.run_nSteps(nSteps=nt, dnLog=nt)
    bouss.update_state_variables()

    plt.subplot(121)
    plt.pcolormesh(bouss.xx, bouss.yy, bouss.q, cmap='YlGnBu_r') 

    plt.subplot(122)
    plt.pcolormesh(bouss.xx, bouss.yy, bouss.u, cmap='YlGnBu_r')

    plt.pause(0.1)

print("Close the plot to end the program")
plt.show()
