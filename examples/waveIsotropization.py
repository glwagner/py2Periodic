import sys; sys.path.append('../py2Periodic/')
import hydrostaticWaveEqn_xy
import twoDimTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

# Parameters:
# * Physical
f0 = 1e-4
alpha = 1
sigma = f0*np.sqrt(1+alpha)
kappa = 32.0*pi / (Lx*np.sqrt(alpha))
# * Numerical
dt = 0.05 * 2.0*pi/f0
nx = 256
Lx = 1e6
# * Frictional
(turbVisc, turbViscOrder) = (3e8, 4.0)
(waveVisc, waveViscOrder) = (1e24, 8,0)

# Initialize models
turb = twoDimTurbulence.model(nx=nx, Lx=Lx, dt=dt, visc=turbVisc, 
    viscOrder=turbViscOrder, nThreads=2, timeStepper='RK4')

hwe = hydrostaticWaveEqn_xy.model(nx=nx, Lx=Lx, dt=dt, meanVisc=turbVisc, 
    meanViscOrder=turbViscOrder, waveVisc=waveVisc, waveViscOrder=waveViscOrder,
    f0=f0, sigma=sigma, kappa=kappa, nThreads=2, timeStepper='ETDRK4')

# Generate turbulence initial condition, rescaling so that rms(q0)=q0rms
phase = 2.0*pi*np.random.rand(turb.nl, turb.nk)
peakK = 128.0*pi/Lx
q0rms = 0.2*f0

q0h = -np.exp(1j*phase)*(turb.k**2.0+turb.l**2.0)**(3.0/2.0) \
    / (1.0 + np.sqrt(turb.k**2.0+turb.l**2.0)/peakK)**8.0

q0 = turb.ifft2(q0h)
q0 *= q0rms / np.sqrt( (q0**2.0).mean() )

turb.set_q(q0)

# Run turbulence model
stopTime = 400.0*pi/f0
nSteps = int(np.ceil(stopTime / dt))
nPlots = 10
nSubsteps = int(np.ceil(nSteps/nPlots))

print("Iterating {:d} times and making {:d} plots:".format(nSteps, nPlots))
for i in xrange(nPlots):

    turb.step_nSteps(nSteps=nSubsteps, dnLog=nSubsteps)
    turb.update_state_variables()

    # Plot the result
    plt.figure('Flow', figsize=(12, 8)); plt.clf()
    plt.imshow(turb.q)
    plt.pause(0.01)

print("Finished the preliminary turbulence integration! Now let's put a wave field in it...")

# Give final turbulence field to wave model, and run
hwe.set_q(turb.q)

stopTime = 200.0*pi/f0
nSteps = int(np.ceil(stopTime / dt))
nPlots = 100
nSubsteps = int(np.ceil(nSteps/nPlots))

print("Iterating {:d} times and making {:d} plots:".format(nSteps, nPlots))
for i in xrange(nPlots):
    
    hwe.step_nSteps(nSteps=nSubsteps, dnLog=nSubsteps)
    hwe.update_state_variables()
    
    plt.figure('Waves and flow', figsize=(12, 8)); plt.clf()
    plt.subplot(121); plt.imshow(hwe.q)
    plt.subplot(122); plt.imshow(np.sqrt(hwe.u**2.0 + hwe.v**2.0))
    plt.pause(0.01)
