import sys; sys.path.append('../py2Periodic/')
import hydrostaticWaveEqn_xy
import twoDimTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

# Parameters
nx = 256
Lx = 1e6
f0 = 1e-4
dt = 0.05 * 2.0*pi/f0
alpha = 1
sigma = f0*np.sqrt(1+alpha)
kappa = 32.0*pi / (Lx*np.sqrt(alpha))

turbVisc = 3e8
turbViscOrder = 4.0
waveVisc = 1e24
waveViscOrder = 8.0

# Initialize models
turb = twoDimTurbulence.model(nx=nx, Lx=Lx, dt=dt, visc=turbVisc, 
    viscOrder=turbViscOrder, nThreads=2, timeStepper='RK4')

hwe = hydrostaticWaveEqn_xy.model(nx=nx, Lx=Lx, dt=dt, meanVisc=turbVisc, 
    meanViscOrder=turbViscOrder, waveVisc=waveVisc, waveViscOrder=waveViscOrder,
    f0=f0, sigma=sigma, kappa=kappa, nThreads=2, timeStepper='ETDRK4')

# Generate turbulence initial condition, rescaling so that rms(q0)=q0rms
phase = np.random.rand(turb.nl, turb.nk)
peakK = 128.0*pi/Lx
q0rms = 0.2*f0

q0h = -np.exp(1j*phase)*(turb.k**2.0+turb.l**2.0)**(3.0/2.0) \
    / (1.0 + np.sqrt(turb.k**2.0+turb.l**2.0)/peakK)**8.0

q0 = turb.ifft2(q0h)
q0 *= q0rms / np.sqrt( (q0**2.0).mean() )

turb.set_q(q0)

# Run turbulence model
stopTime = 400.0*pi/f0
turb.step_until(stopTime=stopTime)
turb.update_state_variables()

# Plot?

# Give final turbulence field to wave model, and run
hwe.set_q(turb.q)
hwe.step_until(stopTime=128*2.0*pi/sigma)
