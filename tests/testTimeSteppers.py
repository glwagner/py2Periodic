import sys; sys.path.append('../py2Periodic/')
import twoDimensionalTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

commonParams = {
    'dt'        : 1.0e-3, 
    'nx'        : 128,
    'Lx'        : 2.0*pi, 
    'visc'      : 1.0e-4, 
    'viscOrder' : 4.0
    'nThreads'  : 4,
}

# Instantiate models with various time-steppers.
m_FE = twoDimensionalTurbulence.model(**commonParams,
    timeStepper = 'forwardEuler')

m_RK4 = twoDimensionalTurbulence.model(**commonParams,
    timeStepper = 'RK4')

m_RKW3 = twoDimensionalTurbulence.model(**commonParams,
    timeStepper = 'RKW3')

m_ETDRK4 = twoDimensionalTurbulence.model(**commonParams,
    timeStepper = 'ETDRK4')

# Define initial condition
q0 = np.random.standard_normal(m_FE.physSolnShape)

# Set the same initial condition for all models
m_FE.set_physical_soln(q0)
m_RK4.set_physical_soln(q0)
m_RKW3.set_physical_soln(q0)
m_ETDRK4.set_physical_soln(q0)

# Run the models
(nSteps, dnLog) = (1e5, 1e3)

print("\nRunning model with forward Euler time-stepping.")
m_FE.run_nSteps(nSteps=nSteps, dnLog=dnLog)

print("\nRunning model with RK4 time-stepping.")
m_RK4.run_nSteps(nSteps=nSteps, dnLog=dnLog)

print("\nRunning model with RKW3 time-stepping.")
m_RKW3.run_nSteps(nSteps=nSteps, dnLog=dnLog)

print("\nRunning model with ETDRK4 time-stepping.")
m_ETDRK4.run_nSteps(nSteps=nSteps, dnLog=dnLog)

# Update variables like vorticity, speed, etc
m_FE.update_state_variables()
m_RK4.update_state_variables()
m_ETDRK4.update_state_variables()
m_RKW3.update_state_variables()

# Plot the result
fig = plt.figure('forward Euler', figsize=(6, 6)); plt.clf()
plt.pcolormesh(m_FE.xx, m_FE.yy, m_FE.q, cmap='YlGnBu_r')
plt.xlabel('$x$', labelpad=5.0); plt.ylabel('$y$', labelpad=12.0)

fig = plt.figure('RK4', figsize=(6, 6)); plt.clf()
plt.pcolormesh(m_RK4.xx, m_RK4.yy, m_RK4.q, cmap='YlGnBu_r')
plt.xlabel('$x$', labelpad=5.0); plt.ylabel('$y$', labelpad=12.0)

fig = plt.figure('ETDRK4', figsize=(6, 6)); plt.clf()
plt.pcolormesh(m_ETDRK4.xx, m_ETDRK4.yy, m_ETDRK4.q, cmap='YlGnBu_r')
plt.xlabel('$x$', labelpad=5.0); plt.ylabel('$y$', labelpad=12.0)

fig = plt.figure('RKW3', figsize=(6, 6)); plt.clf()
plt.pcolormesh(m_RKW3.xx, m_RKW3.yy, m_RKW3.q, cmap='YlGnBu_r')
plt.xlabel('$x$', labelpad=5.0); plt.ylabel('$y$', labelpad=12.0)

print("\nClose the figure to end the program")
plt.show()
