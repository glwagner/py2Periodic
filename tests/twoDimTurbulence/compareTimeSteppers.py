import sys; sys.path.append('../../')
from py2Periodic.physics import twoDimTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

answerRefinement = 1e2
nSteps = 1e3

commonParams = {
    'dt'        : 1.0e-1, 
    'nx'        : 128,
    'Lx'        : 2.0*pi, 
    'visc'      : 4.0e-4, 
    'viscOrder' : 4.0,
    'nThreads'  : 4,
}

# Instantiate models with various time-steppers.
answerParams = commonParams.copy(); del answerParams['dt']
m_answer = twoDimTurbulence.model(timeStepper = 'RK4', 
    dt = commonParams['dt']/answerRefinement,
    **answerParams)

m_FE = twoDimTurbulence.model(timeStepper = 'forwardEuler',
    **commonParams)

m_RK4 = twoDimTurbulence.model(timeStepper = 'RK4',
    **commonParams)

m_AB3 = twoDimTurbulence.model(timeStepper = 'AB3',
    **commonParams)

m_RKW3 = twoDimTurbulence.model(timeStepper = 'RKW3',
    **commonParams)

m_ETDRK4 = twoDimTurbulence.model(timeStepper = 'ETDRK4',
    **commonParams)

# Define initial condition
q0 = np.random.standard_normal(m_FE.physSolnShape)

# Set the same initial condition for all models
m_FE.set_physical_soln(q0)
m_RK4.set_physical_soln(q0)
m_AB3.set_physical_soln(q0)
m_RKW3.set_physical_soln(q0)
m_ETDRK4.set_physical_soln(q0)

# Run the models
nSteps = 1e3

print("\nRunning 'answer model' with RK4 time-stepping.")
timer = time.time()
m_answer.step_nSteps(nSteps=nSteps*answerRefinement)
print("Total RK4 'answer model' time = {:.3f} s".format(time.time()-timer))

print("\nRunning model with forward Euler time-stepping.")
timer = time.time()
m_FE.step_nSteps(nSteps=nSteps)
print("Total forwardEuler time = {:.3f} s".format(time.time()-timer))

print("\nRunning model with RK4 time-stepping.")
timer = time.time()
m_RK4.step_nSteps(nSteps=nSteps)
print("Total RK4 time = {:.3f} s".format(time.time()-timer))

print("\nRunning model with AB3 time-stepping.")
timer = time.time()
m_AB3.step_nSteps(nSteps=nSteps)
print("Total AB3 time = {:.3f} s".format(time.time()-timer))

print("\nRunning model with RKW3 time-stepping.")
timer = time.time()
m_RKW3.step_nSteps(nSteps=nSteps)
print("Total RK4 time = {:.3f} s".format(time.time()-timer))

print("\nRunning model with ETDRK4 time-stepping.")
timer = time.time()
m_ETDRK4.step_nSteps(nSteps=nSteps)
print("Total ETDRK4 time = {:.3f} s".format(time.time()-timer))

# Update variables like vorticity, speed, etc
m_FE.update_state_variables()
m_RK4.update_state_variables()
m_AB3.update_state_variables()
m_ETDRK4.update_state_variables()
m_RKW3.update_state_variables()

# Plot the result
fig = plt.figure('Time-stepper comparison', figsize=(12, 12)); plt.clf()

plt.subplot(231)
plt.title('forward Euler')
plt.pcolormesh(m_FE.x, m_FE.y, np.abs(m_FE.q-m_answer.q), cmap='YlGnBu_r')

plt.subplot(232)
plt.title('RK4')
plt.pcolormesh(m_RK4.x, m_RK4.y, np.abs(m_RK4.q-m_answer.q), cmap='YlGnBu_r')

plt.subplot(233)
plt.title('AB3')
plt.pcolormesh(m_AB3.x, m_AB3.y, np.abs(m_AB3.q-m_answer.q), cmap='YlGnBu_r')

plt.subplot(234)
plt.title('ETDRK4')
plt.pcolormesh(m_ETDRK4.x, m_ETDRK4.y, np.abs(m_ETDRK4.q-m_answer.q), cmap='YlGnBu_r')

plt.subplot(235)
plt.title('RKW3')
plt.pcolormesh(m_RKW3.x, m_RKW3.y, np.abs(m_RKW3.q-m_answer.q), cmap='YlGnBu_r')

print("\nClose the figure to end the program")
plt.show()
