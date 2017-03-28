import sys; sys.path.append('../py2Periodic/')
import twoDimensionalTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

def many2DTurbulenceSimulations(params, dt, nSteps, timeSteppers, q0):

    solutions = {}
    # Instantiate and run a bunch of two-dimensional turbulence models.
    for timeStepper in timeSteppers:
        m = twoDimensionalTurbulence.model(timeStepper = timeStepper,
                dt = dt,
                **params)

        m.set_physical_soln(q0)
        m.run_nSteps(nSteps=nSteps)
        m.update_state_variables()

        solutions[timeStepper] = m.q

    return solutions

nSteps0 = 1e2
dtz = np.array([1.0e-3, 2.0e-3, 4.0e-3, 1.0e-2, 2.0e-2, 4.0e-2, 1.0e-1])
timeSteppers = ['forwardEuler', 'RK4', 'RKW3', 'ETDRK4']

params = {
    'nx'        : 128,
    'Lx'        : 2.0*pi, 
    'visc'      : 4.0e-4, 
    'viscOrder' : 4.0,
    'nThreads'  : 4,
}

# Define initial condition
q0 = np.random.standard_normal((params['nx'], params['nx'], 1))

# Run many simulations
solutions = {}
for dt in dtz:
    solutions[dt] = many2DTurbulenceSimulations( \
        params, dt, int(nSteps0*dtz[-1]/dt), timeSteppers, q0)

# Analyze results
bulkError = {}
answer = solutions[ dtz[-1] ][ 'RK4' ]

for timeStepper in timeSteppers:
    bulkError[timeStepper] = np.zeros_like(dtz)
    for ii, dt in enumerate(dtz):
        errorDensity = np.abs(answer-solutions[dt][timeStepper])**2.0
        bulkError[timeStepper][ii] = errorDensity.sum()
        
# Plot the results
fig = plt.figure(('Bulk error'), figsize=(3, 3)); plt.clf()

plt.plot(dtz, bulkError[timeSteppers[0]], 'ko--')
plt.plot(dtz, bulkError[timeSteppers[1]], 'b^-')
plt.plot(dtz, bulkError[timeSteppers[2]], 'rs--')
plt.plot(dtz, bulkError[timeSteppers[3]], 'gp-')

print("\nClose the figure to end the program")
plt.show()
