import sys; sys.path.append('../../py2Periodic/')
import twoLayerTracers
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt
import tracerPlotter

tracerParams = { 
    'dt'          : 1e4,
    'timeStepper' : 'ETDRK4', 
    'hDiff'       : 1e24, 
    'hDiffOrder'  : 8.0,
    'kappa'       : 1e-4,
}

qgTurb = np.load('sampleQGTurb.npz')

params = eval(str(qgTurb['qgParams']))
params.update(tracerParams)

qg = twoLayerTracers.model(**params)

qg.set_q1_and_q2(qgTurb['q1'], qgTurb['q2'])
qg.set_topography(qgTurb['h'])

# Initial condition
x0, r = qg.Lx/2.0, qg.Lx/20.0

c1 = (2.0+qg.delta)*qg.Lx/(np.sqrt(2.0*pi)*r) \
        * np.exp(-(qg.x-x0)**2.0/(2.0*r**2.0))

qg.set_c1(c1)
#qg.set_kappa(1e-6*np.ones(qg.physVarShape))

# Tracer sponge and source parameters: 
#   * Sponge layers have width d, time-scale tau, 
#       and absorb tracer outside xL and xR
#   * Source has width d and injects tracer at xS

d, tau = qg.Lx/40.0, 1e0*qg.dt
xL, xR, xS = 2.0*d, qg.Lx-2.0*d, 4.0*d

sponge = 1.0/tau * (   0.5*( 1.0 - np.tanh((qg.x-xL)/d) ) \
                     + 0.5*( 1.0 + np.tanh((qg.x-xR)/d) ) )

source = np.exp( -(qg.x-xS)**2.0 / (2.0*d**2.0) )

qg.set_tracer_sponges(sponge, sponge)
qg.set_tracer_sources(source, 0.0*qg.c2)
qg.set_c1_and_c2(0.0*qg.c1, 0.0*qg.c2)

# Plot helper
simPlotter = tracerPlotter.plotter(qg, 
    name='Perturbation vorticity and tracers') 

# Run a loop
nt = 1e3
for i in xrange(100):

    qg.step_nSteps_and_average(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    fig = simPlotter.make_plot()
    plt.pause(0.01)

# Change solution to average solution and recompute everything
qg.soln = qg.avgSoln.copy()
qg.update_state_variables()

avgPlotter = tracerPlotter.plotter(qg, name='Averages')
fig = avgPlotter.make_plot()
plt.pause(0.01)

print("Close the plots to end the program")
plt.show()
