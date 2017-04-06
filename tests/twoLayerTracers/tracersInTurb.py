import sys; sys.path.append('../../py2Periodic/')
import twoLayerTracers
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt

tracerParams = { 
    'dt'          : 1e4,
    'timeStepper' : 'ETDRK4', 
    'hDiff'       : 1e12, 
    'hDiffOrder'  : 8.0,
}

qgTurb = np.load('sampleQGTurb.npz')

params = eval(str(qgTurb['qgParams']))
params.update(tracerParams)

qg = twoLayerTracers.model(**params)

qg.set_q1_and_q2(qgTurb['q1'], qgTurb['q2'])
qg.set_topography(qgTurb['h'])

# Initial condition
(x0, r) = (qg.Lx/2.0, qg.Lx/20.0)
qg.set_c1( np.exp(-(qg.x-x0)**2.0 / (2.0*r**2.0)) )
qg.set_kappa(0.0*qg.c1)

# Tracer sponge and source parameters: 
#   * Sponge layers have width d, time-scale tau, 
#       and absorb tracer outside xL and xR
#   * Source has width d and injects tracer at xS

(d, tau) = (qg.Lx/40.0, 1e2*qg.dt)
(xL, xR, xS) = (2.0*d, qg.Lx-2.0*d, 4.0*d)

sponge = 1.0/tau * (   0.5*( 1.0 - np.tanh((qg.x-xL)/d) ) \
                     + 0.5*( 1.0 + np.tanh((qg.x-xR)/d) ) )

source = np.exp( -(qg.x-xS)**2.0 / (2.0*d**2.0) )

#qg.set_tracer_sponges(sponge, sponge)
#qg.set_tracer_sources(source, 0.0*qg.c2)
#qg.Tracers.set_c1_and_c2(0.0*qg.c1, 0.0*qg.c2)

# Run a loop
nt = 1e1
for ii in np.arange(0, 1e2):

    qg.step_nSteps(nSteps=nt, dnLog=nt)
    qg.update_state_variables()

    fig = plt.figure('Perturbation vorticity and tracers', 
        figsize=(8, 8)); plt.clf()

    plt.subplot(221); plt.imshow(qg.q1)
    plt.subplot(222); plt.imshow(qg.q2)

    plt.subplot(223); plt.imshow(qg.c1)
    plt.subplot(224); plt.imshow(qg.c2)

    plt.pause(0.01), plt.draw()

print("Close the plot to end the program")
plt.show()
