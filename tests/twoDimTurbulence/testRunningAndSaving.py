import time, sys; sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt

from py2Periodic.physics import twoDimTurbulence
from numpy import pi

m = twoDimTurbulence.model(
    nx = 128, 
    Lx = 2.0*pi, 
    dt = 1.0e-1,
    nThreads = 1, 
    timeStepper = 'AB3',
    visc = 1.0e-7, 
    viscOrder = 4.0, 
)

m.set_q(np.random.standard_normal((m.ny, m.nx)))

itemsToSave = {
    'q': np.arange(0, 100, 10), 
    'u': np.arange(0, 100, 10),
    'v': np.arange(0, 100, 10),
}

m.run(nSteps=1000, nLogs=10, nChecks=10, nPlots=10,
        runName='test', overwriteToSave=True, itemsToSave=itemsToSave, 
        saveEndpoint=True)
m.update_state_variables()

m.visualize_model_state(show=True)
