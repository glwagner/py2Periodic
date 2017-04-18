import sys; sys.path.append('../../py2Periodic/')
import hydrostaticWaveEqn_xy
import numpy as np; from numpy import pi
import matplotlib.pyplot as plt

f0 = 1e-4
alpha = 3
sigma = f0*np.sqrt(1+alpha)
Lx = 1e6
k1 = 2.0*pi/Lx
kappa = 16.0*k1 / np.sqrt(alpha)

params = { 
    'nx'            : 128,
    'dt'            : 0.1*2.0*pi/f0,
    'f0'            : f0,
    'sigma'         : sigma,
    'Lx'            : Lx,
    'kappa'         : kappa,
    'meanVisc'      : 1e8, 
    'meanViscOrder' : 4.0, 
    'waveVisc'      : 1e4, 
    'waveViscOrder' : 4.0, 
    'nThreads'      : 2,
}

# Instantiate a model for hydrostatic waves in two-dimensional turbulence.
m = hydrostaticWaveEqn_xy.model(**params)
m.describe_model()

itemsToSave = {
    'q': np.arange(0, 101, 10)*2.0*pi/f0, 
    'A': np.arange(0, 101, 10)*2.0*pi/f0,
}

m.run(nSteps=1000, nLogs=100, nSnaps=100, nPlots=100,
        runName='test', overwrite=True, itemsToSave=itemsToSave)
