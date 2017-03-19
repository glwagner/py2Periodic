import modewiseHydrostaticWaves
import time
import numpy as np
from numpy import pi, sin, cos, exp, sqrt
import matplotlib.pyplot as plt

# Initialize colorbar dictionary
colorbarProperties = {
    'orientation' : 'vertical',
    'shrink'      : 0.8,
    'extend'      : 'neither',
}

f0 = 1.0
sigma = 2.0
alpha = sigma**2.0/f0**2.0 - 1

k0 = 16
kappa = k0 / sqrt(alpha)

# Create the model
m = modewiseHydrostaticWaves.hydrostaticWaveModel(
    nThreads = 4, 
    makingPlots = True,
    f0 = f0, 
    sigma = sigma, 
    kappa = kappa,
)

m.describe_model()
m.update_state_variables()
m.plot_current_state()

plt.pause(1)

# Run a loop
for ii in np.arange(0, 1e3):

    m.run_nSteps(nSteps=1e1)
    m._print_status()
    m.update_state_variables()
    m.plot_current_state()

    plt.pause(0.1)
