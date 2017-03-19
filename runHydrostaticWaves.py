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

# Create the model
m = modewiseHydrostaticWaves.hydrostaticWaveModel(
    nThreads = 4, 
    makingPlots = True,
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
