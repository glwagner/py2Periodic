import twoLayerQG
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
m = twoLayerQG.twoLayerQGModel(
    nThreads = 8, 
    makingPlots = True,
)

m.describe_model()
m.update_state_variables()
m.plot_current_state()

plt.pause(1)

# Run a loop
for ii in np.arange(0, 1e2):

    m.run_nSteps(nSteps=1e2, dnLog=1e2)
    m.update_state_variables()
    m.plot_current_state()

    plt.pause(0.1)
