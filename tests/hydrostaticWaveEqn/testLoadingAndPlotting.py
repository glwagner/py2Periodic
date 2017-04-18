import sys; sys.path.append('../../py2Periodic/')
import hydrostaticWaveEqn_xy as hwe
import h5py
import numpy as np; from numpy import pi
import matplotlib.pyplot as plt

fileName = 'testRunningAndSaving'
runName = 'test'

# Must be in the right directory
dataFile = h5py.File("{}.hdf5".format(fileName)) 
runData = dataFile[runName]

params = { param:value for param, value in runData.attrs.iteritems() }

print(params)

# Instantiate a model for hydrostatic waves in two-dimensional turbulence.
m = hwe.model(**params)
m.describe_model()

#itemsToSave = {
#    'q': np.arange(0, 101, 10)*2.0*pi/f0, 
#    'A': np.arange(0, 101, 10)*2.0*pi/f0,
#}
