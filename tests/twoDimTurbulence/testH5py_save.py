import sys; sys.path.append('../../py2Periodic/')
import twoDimTurbulence
import numpy as np; from numpy import pi
import time
import h5py

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

# Initialize an h5py dataset
nSnapshots = 100
data = h5py.File('./twoDimTurbSnapshots.hdf5', 'w', libver='latest')

# Create two datasets for time vector and vorticity fields
time = data.create_dataset("time", (nSnapshots,))
vort = data.create_dataset("vorticity", (m.ny, m.nx, nSnapshots)) 

# Store the model input parameters as attributes of the dataset
for key, value in m._input.iteritems(): 
    data.attrs.create(key, value)

# Label the dimensions
vort.dims[0].label = 'x'
vort.dims[1].label = 'y'
vort.dims[2].label = 't'

for i in xrange(nSnapshots):

    m.step_nSteps(nSteps=1e2, dnLog=1e2)
    m.update_state_variables()

    # Save data
    time[i] = m.t
    vort[:, :, i] = m.q

# Close the file
data.close()
