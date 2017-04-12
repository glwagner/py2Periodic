import sys; sys.path.append('../../py2Periodic/')
import twoDimTurbulence
import numpy as np; from numpy import pi
import time
import matplotlib.pyplot as plt
import h5py

m = twoDimTurbulence.model(
    nx = 128, 
    Lx = 2.0*pi, 
    dt = 1.0e-1,
    nThreads = 1, 
    timeStepper = 'AB3',
    visc = 1.0e-4, 
    viscOrder = 4.0, 
)

m.set_q(np.random.standard_normal((m.ny, m.nx))

# Initialize an h5py dataset
nSnapshots = 100
file = h5py.File('twoDimTurbSnapshots.hdf5', 'w', libver='latest')

time = file.create_dataset("time", (nSnapshots, ), dtype='f')
vorticity = file.create_dataset("vorticity", 
    (m.nx, m.nx, nSnapshots), dtype='c128')

vorticity.dims[0].label = 'x'
vorticity.dims[1].label = 'y'
time.dims[0].label = 't'

for i in xrange(nSnapshots):

    m.step_nSteps(nSteps=1e2, dnLog=1e2)
    m.update_state_variables()

    # Save data
    time[i] = m.t
    vorticity[:, :, i] = m.q

# Close the file
file.close()
