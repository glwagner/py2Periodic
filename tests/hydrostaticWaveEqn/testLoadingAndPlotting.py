import sys; sys.path.append('../../py2Periodic/')
import hydrostaticWaveEqn_xy as hwe
import h5py
import numpy as np; from numpy import pi
import matplotlib.pyplot as plt
import re


def get_run_params(dataFile, runName=None):
    """ Probe a group with 'runName' inside an HDF5 file to 
        extract a dictionary of model parameters used during
        that particular run """

    # If no runName is specified, choose the last run in the file
    if runName is None:
        names = [ group for group in dataFile.keys() ]
        runName = names[-1]

    # Extract parameters from run attributes 
    runGroup = dataFile[runName]
    params = { param:value for param, value in runGroup.attrs.iteritems() }

    return params


def get_run_snaps_and_items(dataFile, runName=None, 
    runSnapshotName='run_snapshots', snapPrefix='run_', 
    itemSuffix='_snapshots'):

    """ Probe HDF5 file to find groups with soln 
        snapshot data and saved item data. The keyword 
        parameters must correspond to the default save  
        parameters in doublyPeriodic.py. Remember there are two
        kinds of data saved by doublyPeriodic --- "run snapshots", 
        which are saved episodically, and "saved items", which are 
        saved at times specified in an input dictionary. """

    # If no runName is specified, choose the last run in the file
    if runName is None:
        names = [ group for group in dataFile.keys() ]
        runName = names[-1]

    runGroup = dataFile[runName]
    runSnapshots = runGroup[runSnapshotsName]

    # Extract a dictionary of savedItem groups
    savedItemList = [ re.sub(itemSuffix, '', item) for item in runGroup.keys() 
        if snapPrefix not in item ]

    savedItems = { var:runGroup[var+itemSuffix] for var in savedItemList }

    return runSnapshots, savedItems


def get_run_snaps(dataFile, runName=None,
    runSnapshotName='run_snapshots', snapPrefix='run_', 
    itemSuffix='_snapshots'):

    """ Probe HDF5 file to find a group with periodic snapshots 
        from the run. See the docstring for "get_snaps_and_items" 
        for information about the keyword paramters. """

    # If no runName is specified, choose the last run in the file
    if runName is None:
        names = [ group for group in dataFile.keys() ]
        runName = names[-1]

    runGroup = dataFile[runName]
    runSnapshots = runGroup[runSnapshotsName]

    return runSnapshots


def get_run_items(dataFile, runName=None,
    runSnapshotName='run_snapshots', snapPrefix='run_', 
    itemSuffix='_snapshots'):

    """ Probe HDF5 file to find groups corresponding to data from
        items saved at specified times during the run.
        See the docstring for "get_snaps_and_items" 
        for information about the keyword paramters. """

    # If no runName is specified, choose the last run in the file
    if runName is None: runName = dataFile.keys()[-1]
    runGroup = dataFile[runName]

    # Extract a dictionary of savedItem groups
    savedItemList = [ re.sub(itemSuffix, '', item) for item in runGroup.keys() 
        if snapPrefix not in item ]

    savedItems = { var:runGroup[var+itemSuffix] for var in savedItemList }

    return savedItems


# Open read-only hdf5 file
dataFile = h5py.File('testRunningAndSaving.hdf5', 'r')
runName = 'test'

params = get_run_params(dataFile, runName=runName)
savedItems = get_run_items(dataFile, runName=runName)

print(savedItems)
print(params)

# Instantiate a model for hydrostatic waves in two-dimensional turbulence.
m = hwe.model(**params)
m.describe_model()

#itemsToSave = {
#    'q': np.arange(0, 101, 10)*2.0*pi/f0, 
#    'A': np.arange(0, 101, 10)*2.0*pi/f0,
#}
