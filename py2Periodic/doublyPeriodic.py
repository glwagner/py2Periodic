from __future__ import division

import os, sys, time as timeTools
import numpy as np
import numexpr as ne
import mkl, pyfftw, h5py

from py2Periodic import timeStepping
from numpy import pi

try:
    import mklfft
    usingMKLFFT = True
except:
    usingMKLFFT = False
    

class doublyPeriodicModel(object):
    def __init__(self, name = None, physics = None, nVars = 1, realVars = False,
            # Grid resolution and extent
            nx = 64, ny = None, Lx = 2.0*pi, Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0e-2,                   # Fixed numerical time-step.
            step = 0,                      # Initial or current step
            timeStepper = "forwardEuler",  # Time-stepping method
            nThreads = 1,                  # Number of threads for FFTW
            useFilter = False,             # Use exp filter rather than dealias
        ):

        # Default grid is square when user specifes only nx
        if Ly is None: Ly = Lx
        if ny is None: ny = nx

        # Default 'name' is the name of the script that runs the model
        if name is None: 
            scriptName = os.path.basename(sys.argv[0])
            self.name = scriptName[:-3] # Remove .py from the end.
        else:
            self.name = name

        self.physics = physics
        self.nVars = nVars
        self.realVars = realVars

        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly

        self.t = t
        self.dt = dt
        self.step = step

        self.timeStepper = timeStepper
        self.useFilter = useFilter

        if nThreads is 'maximum' or nThreads > mkl.get_max_threads(): 
            self.nThreads = mkl.get_max_threads()
        else:
            self.nThreads = nThreads

        self._input = { key:value for key, value in self.__dict__.items()
            if type(value) in (str, float, int, bool) and
            key not in ('realVars', 'nVars', 'physics')
        }

        np.use_fastnumpy = True
        mkl.set_num_threads(self.nThreads)
        ne.set_num_threads(self.nThreads)

        # Initialization routines
        self._init_numerical_parameters()   # Defined in doublyPeriodic class
        self._init_problem_parameters()     # Defined in Physical Problem subclass
        self._init_linear_coeff()           # Defined in Physical Problem subclass
        self._init_fft()                    # Defined in doublyPeriodic class

        # Initialize the time-stepper
        self._timeStepper = getattr(timeStepping.methods, self.timeStepper)(self)
        self._step_forward = self._timeStepper.step_forward


    # Initialization routines - - - - - - - - - - - - - - - - - - - - - - - - -   
    def _init_numerical_parameters(self):
        """ Define the grid, initialize core variables, and initialize 
            miscallenous model parameters. """

        # Physical grid
        self.dx = self.Lx/self.nx
        self.dy = self.Ly/self.ny

        xx = np.arange(0.0, self.Lx, self.dx)
        yy = np.arange(0.0, self.Ly, self.dy)

        self.x, self.y = np.meshgrid(xx, yy)

        # Spectral grid
        k1 = 2.0*pi/self.Lx
        l1 = 2.0*pi/self.Ly

        if self.realVars: 
            kk = k1*np.arange(0.0, self.nx/2.0+1.0)
        else: 
            kk = k1*np.append(np.arange(0.0, self.nx/2.0),
                np.arange(-self.nx/2.0, 0.0) )

        ll = l1*np.append(np.arange(0.0, self.ny/2.0),
            np.arange(-self.ny/2.0, 0.0) )
        
        (self.nl, self.nk) = (ll.size, kk.size)
        self.k, self.l = np.meshgrid(kk, ll)
        
        # Create tuples with shapes of physical and spectral variables
        self.physVarShape = (self.ny, self.nx)
        self.physSolnShape = (self.ny, self.nx, self.nVars)

        if self.realVars:
            self.specVarShape = (self.ny, self.nx//2+1)
            self.specSolnShape = (self.ny, self.nx//2+1, self.nVars)
        else:
            self.specVarShape = (self.ny, self.nx)
            self.specSolnShape = (self.ny, self.nx, self.nVars)

        # Initialize variables common to all doubly periodic models
        self.soln        = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.linearCoeff = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.RHS         = np.zeros(self.specSolnShape, np.dtype('complex128'))

        # Initialize the spectral filter 
        filterOrder = 4.0
        (innerK, outerK) = (0.65, 0.95)
        decayRate = 15.0*np.log(10.0) / (outerK-innerK)**filterOrder

        # Construct the filter
        self._filter = np.zeros(self.specVarShape, np.dtype('complex128'))

        nonDimK = np.sqrt( (self.k*self.dx/pi)**2.0 + (self.l*self.dy/pi)**2.0 )
        self._filter = np.exp( -decayRate * (nonDimK-innerK)**filterOrder )

        # Set filter to 1 inside filtering range and 0 outside
        self._filter[ nonDimK < innerK ] = 1.0
        self._filter[ nonDimK > outerK ] = 0.0

        # Broadcast to correct size
        self._filter = self._filter[:, :, np.newaxis] \
            * np.ones((1, 1, self.nVars))


    def _init_fft(self):
        """ Initialize fast Fourier transform routines. """

        fftwEffort = 'FFTW_MEASURE'

        if usingMKLFFT:
            if self.realVars:
                self.fft2  = mklfft.rfft2
                self.ifft2 = mklfft.irfft2
            else:
                self.fft2  = mklfft.fft2
                self.ifft2 = mklfft.ifft2
        else:
            pyfftw.interfaces.cache.enable() 
            if self.realVars:
                self.fft2 = (lambda x: pyfftw.interfaces.numpy_fft.rfft2(x, 
                                threads=self.nThreads, planner_effort=fftwEffort))
                self.ifft2 = (lambda x: pyfftw.interfaces.numpy_fft.irfft2(x, 
                                threads=self.nThreads, planner_effort=fftwEffort))
            else:
                self.fft2 = (lambda x: pyfftw.interfaces.numpy_fft.fft2(x, 
                                threads=self.nThreads, planner_effort=fftwEffort))
                self.ifft2 = (lambda x: pyfftw.interfaces.numpy_fft.ifft2(x, 
                                threads=self.nThreads, planner_effort=fftwEffort))

        # Build transform objects for user-specified transforms.
        if hasattr(self, 'forwardTransforms'):
            for var in self.forwardTransforms:
                if self.realVars:
                    setattr(self, 'fft2_'+var, pyfftw.builders.rfft2(
                        getattr(self, var), planner_effort=fftwEffort, 
                        threads=self.nThreads))

                else:
                    setattr(self, 'fft2_'+var, pyfftw.builders.fft2(
                        getattr(self, var), planner_effort=fftwEffort, 
                        threads=self.nThreads))
                    
                # Ensure that input array is contiguous and byte-aligned.
                setattr(self, var, np.ascontiguousarray(getattr(self, var)))
                pyfftw.byte_align(getattr(self, var))

        if hasattr(self, 'inverseTransforms'):
            for var in self.inverseTransforms:
                if self.realVars:
                    setattr(self, 'ifft2_'+var, pyfftw.builders.irfft2(
                        getattr(self, var), planner_effort=fftwEffort, 
                        threads=self.nThreads))
                else:
                    setattr(self, 'ifft2_'+var, pyfftw.builders.ifft2(
                        getattr(self, var), planner_effort=fftwEffort, 
                        threads=self.nThreads))

                # Ensure that input array is contiguous and byte-aligned.
                setattr(self, var, np.ascontiguousarray(getattr(self, var)))
                pyfftw.byte_align(getattr(self, var))

    
    # User inteaction - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def step_nSteps(self, nSteps=1e2, dnLog=float('inf')):
        """ Simply step forward nStep times. In some respect
            this is a legacy function """

        if not hasattr(self, 'timer'): self.timer = timeTools.time()

        for runStep in xrange(int(nSteps)):

            self._step_forward()

            if (runStep+1) % dnLog == 0.0:
                self._print_status()


    def run(self, nSteps=100, stopTime=None, outputFileName=None, runName=None,
        nLogs=0, logInterval=float('inf'),
        nPlots=0, plotInterval=float('inf'),
        nChecks=0, checkInterval=float('inf'), itemsToSave=None, 
        overwriteToSave=False, saveEndpoint=False, saveEndpointVars=None,
        calcAvgSoln=False,
        ):
        """ Step the model forward. The behavior of this method depends strongly
            on the arguments passed to it. """

        # Initialization
        if stopTime is not None: countingSteps = False
        else:                    countingSteps = True

        ## Give "nTask" arguments priority over "taskInterval" specification
        if nLogs   > 0: logInterval   = int(nSteps/nLogs)
        if nPlots  > 0: plotInterval  = int(nSteps/nPlots)
        if nChecks > 0: checkInterval = int(max(np.ceil(nSteps/nChecks), 1))

        if stopTime is not None and checkInterval < float('inf'):
            raise ValueError("Checkpointing is not allowed when "
                "integrating to a specified stopTime")
            
        # HDF5 save routine initialization
        if ( checkInterval < float('inf') or 
             itemsToSave is not None or
             saveEndpoint ):

            outputFile, runOutput = self._init_hdf5_file(outputFileName, 
                runName, overwriteToSave)

            if checkInterval < float('inf'):
                nChecks = nSteps // checkInterval + 1
                checkTime, checkData = self._init_check_datasets(runOutput, nChecks)

                # Save initial state
                (checkTime[0], checkData[:, :, :, 0]) = (self.t, self.soln)
                iSnap = 0
            
            if itemsToSave is not None:
                (itemBeingSaved, itemSaveNums, itemGroups, itemDatasets, 
                    itemTimeData) = self._init_item_datasets(runOutput, itemsToSave)

        # Averaging
        if calcAvgSoln: 
            (dt0, soln0) = (self.dt, self.soln.copy())
            if not hasattr(self, 'avgSoln'):
                self.avgSoln = np.zeros(self.specSolnShape, np.dtype('complex128'))
                self.avgTime = 0.0

        # Plot directory initialization
        if plotInterval < float('inf'):
            if not hasattr(self, 'runName'): self.runName = 'plot'
            self.plotDirectory = '{}_plots'.format(self.name)
            if not os.path.exists(self.plotDirectory):
                os.makedirs(self.plotDirectory)

        # Run
        if runName is not None:
            if countingSteps:
                print("\n(" + self.physics + ") Running '" + runName 
                    + "' for {:d} steps...".format(int(nSteps)))
            else:
                print("\n(" + self.physics + ") Running '" + runName 
                    + "' from t={:.2e} to t={:2.e}".format(
                    self.t, stopTime))

        (runStep, running, self.timer) = (0, True, timeTools.time())
        while running:

            self._step_forward()
            runStep += 1

            # Hooks for logging, plotting, saving, and averaging
            if runStep % plotInterval == 0.0: self.visualize_model_state()
            if runStep % checkInterval == 0.0:
                iSnap += 1
                (checkTime[iSnap], checkData[..., iSnap]) = (self.t, self.soln)

            if itemsToSave is not None:
                for var, saveTimes in itemsToSave.iteritems():
                    saveTimes = np.array([saveTimes]).flatten()

                    if itemBeingSaved[var] and (
                        self.t >= saveTimes[itemSaveNums[var]] or
                        saveTimes[itemSaveNums[var]]-self.t <= self.dt/2.0 
                    ):
                        itemDatasets[var][..., itemSaveNums[var]] = getattr(self, 
                            var)
                        itemTimeData[var][itemSaveNums[var]] = self.t 

                        if itemSaveNums[var]+1 == len(saveTimes):
                            itemBeingSaved[var] = False
                        else:
                            itemSaveNums[var] += 1

            if calcAvgSoln:
                self.avgTime += dt0
                self.avgSoln +=  (soln0+self.soln) / (2.0*dt0)
                (dt0, soln0) = (self.dt, self.soln.copy())

            if runStep % logInterval  == 0.0: self._print_status()

            # Assess run completion
            if countingSteps and runStep >= nSteps:            
                running = False
            elif not countingSteps:
                if self.t >= stopTime:
                    running = False

                # If the planned final step will overshoot the stopTime 
                # the run is finished with 10 small forward Euler substeps.
                elif self.t + self.dt > stopTime:
                    finalSteps = 10
                    finalDt = (stopTime - self.t) / finalSteps
                    for step in xrange(finalSteps): 
                        self._step_forward_forwardEuler(dt=finalDt)
                
                    running = False

        # Run is complete 
        if calcAvgSoln:
            self.avgSoln *= 1.0/self.avgTime

        if (saveEndpoint or 
            checkInterval < float('inf') or itemsToSave is not None):

            if saveEndpoint:
                endData = runOutput.create_group('endpoint')
                self._save_current_state(endData, vars=saveEndpointVars)

            outputFile.close()

        # Final visualization
        if plotInterval < float('inf'): 
            self.visualize_model_state()


    # Helper functions for loading and saving - - - - - - - - - - - - - - - - - 
    def init_from_endpoint(self, fileName, runName):
        """ Initialize the model from a run with saved endpoint """

        dataFile = h5py.File(fileName, 'r', libver='latest')

        if 'endpoint' not in dataFile[runName]:
            raise ValueError("The run named {} in {}".format(runName, fileName)
                + " does not have a saved enpoint.")

        # Get model input and re-initialize
        params = { param:value
            for param, value in dataFile[runName].attrs.iteritems() }

        self.__init__(**params)

        self.t = dataFile[runName]['endpoint']['t'].value
        self.set_spectral_soln(dataFile[runName]['endpoint']['soln'][:])
            
        if 'avgSoln' in dataFile[runName]['endpoint']:
            self.avgSoln = dataFile[runName]['endpoint']['avgSoln'][:]
            self.avgTime = dataFile[runName]['endpoint']['avgTime'].value

        self.evaluate_diagnostics()


    def _save_current_state(self, dataGroup, vars=None):
        """ Save the current model state """

        data = dataGroup.create_dataset('soln', data=self.soln)
        data = dataGroup.create_dataset('t',    data=self.t)
        
        if vars is not None:
            for var in vars:
                data = dataGroup.create_dataset(var, data=getattr(self, var))

        if hasattr(self, 'avgSoln'):
            data = dataGroup.create_dataset('avgSoln', data=self.avgSoln)
            data = dataGroup.create_dataset('avgTime', data=self.avgTime)


    def _init_hdf5_file(self, outputFileName, runName, overwriteToSave):
        """ Open and prepare the HDF5 file for saving run output """

        if outputFileName is None: 
            outputFileName = "{}.hdf5".format(self.name)

        outputFile = h5py.File(outputFileName, 'a', libver='latest')

        # Generate a unique runName if one is not provided.
        if runName is None:
            (defaultName, nName) = ('run{:03d}', 0)
            while '/' + defaultName.format(nName) in outputFile: nName+=1
            runName = defaultName.format(nName)

        if runName in outputFile:
            if overwriteToSave:
                del outputFile[runName]
            else:
                raise ValueError("There is existing data in {}/{}! "
                    .format(outputFileName, runName) +
                    "Find a unique runName or set overwriteToSave=True.")
                    
        runOutput = outputFile.create_group(runName)

        for param, value in self._input.iteritems(): 
            runOutput.attrs.create(param, value)

        self.outputFileName = outputFileName            
        self.runName = runName

        return outputFile, runOutput


    def _init_check_datasets(self, runOutput, nChecks):
        """ Initialize a data group 'checkpoints' with time and checkpoint 
            datasets for saving of model checkpoints during run """

        checkpoints = runOutput.create_group('checkpoints')

        checkTime = checkpoints.create_dataset('t', 
            (nChecks, ), np.dtype('float64'))

        checkData = checkpoints.create_dataset('soln', 
            (self.nl, self.nk, self.nVars, nChecks), np.dtype('complex128'))

        checkData.dims[0].label = 'l'
        checkData.dims[1].label = 'k'
        checkData.dims[2].label = 'var'
        checkData.dims[3].label = 't'

        checkTime.dims[0].label = 't'

        return checkTime, checkData


    def _init_item_datasets(self, runOutput, itemsToSave):
        """ Initialize dictionaries with parameters and hdf5 objects needed
            for the itemized saving routine """

        itemBeingSaved = dict()
        itemSaveNums   = dict()
        itemGroups     = dict()
        itemDatasets   = dict()
        itemTimeData   = dict()

        for var, saveTimes in itemsToSave.iteritems():
            saveTimes = np.array([saveTimes]).flatten()

            if saveTimes[-1] < self.t: 
                itemBeingSaved[var] = False

            else:
                itemBeingSaved[var] = True

                itemDataShape = [ dim for dim in getattr(self, var).shape ]
                itemDataShape.append(len(saveTimes))

                itemGroups[var] = runOutput.create_group(
                    '{}_snapshots'.format(var))

                itemDatasets[var] = itemGroups[var].create_dataset(
                    var, tuple(itemDataShape), np.result_type(getattr(self, var)) )

                itemTimeData[var] = itemGroups[var].create_dataset(
                    't', (len(saveTimes),) )

                itemTimeData[var].dims[0].label = 't'

                # Initialize data and itemSaveNums for each data set.
                (itemSaveNums[var], readyToSave) = (0, False)
                while not readyToSave:
                    if saveTimes[itemSaveNums[var]] > self.t + self.dt:
                        readyToSave = True

                    elif np.abs(self.t 
                            - saveTimes[itemSaveNums[var]]) <= self.dt/2.0:

                        itemTimeData[var][itemSaveNums[var]] = self.t 
                        itemDatasets[var][..., itemSaveNums[var]] = getattr(
                            self, var)

                        readyToSave = True 
                        itemSaveNums[var] += 1

                    else:
                        itemSaveNums[var] += 1

        return (itemBeingSaved, itemSaveNums, itemGroups, 
                    itemDatasets, itemTimeData)


    def delete_run_data(self, outputFileName=None, runName=None):
        """ Delete all data from a model run """

        if outputFileName is None: outputFileName = self.outputFileName
        if runName is None: runName = self.runName

        dataFile = h5py.File(outputFileName, 'a', libver='latest')

        del dataFile[runName]

    # Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _dealias_RHS(self):

        if self.useFilter:
            self.RHS *= self._filter
        elif self.realVars:
            self.RHS[self.ny//3:(2*self.ny)//3, :, :] = 0.0
            self.RHS[:, self.nx//3:, :] = 0.0
        else:
            self.RHS[self.ny//3:2*self.ny//3, :, :] = 0.0
            self.RHS[:, self.nx//3:2*self.nx//3, :] = 0.0


    def _dealias_soln(self):
        """ Dealias the Fourier transform of the soln array """

        if self.useFilter:
            self.soln *= self._filter
        elif self.realVars:
            self.soln[self.ny//3:2*self.ny//3, :, :] = 0.0
            self.soln[:, self.nx//3:, :] = 0.0
        else:
            self.soln[self.ny//3:2*self.ny//3, :, :] = 0.0
            self.soln[:, self.nx//3:2*self.nx//3, :] = 0.0
        

    def _dealias_var(self, var):
        """ Dealias the Fourier transform of a single variable """

        if self.realVars:
            var[self.ny//3:2*self.ny//3, :] = 0.0
            var[:, self.nx//3:] = 0.0
        else:
            var[self.ny//3:2*self.ny//3, :] = 0.0
            var[:, self.nx//3:2*self.nx//3] = 0.0
        
        return var

    def visualize_model_state(self):
        """ Dummy routine meant to be overridden by a physical-problem subclass
            routine that visualizes the state of a model by generating a plot to
            either appears on-screen or is saved to disk """
        pass


    def _print_status(self):
        """ Print model status """

        tc = timeTools.time() - self.timer
        print("step = {:.2e}, clock = {:.2e} s, ".format(self.step, tc) + \
                "t = {:.2e} s".format(self.t))
        self.timer = timeTools.time()


    def describe_model(self):
        print(
            "\nThis is a doubly-periodic spectral model with the following "
                "attributes:\n\n"
              + "   Domain       : {:.2e} X {:.2e} m\n".format(self.Lx, self.Ly)
              + "   Grid         : {:d} X {:d}\n".format(self.nx, self.ny)
              + "   Current time : {:.2e} s\n\n".format(self.t)
              + "The FFT scheme uses {:d} thread(s).\n".format(self.nThreads)
        )


    def set_physical_soln(self, soln):
        """ Initialize model with a physical space solution """ 

        for iVar in np.arange(self.nVars):
            self.soln[:, :, iVar] = self.fft2(soln[:, :, iVar])

        self._dealias_soln()
        self.update_state_variables()


    def set_spectral_soln(self, soln):
        """ Initialize model with a spectral space solution """ 

        self.soln = soln
        self._dealias_soln()
        self.update_state_variables()


    def _step_forward_forwardEuler(self):
        """ Step the solution forward in time """

        if dt is None: dt=self.dt

        self._calc_right_hand_side(self.soln, self.t)
        self.soln += dt*(self.RHS + self.linearCoeff*self.soln)

        self.t += dt
        self.step += 1

    def update_state_variables(self):
        """ Placeholder function to be redefined in child model """
        pass


    # Diagnostics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def add_diagnostic(self, name, function, description=None):
        """ Add a new scalar diagnostic, associated function, and English 
            description to the diagnostics dictionary """

        # Create diagnostics dictionary with the basic diagnostic 'time'
        # if it doesn't exist
        if not hasattr(self, 'diagnostics'):
            self.diagnostics = dict()
            self.diagnostics['t'] = {
                'description': 'time',
                'value'      : self.t,
                'function'   : lambda self: self.t,
            }

        self.diagnostics[name] = {
            'description': description,
            'value'      : None,
            'function'   : function,
        }


    def evaluate_diagnostics(self): 
        """ Evaluate diagnostics of the physical model """

        for diag in self.diagnostics:
            self.diagnostics[diag]['value'] = \
                self.diagnostics[diag]['function'](self)
