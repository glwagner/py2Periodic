from __future__ import division
import numpy as np; from numpy import pi
import time as timeTools
import pyfftw
import mkl
import h5py

class model(object):
    def __init__(self, physics = None, nVars = 1, realVars = False,
            # Grid resolution and extent
            nx = 64, ny = None, Lx = 2.0*pi, Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0e-2,                   # Fixed numerical time-step.
            step = 0,                      # Initial or current step
            timeStepper = "forwardEuler",  # Time-stepping method
            nThreads = 1,                  # Number of threads for FFTW
            useFilter = False,             # Use exp jilter rather than dealias
        ):

        # Default grid is square when user specifes only nx
        if Ly is None: Ly = Lx
        if ny is None: ny = nx

        self.physics = physics
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.nVars = nVars
        self.realVars = realVars
        self.t = t 
        self.dt = dt
        self.step = step
        self.timeStepper = timeStepper
        self.useFilter = useFilter

        if nThreads is 'maximum' or nThreads > mkl.get_max_threads(): 
            self.nThreads = mkl.get_max_threads()
        else:
            self.nThreads = nThreads

        # Set the default time-stepping method attributes for the model
        self._describe_time_stepper = getattr(self, 
            "_describe_time_stepper_{}".format(self.timeStepper))
        self._init_time_stepper = getattr(self, 
            "_init_time_stepper_{}".format(self.timeStepper))
        self._step_forward = getattr(self, 
            "_step_forward_{}".format(self.timeStepper))

        # Initialize fastnumpy and numpy multithreading
        np.use_fastnumpy = True
        mkl.set_num_threads(self.nThreads)


    # Initialization methods  - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _init_model(self):
        """ Run various initialization routines """

        # Store a dictionary with input parameters
        self._input = { key:value for key, value in self.__dict__.items()
            if type(value) in (str, float, int, bool) }

        # Initialization routines defined in doublyPeriodic Base Class 
        self._init_numerical_parameters()
        self._init_fft()

        # Initialization routines defined in the physical problem's subclass
        self._init_problem_parameters()
        self._init_linear_coeff()

        # Initialize time-stepper once linear coefficient is determined
        self._init_time_stepper()


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
        """ Initialize the fast Fourier transform routine. """

        pyfftw.interfaces.cache.enable() 

        if self.realVars:
            self.fft2 = (lambda x:
                    pyfftw.interfaces.numpy_fft.rfft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
            self.ifft2 = (lambda x:
                    pyfftw.interfaces.numpy_fft.irfft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
        else:
            self.fft2 = (lambda x:
                    pyfftw.interfaces.numpy_fft.fft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
            self.ifft2 = (lambda x:
                    pyfftw.interfaces.numpy_fft.ifft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))


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
            

    # Methods for model operation - - - - - - - - - - - - - - - - - - - - - - - 
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
        nSnaps=0, snapInterval=float('inf'), itemsToSave=None, overwrite=False,
        calcAvgSoln=False,
        ):
        """ Step the model forward. The behavior of this method depends strongly
            on the arguments passed to it. """

        # Initialization
        if stopTime is not None: countingSteps = False
        else:                    countingSteps = True
            
        ## Give "nTask" arguments priority over "taskInterval" specification
        if nLogs  > 0: logInterval  = int(nSteps/nLogs)
        if nPlots > 0: plotInterval = int(nSteps/nPlots)
        if nSnaps > 0: snapInterval = int(max(np.ceil(nSteps/nSnaps), 1))
            
        ## HDF5 save routine initialization
        if snapInterval < float('inf') or itemsToSave is not None:
            outputFile, runOutput = self._init_hdf5_file(outputFileName, 
                runName, overwrite)

            if snapInterval < float('inf'):
                nSnaps = int(nSteps / snapInterval)
                snapTime, snapData = self._init_snapshot_datasets(runOutput, nSnaps)

                # Save initial state
                snapTime[0] = self.t
                snapData[:, :, :, 0] = self.soln
                iSnap = 0
            
            if itemsToSave is not None:
                (itemBeingSaved, itemSaveNums, itemGroups, itemDatasets, 
                    itemTimeData) = self._init_itemized_saving(runOutput, itemsToSave)

        ## Averaging
        if calcAvgSoln: 
            (dt0, soln0) = (self.dt, self.soln.copy())
            self.avgSoln = np.zeros(self.specSolnShape, np.dtype('complex128'))
            self.avgTime = 0.0

        # Run
        (runStep, running, self.timer) = (0, True, timeTools.time())
        while running:

            self._step_forward()
            runStep += 1

            # Hooks for logging, plotting, saving, and averaging
            if runStep % logInterval  == 0.0: self._print_status()
            if runStep % plotInterval == 0.0: self.visualize_model_state()
            if runStep % snapInterval == 0.0:
                iSnap += 1
                (snapTime[iSnap], snapData[..., iSnap]) = (self.t, self.soln)

            if itemsToSave is not None:
                for var, saveTimes in itemsToSave.iteritems():
                    if itemBeingSaved[var] and (
                        self.t >= saveTimes[itemSaveNums[var]] or
                        saveTimes[itemSaveNums[var]]-self.t <= self.dt/2.0 
                    ):
                        itemDatasets[var][..., itemSaveNums[var]] = getattr(self, var)
                        itemTimeData[var][itemSaveNums[var]] = self.t 

                        if itemSaveNums[var]+1 == len(saveTimes):
                            itemBeingSaved[var] = False
                        else:
                            itemSaveNums[var] += 1

            if calcAvgSoln:
                self.avgSoln += self.avgTime / (self.avgTime+dt0) \
                    * (soln0+self.soln) / (2.0*dt0)
                self.avgTime += dt0
                (dt0, soln0) = (self.dt, self.soln.copy())

            # Assess run completion
            if countingSteps and runStep >= nSteps:            
                running = False
            elif not countingSteps:
                if self.t >= stopTime:
                    running = False
                elif self.t + self.dt > stopTime:
                    # This hook handles cases where the planned final step will 
                    # overshoot the stopTime. In this case, 10 small steps are 
                    # carried out with the forward Euler scheme to finish the run.
                    finalSteps = 10
                    finalDt = (stopTime - self.t) / finalSteps
                    for step in xrange(finalSteps): 
                        self._step_forward_forwardEuler(dt=finalDt)
                
                    running = False

        # Run is complete. Finishing up by saving final state if 
        # not already saved, and closing file.
        if snapInterval < float('inf') or itemsToSave is not None:
            if nSnaps > 0 and iSnap < nSnaps:
                iSnap += 1
                snapTime[iSnap] = self.t
                snapData[..., iSnap] = self.soln

            outputFile.close()


    # Helper functions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _init_hdf5_file(self, outputFileName, runName, overwrite):
        """ Open and prepare the HDF5 file for saving run output """

        if outputFileName is None: outputFileName = self.name

        outputFile = h5py.File("{}.hdf5".format(outputFileName), 
            'a', libver='latest')

        # Generate a unique runName if one is not provided.
        # TODO: Spit warning if nDefault > 99
        if runName is None:
            defaultName = 'test'
            nDefault = 0
            while '/{}_{:02d}'.format(defaultName, nDefault) in outputFile:
                nDefault += 1
            runName = '{}_{:02d}'.format(defaultName, nDefault)

        # Create new group for the run and store inputs as attributes.
        # Delete output file is 'overwrite' is selected.
        if overwrite and runName in outputFile:
            del outputFile[runName]

        runOutput = outputFile.create_group(runName)

        for param, value in self._input.iteritems(): 
            runOutput.attrs.create(param, value)

        self.outputFileName = outputFileName            
        self.runName = runName

        return outputFile, runOutput


    def delete_run_data(self, outputFileName=None, runName=None):
        """ Delete all data from a model run """

        if outputFileName is None: outputFileName = self.outputFileName
        if runName is None: runName = self.runName

        dataFile = h5py.File("{}.hdf5".format(outputFileName), 
            'a', libver='latest')

        del dataFile[runName]


    def _init_snapshot_datasets(self, runOutput, nSnaps):
        """ Initialize a data group 'snapshots' with time and snapshot 
            datasets for saving of model snapshots during run """

        snapshots = runOutput.create_group('snapshots')
        snapTime = snapshots.create_dataset('t', 
            (nSnaps+1, ), np.dtype('float64'))
        snapData = snapshots.create_dataset('soln', 
            (self.nl, self.nk, self.nVars, nSnaps+1), np.dtype('complex128'))

        snapData.dims[0].label = 'l'
        snapData.dims[1].label = 'k'
        snapData.dims[2].label = 'var'
        snapData.dims[3].label = 't'

        snapTime.dims[0].label = 't'

        return snapTime, snapData


    def _init_itemized_saving(self, runOutput, itemsToSave):
        """ Initialize dictionaries with parameters and hdf5 objects needed
            for the itemized saving routine """

        # TODO: spit warning if the specified time points are not in order.
        itemBeingSaved = dict()
        itemSaveNums = dict()
        itemGroups = dict()
        itemDatasets = dict()
        itemTimeData = dict()

        for var, saveTimes in itemsToSave.iteritems():
            if saveTimes[-1] < self.t: 
                itemBeingSaved[var] = False
            else:
                itemBeingSaved[var] = True
                itemDataShape = [ dim for dim in getattr(self, var).shape ]
                itemDataShape.append(len(saveTimes))

                itemGroups[var] = runOutput.create_group('{}_data'.format(var))
                itemDatasets[var] = itemGroups[var].create_dataset(
                    var, tuple(itemDataShape) )
                itemTimeData[var] = itemGroups[var].create_dataset(
                    't', (len(saveTimes),) )

                # Initialize data and itemSaveNums for each data set.
                (itemSaveNums[var], readyToSave) = (0, False)
                while not readyToSave:
                    if saveTimes[itemSaveNums[var]] > self.t + self.dt:
                        readyToSave = True
                    elif np.abs(self.t-saveTimes[itemSaveNums[var]]) <= self.dt/2.0:
                        itemDatasets[var][..., itemSaveNums[var]] = getattr(self, var)
                        itemTimeData[var][itemSaveNums[var]] = self.t 

                        readyToSave = True 
                        itemSaveNums[var] += 1
                    else:
                        itemSaveNums[var] += 1

        return (itemBeingSaved, itemSaveNums, itemGroups, itemDatasets, itemTimeData)


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

        print( \
            "\nThis is a doubly-periodic spectral model with the following " + \
                "attributes:\n\n" + \
                "   Domain       : {:.2e} X {:.2e} m\n".format(self.Lx, self.Ly) + \
                "   Grid         : {:d} X {:d}\n".format(self.nx, self.ny) + \
                "   Current time : {:.2e} s\n\n".format(self.t) + \
                "The FFT scheme uses {:d} thread(s).\n".format(self.nThreads) \
        )


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


    # Time steppers for the doublyPeriodicModel class - - - - - - - - - - - - - 
    ## Forward Euler  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _describe_time_stepper_forwardEuler(self):
        print("""
            The forward Euler time-stepping method is a simple 1st-order 
            explicit method with poor stability characteristics. It is \n
            described, among other places, in Bewley's Numerical Renaissance.\n
              """)


    def _init_time_stepper_forwardEuler(self):
        pass


    def _step_forward_forwardEuler(self, dt=None):
        """ Step the solution forward in time using the forward Euler scheme """
        if dt is None: dt=self.dt

        self._calc_right_hand_side(self.soln, self.t)
        self.soln += dt*(self.RHS + self.linearCoeff*self.soln)
        self.t += dt
        self.step += 1


    ## 4th-order Runge-Kutta (RK4)  - - - - - - - - - - - - - - - - - - - - - - 
    def _describe_time_stepper_RK4(self):
        print("""
            RK4 is the classical explicit 4th-order Runge-Kutta time-stepping \n
            method. It uses a series of substeps/estimators to achieve 4th-order \n 
            accuracy over each individual time-step, at the cost of requiring \n
            relatively more evaluations of the nonlinear right hand side. \n
            It is described, among other places, in Bewley's Numerical \n
            Renaissance.
              """)


    def _init_time_stepper_RK4(self):
        """ Initialize and allocate vars for RK4 time-stepping """

        # Allocate intermediate solution variable
        self.__soln1 = np.zeros(self.specSolnShape, np.dtype('complex128'))

        # Allocate nonlinear terms
        self.__RHS1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.__RHS2 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.__RHS3 = np.zeros(self.specSolnShape, np.dtype('complex128'))


    def _step_forward_RK4(self, dt=None):
        """ Step the solution forward in time using the RK4 scheme """

        if dt is None: dt=self.dt

        # Substep 1
        self._calc_right_hand_side(self.soln, self.t)
        self.__RHS1 = self.RHS + self.linearCoeff*self.soln

        # Substep 2
        t1 = self.t + dt/2.0
        self.__soln1 = self.soln + dt/2.0*self.__RHS1

        self._calc_right_hand_side(self.__soln1, t1) 
        self.__RHS2 = self.RHS + self.linearCoeff*self.__soln1

        # Substep 3
        self.__soln1 = self.soln + dt/2.0*self.__RHS2

        self._calc_right_hand_side(self.__soln1, t1) 
        self.__RHS3 = self.RHS + self.linearCoeff*self.__soln1

        # Substep 4
        t1 = self.t + dt
        self.__soln1 = self.soln + dt*self.__RHS3

        self._calc_right_hand_side(self.__soln1, t1) 
        self.RHS += self.linearCoeff*self.__soln1

        # Final step
        self.soln += dt*(   1.0/6.0*self.__RHS1 + 1.0/3.0*self.__RHS2 \
                          + 1.0/3.0*self.__RHS3 + 1.0/6.0*self.RHS )
        self.t += dt
        self.step += 1

    
    ## 4th Order Runge-Kutta Exponential Time Differenceing (ETDRK4)  - - - - - 
    def _describe_time_stepper_ETDRK4(self):
        print("""
            ETDRK4 is a 4th-order Runge-Kutta exponential time-differencing \n
            method described by Cox and Matthews (2002). The prefactors are \n
            computed by contour integration in the complex plane, as described \n
            by Kassam and Trefethen (2005).
              """)

        
    def _init_time_stepper_ETDRK4(self):
        """ Initialize and allocate vars for ETDRK4 time-stepping """

        # Calculate coefficients with circular line integral in complex plane
        nCirc = 32          
        rCirc = 1.0       
        circ = rCirc*np.exp(2j*pi*(np.arange(1, nCirc+1)-1/2)/nCirc) 

        # Circular contour around the point to be calculated
        linearCoeffDt = self.dt*self.linearCoeff
        zc = linearCoeffDt[..., np.newaxis] \
                + circ[np.newaxis, np.newaxis, np.newaxis, ...]

        # Four coefficients, zeta, alpha, beta, and gamma
        self.__zeta = self.dt*( \
                        (np.exp(zc/2.0) - 1.0) / zc \
                            ).mean(axis=-1)

        self.__alph = self.dt*( \
                      (-4.0 - zc + np.exp(zc)*(4.0-3.0*zc+zc**2.0)) / zc**3.0 \
                            ).mean(axis=-1)

        self.__beta = self.dt*( \
                      (2.0 + zc + np.exp(zc)*(-2.0+zc) ) / zc**3.0 \
                            ).mean(axis=-1)

        self.__gamm = self.dt*( \
                      (-4.0 - 3.0*zc - zc**2.0 + np.exp(zc)*(4.0-zc)) / zc**3.0 \
                            ).mean(axis=-1)
                              
        # Pre-calculate an exponential     
        self.__linearExpDt     = np.exp(self.dt*self.linearCoeff)
        self.__linearExpHalfDt = np.exp(self.dt/2.0*self.linearCoeff)

        # Allocate intermediate solution variable
        self.__soln1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.__soln2 = np.zeros(self.specSolnShape, np.dtype('complex128'))

        # Allocate nonlinear terms
        self.__NL1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.__NL2 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.__NL3 = np.zeros(self.specSolnShape, np.dtype('complex128'))


    def _step_forward_ETDRK4(self):
        """ Step the solution forward in time using the ETDRK4 scheme """

        self._calc_right_hand_side(self.soln, self.t)
        self.__NL1 = self.RHS.copy()

        t1 = self.t + self.dt/2
        self.__soln1 = self.__linearExpHalfDt*self.soln + self.__zeta*self.__NL1
        self._calc_right_hand_side(self.__soln1, t1)
        self.__NL2 = self.RHS.copy()

        self.__soln2 = self.__linearExpHalfDt*self.soln + self.__zeta*self.__NL2
        self._calc_right_hand_side(self.__soln2, t1)
        self.__NL3 = self.RHS.copy()

        t1 = self.t + self.dt
        self.__soln2 = self.__linearExpHalfDt*self.__soln1 \
            + self.__zeta*(2.0*self.__NL3-self.__NL1)
        self._calc_right_hand_side(self.__soln2, t1)

        # The final step
        self.soln = self.__linearExpDt*self.soln \
                    +     self.__alph * self.__NL1 \
                    + 2.0*self.__beta * (self.__NL2 + self.__NL3) \
                    +     self.__gamm * self.RHS
        self.t += self.dt
        self.step += 1


    ## 3rd-order Adams-Bashforth (AB3) - - - - - - - - - - - - - - - - - - - - - 
    def _describe_time_stepper_AB3(self):
        print("""
            AB3 is the 3rd-order explicity Adams-Bashforth scheme, which employs 
            solutions from prior time-steps to achieve higher-order accuracy   \n
            over forward Euler. AB3 is faster, but has a smaller linear \n
            stability region compared to RK4.
              """)


    def _init_time_stepper_AB3(self):
        """ Initialize and allocate vars for AB3 time-stepping """

        # Allocate right hand sides to be stored from previous steps
        self.__RHSm1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.__RHSm2 = np.zeros(self.specSolnShape, np.dtype('complex128'))


    def _step_forward_AB3(self):
        """ Step the solution forward in time using the AB3 scheme """

        # While RHS_{n-2} is unfilled, step forward with foward Euler.
        if not self.__RHSm2.any():
            self._calc_right_hand_side(self.soln, self.t)
            self.soln += self.dt*(self.RHS + self.linearCoeff*self.soln)
        else:
            self._calc_right_hand_side(self.soln, self.t)
            self.RHS += self.linearCoeff*self.soln

            self.soln +=   23.0/12.0 * self.dt * self.RHS \
                         - 16.0/12.0 * self.dt * self.__RHSm1 \
                         +  5.0/12.0 * self.dt * self.__RHSm2

        # Store RHS for use in future time-steps.
        self.__RHSm2 = self.__RHSm1.copy()
        self.__RHSm1 = self.RHS.copy()

        self.t += self.dt
        self.step += 1
