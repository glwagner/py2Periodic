import numpy as np; from numpy import pi
import pyfftw
import mkl
import h5py
import time

# TODO 1. Add an exception error for the violation of physical
#           conditions, if physical parameters are unset, etc.
# TODO 2. Make more of the defaults 'None' if values for them can
#           be determined diagnostically, rather than prescribed.
# TODO 3. For example, set dt=None for default, and specify an adaptive
#           time-stepper.
# TODO 4. Allow dt to be specified at run time, and generate a warning if
#           an implicit time-stepper is specified.

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

        # TODO: permit an input 'maximum'; also, if nThreads is
        # greater than number on the machine, set to number on the machine.
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

        # Copy the model inputs into an independent dictionary prior to running 
        # initialization routines
        self._input = self.__dict__.copy()

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
        (innerK, outerK) = (0.63, 0.65)
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

    def step_nSteps(self, nSteps=1e2, dnLog=float('Inf')):
        """ Step forward nStep times """

        if not hasattr(self, 'timer'): self.timer = time.time()

        for substep in xrange(int(nSteps)):
            self._step_forward()
            if (substep+1) % dnLog == 0.0:
                self._print_status()

    def step_nSteps_and_average(self, nSteps=1e2, dnLog=float('Inf')):
        """ Step forward nStep times """

        if not hasattr(self, 'timer'): self.timer = time.time()

        # If avgSoln already exists, continue to average
        if not hasattr(self, 'avgSoln'): 
            self.avgSoln = np.zeros(self.specSolnShape, 'complex128')

        for substep in xrange(int(nSteps)):

            # Store solution and time-step prior to stepping forward
            dt0 = self.dt
            soln0 = self.soln.copy()

            self._step_forward()

            # Accumulate average using the trapezoidal rule
            self.avgSoln += (soln0+self.soln) / (2.0*dt0)

            if (substep+1) % dnLog == 0.0:
                self._print_status()


    def step_until(self, stopTime=None, dnLog=float('Inf')):
        """ Step forward nStep times """

        if not hasattr(self, 'timer'): self.timer = time.time()
        if stopTime is None: stopTime = step.t + 10.0*self.dt
        if stopTime < self.t: 
            print("\nThe stop time is less than the current time! " \
                "The model will not step forward.\n")

        substep = 0
        while True:
            if self.t >= stopTime: break
            elif self.t + self.dt > stopTime:
                # This hook handles cases where the planned final step will 
                # overshoot the stopTime. In this case, ten small steps are 
                # carried out with the forward Euler scheme to finish the run.
                finalSteps = 10
                finalDt = (stopTime - self.t) / float(finalSteps)
                for step in xrange(finalSteps): 
                    self._step_forward_forwardEuler(dt=finalDt)
                break
            else:
                self._step_forward()
                substep += 1
                if (substep+1) % dnLog == 0.0:
                    self._print_status()

    def _print_status(self):
        """ Print model status """
        tc = time.time() - self.timer
        print("step = {:.2e}, clock = {:.2e} s, ".format(self.step, tc) + \
                "t = {:.2e} s".format(self.t))
        self.timer = time.time()

    # Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def describe_model(self):
        print("\nThis is a doubly-periodic spectral model with the following " + \
                "attributes:\n\n" + \
                "   Domain       : {:.2e} X {:.2e} m\n".format(self.Lx, self.Ly) + \
                "   Grid         : {:d} X {:d}\n".format(self.nx, self.ny) + \
                "   Current time : {:.2e} s\n\n".format(self.t) + \
                "The FFT scheme uses {:d} thread(s).\n".format(self.nThreads))

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
