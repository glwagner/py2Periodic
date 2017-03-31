import numpy as np; from numpy import pi
import pyfftw, time
try:
    import mkl
    usingMKL = True
    np.use_fastnumpy = True
except ImportError:
    usingMKL = False

class model(object):
    def __init__(
            self,
            physics = None,
            nVars = 1,
            realVars = False,
            # Grid parameters
            nx = 64,
            Lx = 2.0*pi, 
            ny = None,
            Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0,                      # Fixed numerical time-step.
            step = 0,                      # Initial or current step of the model
            timeStepper = "forwardEuler",  # Time-stepping method
            nThreads = 1,                  # Number of threads for FFTW
            useFilter = False,             # Use exp filter rather than dealias
        ):

        # For convenience, use a default square, uniformly-gridded domain when 
        # user specifies only nx or Lx
        if ny is None: ny = nx
        if Ly is None: Ly = Lx

        # Assign initial parameters
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
        self.nThreads = nThreads
        self.useFilter = useFilter

        # Set time-stepping method attributes for the model
        self._describe_time_stepper = getattr(self, 
            "_describe_time_stepper_{}".format(self.timeStepper))
        self._init_time_stepper = getattr(self, 
            "_init_time_stepper_{}".format(self.timeStepper))
        self._step_forward = getattr(self, 
            "_step_forward_{}".format(self.timeStepper))

        # Initialize numpy's multithreading
        if usingMKL:
            mkl.set_num_threads(self.nThreads)

        # Call initialization routines for the doubly-periodic model
        self._init_physical_grid()
        self._init_spectral_grid()
        self._init_fft()
        self._init_shapes_and_vars()

    # Hidden methods  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _init_shapes_and_vars(self):
        """ Define shapes of physical and spectral variables and initialize
            solution, the left-side linear coeffient, and the right hand side """
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

        if self.useFilter:
            # Initialize
            self.filter = np.zeros( self.specVarShape )

            # Specify filter parameters
            filterOrder = 4.0
            cutOffK = 0.65*pi
            decayRate = 15.0*np.log(10.0) / (pi-cutOffK)**filterOrder

            # Construct the filter
            nonDimK = np.sqrt( (self.KK*self.dx)**2.0 + (self.LL*self.dy)**2.0 )
            self.filter = np.exp( -decayRate*( nonDimK-cutOffK )**filterOrder )


            # Set filter to 1 outside pseudo-ovoid filtering range
            self.filter[ np.sqrt((self.KK*self.dx)**2.0 \
                + (self.LL*self.dy)**2.0) < cutOffK ] = 1.0

            # Broadcast to correct size
            self.filter = self.filter[:, :, np.newaxis] \
                * np.ones((1, 1, self.nVars))


    def _init_physical_grid(self):
        """ Initialize the physical grid """

        # Grid spacing
        self.dx = self.Lx/self.nx
        self.dy = self.Ly/self.ny

        self.xx = np.arange(0.0, self.Lx, self.dx)
        self.yy = np.arange(0.0, self.Ly, self.dy)

        # Two-dimensional grid space arrays
        self.XX, self.YY = np.meshgrid(self.xx, self.yy)

    def _init_spectral_grid(self):
        """ Initialize the spectral grid"""

        # Grid spacing
        self.dk = 2.0*pi/self.Lx
        self.dl = 2.0*pi/self.Ly

        self.ll = self.dl*np.append(np.arange(0.0, self.ny/2),
                                        np.arange(-self.ny/2, 0.0) )
    
        if self.realVars:
            self.kk = self.dk*np.arange(0.0, self.nx/2+1)
        else:
            self.kk = self.dk*np.append(np.arange(0.0, self.nx/2),
                                        np.arange(-self.nx/2, 0.0) )

        # Two-dimensional wavenumber arrays
        self.KK, self.LL = np.meshgrid(self.kk, self.ll)

        # Pre-computed products of 1j and wavenumber arrays
        self.jKK = 1j*self.KK
        self.jLL = 1j*self.LL

    def _init_fft(self):
        """ Initialize the fast Fourier transform routine. """
        pyfftw.interfaces.cache.enable() 

        if self.realVars:
            self.fft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.rfft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
            self.ifft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.irfft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
        else:
            self.fft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.fft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
            self.ifft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.ifft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
            
    def _print_status(self):
        """ Print model status """
        tc = time.time() - self.timer
        print("step = {:.2e}, clock = {:.2e} s, ".format(self.step, tc) + \
                "t = {:.2e} s".format(self.t))
        self.timer = time.time()

    def _dealias_RHS(self):
        """ Dealias the RHS """
        if self.useFilter:
            self.RHS *= self.filter
        elif self.realVars:
            self.RHS[self.ny//3:2*self.ny//3, :, :] = 0.0
            self.RHS[:, self.nx//3:, :] = 0.0
        else:
            self.RHS[self.ny//3:2*self.ny//3, :, :] = 0.0
            self.RHS[:, self.nx//3:2*self.nx//3, :] = 0.0

    def _dealias_array(self, array):
        """ Dealias the Fourier transform of a real array """
        if self.useFilter:
            array *= self.filter
        elif self.realVars:
            array[self.nx//3:2*self.nx//3, :, :] = 0.0
            array[:, self.ny//3:, :] = 0.0
        else:
            array[self.ny//3:2*self.ny//3, :, :] = 0.0
            array[:, self.nx//3:2*self.nx//3, :] = 0.0
        
        return array

    # Visible methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def set_physical_soln(self, soln):
        """ Initialize model with a physical space solution """ 
        for iVar in np.arange(self.nVars):
            self.soln[:, :, iVar] = self.fft2(soln[:, :, iVar])

        self.soln = self._dealias_array(self.soln)
        self.update_state_variables()

    def set_spectral_soln(self, soln):
        """ Initialize model with a spectral space solution """ 
        self.soln = soln
        self.soln = self._dealias_array(self.soln)
        self.update_state_variables()

    def run_nSteps(self, nSteps=1e2, dnLog=None):
        """ Step forward nStep times """
        # Initialize run
        step0 = self.step

        if dnLog is None: dnLog = np.floor(nSteps/10.0)
         
        if step0 == 0:
            self.timer = time.time()
        elif not hasattr(self, 'timer'):
            self.timer = time.time()

        # Step forward
        while (self.step < step0+nSteps):
            self._step_forward()

            if self.step > 0 and self.step % dnLog == 0.0:
                self._print_status()

            self.t += self.dt
            self.step += 1

    def describe_model(self):
        print("\nThis is a doubly-periodic spectral model with the following " + \
                "attributes:\n\n" + \
                "   Domain       : {:.2e} X {:.2e} m\n".format(self.Lx, self.Ly) + \
                "   Resolution   : {:d} X {:d}\n".format(self.nx, self.ny) + \
                "   Timestep     : {:.2e} s\n".format(self.dt) + \
                "   Current time : {:.2e} s\n\n".format(self.t) + \
                "The FFT scheme uses {:d} thread(s).\n".format(self.nThreads))

    # Time steppers for the doublyPeriodicModel class - - - - - - - - - - - - - 
    
    ## Forward Euler  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _describe_time_stepper_forwardEuler(self):
        print("""
            The forward Euler time-stepping method is a simple 1st-order explicit \n
            method with poor stability characteristics. It is described, among \n
            other places, in Bewley's Numerical Renaissance.
              """)

    def _init_time_stepper_forwardEuler(self):
        pass

    def _step_forward_forwardEuler(self):
        """ Step the solution forward in time using the forward Euler scheme """
        self._calc_right_hand_side(self.soln, self.t)
        self.soln += self.dt*(self.RHS - self.linearCoeff*self.soln)

    ## Low-storage Runge-Kutta-Wray-theta scheme - - - - - - - - - - - - - - - - - - - 
    def _describe_time_stepper_RKW3(self):
        print("""
            RKW3, or more completely, RKW3-theta, is the 3rd-order low-storage \n
            Runge-Kutta-Wray time-stepping method. It may be semi-implicit, but \n
            that is not known right now because this method was implemented by \n
            Cesar Rocha and we do not have a reference or documentation.
              """)

    def _init_time_stepper_RKW3(self):
        """ Initialize and allocate vars for RK3W time-stepping """

        self.a1, self.a2, self.a3 = 29./96., -3./40., 1./6.
        self.b1, self.b2, self.b3 = 37./160., 5./24., 1./6.
        self.c1, self.c2, self.c3 = 8./15., 5./12., 3./4.
        self.d1, self.d2 = -17./60., -5./12.

        self.L0 = -self.dt*self.linearCoeff
        self.L1 = ( (1. + self.a1*self.L0)/(1. - self.b1*self.L0) )     
        self.L2 = ( (1. + self.a2*self.L0)/(1. - self.b2*self.L0) )
        self.L3 = ( (1. + self.a2*self.L0)/(1. - self.b3*self.L0) )

        # Allocate nonlinear terms
        self.NL1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.NL2 = np.zeros(self.specSolnShape, np.dtype('complex128'))

    def _step_forward_RKW3(self):
        """ Step the solution forward in time using the RKW3 scheme """

        self._calc_right_hand_side(self.soln, self.t)
        self.NL1 = self.RHS.copy()

        self.soln  = (self.L1*self.soln + self.c1*self.dt*self.NL1).copy()

        self._calc_right_hand_side(self.soln, self.t)
        self.NL2 = self.NL1.copy()
        self.NL1 = self.RHS.copy()

        self.soln = (self.L2*self.soln + self.c2*self.dt*self.NL1 \
                    + self.d1*self.dt*self.NL2).copy()

        self._calc_right_hand_side(self.soln, self.t)
        self.NL2 = self.NL1.copy()
        self.NL1 = self.RHS.copy()

        self.soln = (self.L3*self.soln + self.c3*self.dt*self.NL1 \
                    + self.d2*self.dt*self.NL2).copy()

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
        self.soln1 = np.zeros(self.specSolnShape, np.dtype('complex128'))

        # Allocate nonlinear terms
        self.RHS1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.RHS2 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.RHS3 = np.zeros(self.specSolnShape, np.dtype('complex128'))

    def _step_forward_RK4(self):
        """ Step the solution forward in time using the RK4 scheme """

        # Substep 1
        self._calc_right_hand_side(self.soln, self.t)
        self.RHS1 = self.RHS - self.linearCoeff*self.soln

        # Substep 2
        t1 = self.t + self.dt/2.0
        self.soln1 = self.soln + self.dt/2.0*self.RHS1

        self._calc_right_hand_side(self.soln1, t1) 
        self.RHS2 = self.RHS - self.linearCoeff*self.soln1

        # Substep 3
        self.soln1 = self.soln + self.dt/2.0*self.RHS2

        self._calc_right_hand_side(self.soln1, t1) 
        self.RHS3 = self.RHS - self.linearCoeff*self.soln1

        # Substep 4
        t1 = self.t + self.dt
        self.soln1 = self.soln + self.dt*self.RHS3

        self._calc_right_hand_side(self.soln1, t1) 
        self.RHS -= self.linearCoeff*self.soln1

        # Final step
        self.soln += self.dt*(   1.0/6.0*self.RHS1 + 1.0/3.0*self.RHS2 \
                               + 1.0/3.0*self.RHS3 + 1.0/6.0*self.RHS )

    
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
        linearCoeffDt = self.dt*self.linearCoeff
        
        # Calculate coefficients with circular line integral in complex plane
        nCirc = 32          
        rCirc = 1.0       
        circ = rCirc*np.exp(2j*pi*(np.arange(1, nCirc+1)-1/2)/nCirc) 

        # Circular contour around the point to be calculated
        zc = -linearCoeffDt[..., np.newaxis] \
                + circ[np.newaxis, np.newaxis, np.newaxis, ...]

        # Four coefficients, zeta, alpha, beta, and gamma
        self.zeta = self.dt*( \
                        (np.exp(zc/2.0) - 1.0) / zc \
                            ).mean(axis=-1)

        self.alph = self.dt*( \
                      (-4.0 - zc + np.exp(zc)*(4.0-3.0*zc+zc**2.0)) / zc**3.0 \
                            ).mean(axis=-1)

        self.beta = self.dt*( \
                      (2.0 + zc + np.exp(zc)*(-2.0+zc) ) / zc**3.0 \
                            ).mean(axis=-1)

        self.gamm = self.dt*( \
                      (-4.0 - 3.0*zc - zc**2.0 + np.exp(zc)*(4.0-zc)) / zc**3.0 \
                            ).mean(axis=-1)
                              
        # Pre-calculate an exponential     
        self.linearExpDt     = np.exp(-self.dt*self.linearCoeff)
        self.linearExpHalfDt = np.exp(-self.dt/2.0*self.linearCoeff)

        # Allocate intermediate solution variable
        self.soln1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.soln2 = np.zeros(self.specSolnShape, np.dtype('complex128'))

        # Allocate nonlinear terms
        self.NL1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.NL2 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.NL3 = np.zeros(self.specSolnShape, np.dtype('complex128'))

    def _step_forward_ETDRK4(self):
        """ Step the solution forward in time using the ETDRK4 scheme """
        self._calc_right_hand_side(self.soln, self.t)
        self.NL1 = self.RHS.copy()

        t1 = self.t + self.dt/2
        self.soln1 = self.linearExpHalfDt*self.soln + self.zeta*self.NL1
        self._calc_right_hand_side(self.soln1, t1)
        self.NL2 = self.RHS.copy()

        self.soln2 = self.linearExpHalfDt*self.soln + self.zeta*self.NL2
        self._calc_right_hand_side(self.soln2, t1)
        self.NL3 = self.RHS.copy()

        t1 = self.t + self.dt
        self.soln2 = self.linearExpHalfDt*self.soln1 + self.zeta*(2.0*self.NL3-self.NL1)
        self._calc_right_hand_side(self.soln2, t1)

        # The final step
        self.soln = self.linearExpDt*self.soln \
                    +     self.alph * self.NL1 \
                    + 2.0*self.beta * (self.NL2 + self.NL3) \
                    +     self.gamm * self.RHS

    ## 3rd-order Adams-Bashforth (AB3) - - - - - - - - - - - - - - - - - - - - - 
    def _describe_time_stepper_AB3(self):
        print("""
            AB3 is the 3rd-order explicity Adams-Bashforth scheme, which employs 
            solutions from prior time-steps to achieve higher-order accuracy   \n
            over forward Euler. Unlike a multistep method like RK4, the stability
            properties of AB methods decrease as the order increases.
              """)

    def _init_time_stepper_AB3(self):
        """ Initialize and allocate vars for RK4 time-stepping """

        # Allocate right hand sides to be stored from previous steps
        self.RHSm1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
        self.RHSm2 = np.zeros(self.specSolnShape, np.dtype('complex128'))

    def _step_forward_AB3(self):
        """ Step the solution forward in time using the AB3 scheme """

        # While RHS_{n-2} is unfilled, step forward with foward Euler.
        if not self.RHSm2.any():
            self._calc_right_hand_side(self.soln, self.t)
            self.soln += self.dt*(self.RHS - self.linearCoeff*self.soln)
        else:
            self._calc_right_hand_side(self.soln, self.t)
            self.RHS -= self.linearCoeff*self.soln

            self.soln +=   23.0/12.0 * self.dt * self.RHS \
                         - 16.0/12.0 * self.dt * self.RHSm1 \
                         +  5.0/12.0 * self.dt * self.RHSm2

        # Store RHS for use in future time-steps.
        self.RHSm2 = self.RHSm1.copy()
        self.RHSm1 = self.RHS.copy()
