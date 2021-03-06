import doublyPeriodic
import numpy as np; from numpy import pi 
import time

class model(doublyPeriodic.numerics):
    def __init__(
            self,
            name = "twoDimensionalTurbulenceExample", 
            # Grid parameters
            nx = 128,
            Lx = 2.0*pi,
            ny = None,
            Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0e-2,                    # Numerical timestep
            step = 0, 
            timeStepper = "forwardEuler",   # Time-stepping method
            nThreads = 1,                   # Number of threads for FFTW
            # 
            # Two-dimensional turbulence parameters: arbitrary-order viscosity
            visc = 1.0e-4,
            viscOrder = 2.0,
        ):

        # Initialize super-class.
        #super(model, self).__init__(self, 
        doublyPeriodic.numerics.__init__(self, 
            physics = "two-dimensional turbulence",
            nVars = 1, 
            realVars = True,
            # Grid parameters
            nx = nx,
            ny = ny,
            Lx = Lx,
            Ly = Ly,
            # Solver parameters
            t  = t,   
            dt = dt,                        # Numerical timestep
            step = step,                    # Current step
            timeStepper = timeStepper,      # Time-stepping method
            nThreads = nThreads,            # Number of threads for FFTW
        )

        # Parameters specific to the Physical Problem
        self.name = name
        self.visc = visc
        self.viscOrder = viscOrder
            
        # Initial routines
        ## Initialize variables and parameters specific to this problem
        self._init_parameters()
        self._set_linear_coeff()
        self._init_time_stepper()

        # Set the initial condition to default.
        self.set_physical_soln(0.1*np.random.standard_normal(self.physSolnShape))
        self.update_state_variables()
        
    # Methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def describe_physics(self):
        print("""
            This model solves the two-dimensional Navier-Stokes equation in \n
            streamfunction-vorticity formulation, with a variable-order \n
            hyperdissipation operator. There is a single prognostic solution \n
            variable --- vorticity in Fourier space.
        """)

    def _set_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """
        # The default model is 2D turbulence with Laplacian viscosity.
        self.linearCoeff[:, :, 0] = self.visc*(self.KK**2.0 + self.LL**2.0)
       
    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side of the equation """
        # View for clarity:
        qh = soln[:, :, 0]

        # Streamfunction
        self.psih = - qh / self.divideSafeKay2 

        # Vorticity and velocity
        self.q = self.ifft2(qh)
        self.u = -self.ifft2(self.jLL*self.psih)
        self.v =  self.ifft2(self.jKK*self.psih)

        self.RHS[:, :, 0] = - self.jKK*self.fft2(self.u*self.q) \
                                - self.jLL*self.fft2(self.v*self.q) 

        self._dealias_RHS()
         
    def _init_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """
        # Divide-safe square wavenumber
        self.divideSafeKay2 = self.KK**2.0 + self.LL**2.0
        self.divideSafeKay2[0, 0] = float('Inf')

        # Streamfunction
        self.psih = np.zeros(self.physVarShape, np.dtype('complex128'))

        # Vorticity and velocity
        self.q  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.u  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v  = np.zeros(self.physVarShape, np.dtype('float64'))
            
    def update_state_variables(self):
        """ Update diagnostic variables to current model state """
        # View for clarity:
        qh = self.soln[:, :, 0]

        # Streamfunction
        self.psih = - qh / self.divideSafeKay2 

        # Vorticity and velocity
        self.q = self.ifft2(qh)
        self.u = -self.ifft2(self.jLL*self.psih)
        self.v =  self.ifft2(self.jKK*self.psih)

    def describe_model(self):
        """ Describe the current model state """

        print("\nThis is a doubly-periodic spectral model for \n" + \
                "{:s} \n".format(self.physics) + \
                "with the following attributes:\n\n" + \
                "   Domain       : {:.2e} X {:.2e} m\n".format(self.Lx, self.Ly) + \
                "   Resolution   : {:d} X {:d}\n".format(self.nx, self.ny) + \
                "   Timestep     : {:.2e} s\n".format(self.dt) + \
                "   Current time : {:.2e} s\n\n".format(self.t) + \
                "The FFT scheme uses {:d} thread(s).\n".format(self.nThreads) \
        )

