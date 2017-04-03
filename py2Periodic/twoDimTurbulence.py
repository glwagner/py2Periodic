import doublyPeriodic
import numpy as np; from numpy import pi 
import time

class model(doublyPeriodic.model):
    def __init__(self, name = "twoDimensionalTurbulenceExample", 
            # Grid parameters
            nx = 128, ny = None, Lx = 2.0*pi, Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0e-2,                   # Numerical timestep
            step = 0,                      # Initial or current step of the model
            timeStepper = "forwardEuler",  # Time-stepping method
            nThreads = 1,                  # Number of threads for FFTW
            useFilter = False,             # Use exp filter rather than dealias
            # 
            # Two-dimensional turbulence parameters: arbitrary-order viscosity
            visc = 1.0e-4,
            viscOrder = 2.0,
        ):

        doublyPeriodic.model.__init__(self, 
            physics = "two-dimensional turbulence",
            nVars = 1,
            realVars = True,
            # Persistant doublyPeriodic initialization arguments 
            nx = nx, ny = ny, Lx = Lx, Ly = Ly, t = t, dt = dt, step = step,
            timeStepper = timeStepper, nThreads = nThreads, useFilter = useFilter,
        )

        # Scalar attributes specific to the Physical Problem
        self.name = name
        self.visc = visc
        self.viscOrder = viscOrder
            
        # Initialize variables and parameters specific to the problem
        self._init_problem_parameters()
        self._init_linear_coeff()

        # Initialize time-stepper once linear coefficient is determined
        self._init_time_stepper()

        # Set a default initial condition
        self.set_physical_soln( \
            0.1*np.random.standard_normal(self.physSolnShape))
        self.update_state_variables()
        
    # Methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def describe_physics(self):
        print("""
            This model solves the two-dimensional Navier-Stokes equation in \n
            streamfunction-vorticity formulation, with a variable-order \n
            hyperdissipation operator. There is a single prognostic \n
            variable: vorticity in Fourier space.
        """)

    def _init_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """

        self.linearCoeff[:, :, 0] = \
            self.visc*(self.KK**2.0 + self.LL**2.0)**(self.viscOrder/2.0)
       
    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side """

        qh = soln[:, :, 0]

        # Transform of streamfunction and physical vorticity and velocity
        self.psih = - qh / self.divideSafeKay2 
        self.q = self.ifft2(qh)
        self.u = -self.ifft2(self.jLL*self.psih)
        self.v =  self.ifft2(self.jKK*self.psih)

        self.RHS[:, :, 0] = -self.jKK*self.fft2(self.u*self.q) \
                                - self.jLL*self.fft2(self.v*self.q) 

        self._dealias_RHS()
         
    def _init_problem_parameters(self):
        """ Pre-allocate parameters in memory """

        self.jKK = 1j*self.KK
        self.jLL = 1j*self.LL

        # Divide-safe square wavenumber magnitude
        self.divideSafeKay2 = self.KK**2.0 + self.LL**2.0
        self.divideSafeKay2[0, 0] = float('Inf')

        # Transformed streamfunction and physical vorticity and velocity
        self.psih = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.q = np.zeros(self.physVarShape, np.dtype('float64'))
        self.u = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v = np.zeros(self.physVarShape, np.dtype('float64'))
            
    def update_state_variables(self):
        """ Update diagnostic variables to current model state """

        qh = self.soln[:, :, 0]

        # Transform of streamfunction and physical vorticity and velocity
        self.psih = - qh / self.divideSafeKay2 
        self.q = self.ifft2(qh)
        self.u = -self.ifft2(self.jLL*self.psih)
        self.v =  self.ifft2(self.jKK*self.psih)

    def set_q(self, q):
        self.soln[:, :, 0] = self.fft2(q)
        self.soln = self._dealias_array(self.soln)
        self.update_state_variables()

    def describe_model(self):
        print("\nThis is a doubly-periodic spectral model for \n" + \
                "{:s} \n".format(self.physics) + \
                "with the following attributes:\n\n" + \
                " Domain           : {:.2e} X {:.2e} m\n".format( \
                    self.Lx, self.Ly) + \
                " Grid             : {:d} X {:d}\n".format(self.nx, self.ny) + \
                " (Hyper)viscosity : {:.2e} m^{:d}/s\n".format( \
                    self.visc, int(self.viscOrder)) + \
                " Comp. threads    : {:d} \n".format(self.nThreads) \
        )
