import doublyPeriodic
import numpy as np; from numpy import pi 
import time

class model(doublyPeriodic.model):
    def __init__(
            self,
            name = "twoLayerQuasigeostrophicExample", 
            # Grid parameters
            nx = 256,
            ny = None,
            Lx = 1.0e6,
            Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0e2,                     # Numerical timestep
            step = 0, 
            timeStepper = "RK4",            # Time-stepping method
            nThreads = 1,                   # Number of threads for FFTW
            # 
            # Two-layer parameters:
            ## Physical constants
            beta = 2.0e-11,
            defRadius = 1.5e4, 
            ## Layer-wise parameters
            H1 = 1.0e3,
            H2 = 1.0e3,
            U1 = 1.0e-1,
            U2 = 0.0,
            ## Friction parameters
            drag = 0.0,
            visc = 1.0e0,
            viscOrder = 4.0,
        ):

        # Initialize super-class.
        doublyPeriodic.model.__init__(self, 
            physics = "two-layer quasi-geostrophic flow",
            nVars = 2, 
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
        self.beta = beta
        self.defRadius = defRadius
        self.H1 = H1
        self.H2 = H2
        self.U1 = U1
        self.U2 = U2
        self.drag = drag
        self.visc = visc
        self.viscOrder = viscOrder
            
        # Initial routines
        ## Initialize variables and parameters specific to this problem
        self._init_parameters()
        self._set_linear_coeff()
        self._init_time_stepper()

        # Set the initial condition to default.
        self.set_physical_soln( \
            1.0e-1*np.random.standard_normal(self.physSolnShape))
        self.update_state_variables()
        
    # Methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def describe_physics(self):
        print("""
            This model solves the two-layer quasi-geostrophic equations \n
            with a variable-order hyperdissipation operator.
        """)

    def _set_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """
        self.linearCoeff[:, :, 0] = self.jKK*self.U1 \
            + self.visc * (self.KK**2.0 + self.LL**2.0)**(self.viscOrder/2.0)

        self.linearCoeff[:, :, 1] = self.jKK*self.U2 \
            + self.visc * (self.KK**2.0 + self.LL**2.0)**(self.viscOrder/2.0)

    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side of the equation """

        q1h = soln[:, :, 0]
        q2h = soln[:, :, 1]

        # Get self.psi1h and self.psi2h
        self._get_streamfunctions(q1h, q2h)

        # Vorticity and velocity in real space.
        self.q1 = self.ifft2(q1h)
        self.q2 = self.ifft2(q2h)

        self.u1 = -self.ifft2(self.jLL*self.psi1h)
        self.v1 =  self.ifft2(self.jKK*self.psi1h)

        self.u2 = -self.ifft2(self.jLL*self.psi2h)
        self.v2 =  self.ifft2(self.jKK*self.psi2h)

        # RHS of the q1-equation
        self.RHS[:, :, 0] = -self.jKK*self.fft2( self.u1*self.q1 ) \
                                - self.jLL*self.fft2( self.v1*self.q1 ) \
                                - self.jKK*self.psi1h*self.Q1y

        # RHS of the q2-equation
        self.RHS[:, :, 1] = -self.jKK*self.fft2( self.u2*self.q2 ) \
                                - self.jLL*self.fft2( self.v2*self.q2 ) \
                                - self.jKK*self.psi2h*self.Q2y \
                                + self.drag*self.kay2*self.psi2h

        self._dealias_RHS()
         
    def _init_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """

        # Derivative physical parameters for two-layer flow
        ## Layer depth ratio
        self.delta = self.H1/self.H2

        ## Scaled, squared deformation wavenumbers
        self.F1 = self.defRadius**(-2.0) / (1 + self.delta)
        self.F2 = self.delta * self.F1

        ## Background meridional PV gradients
        self.Q1y = self.beta + self.F1*(self.U1 - self.U2)
        self.Q2y = self.beta - self.F2*(self.U1 - self.U2)

        # Square wavenumbers
        self.kay2 = self.KK**2.0 + self.LL**2.0

        # "One over" the determinant of the PV-streamfunction inversion matrix
        detM = self.kay2*(self.kay2 + self.F1 + self.F2)
        detM[0, 0] = float('Inf')
        self.oneOverDetM = 1.0/detM

        # Streamfunction
        self.psi1h = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.psi2h = np.zeros(self.physVarShape, np.dtype('complex128'))

        # Vorticity and velocity
        self.q1  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.q2  = np.zeros(self.physVarShape, np.dtype('float64'))

        self.u1  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.u2  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v1  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v2  = np.zeros(self.physVarShape, np.dtype('float64'))

    def update_state_variables(self):
        """ Update diagnostic variables to current model state """
        # View for clarity:
        q1h = self.soln[:, :, 0]
        q2h = self.soln[:, :, 1]

        # Streamfunction
        self._get_streamfunctions(q1h, q2h) 
            
        # Vorticities and velocities
        self.q1 = self.ifft2(q1h)
        self.q2 = self.ifft2(q2h)

        self.u1 = -self.ifft2(self.jLL*self.psi1h)
        self.v1 =  self.ifft2(self.jKK*self.psi1h)

        self.u2 = -self.ifft2(self.jLL*self.psi2h)
        self.v2 =  self.ifft2(self.jKK*self.psi2h)

    def _get_streamfunctions(self, q1h, q2h):
        """ Calculate the streamfunctions psi1h and psi2h given the input 
            PV fields q1h and q2h """
        self.psi1h = -self.oneOverDetM * ( (self.kay2 + self.F2)*q1h + self.F1*q2h ) 
        self.psi2h = -self.oneOverDetM * ( self.F2*q1h + (self.kay2 + self.F1)*q2h )

    def set_q1_and_q2(self, q1, q2):
        """ Update the model state by setting q1 and q2 and calculating 
            state variables """
        self.soln[:, :, 0] = self.fft2(q1)
        self.soln[:, :, 1] = self.fft2(q2)
        self.soln = self._dealias_array(self.soln)
        self.update_state_variables()

    def set_q1(self, q1):
        """ Update the model state by setting q1 and calculating 
            state variables """
        self.soln[:, :, 0] = self.fft2(q1)
        self.soln = self._dealias_array(self.soln)
        self.update_state_variables()

    def set_q2(self, q2):
        """ Update the model state by setting q2 and calculating 
            state variables """
        self.soln[:, :, 1] = self.fft2(q2)
        self.soln = self._dealias_array(self.soln)
        self.update_state_variables()

    def _print_status(self):
        """ Print model status """
        tc = time.time() - self.timer

        # Calculate kinetic energy
        self.update_state_variables() 
        KE1 = (self.Lx*self.Ly)/(self.nx*self.ny) \
            *1.0/2.0*( (self.KK**2.0+self.LL**2.0)*np.abs(self.psi1h)**2.0 ).sum()
        KE2 = (self.Lx*self.Ly)/(self.nx*self.ny) \
            *1.0/2.0*( (self.KK**2.0+self.LL**2.0)*np.abs(self.psi2h)**2.0 ).sum()
        KE = KE1 + KE2

        # Calculate CFL number
        maxSpeed1 = (np.sqrt(self.u1**2.0 + self.v1**2.0)).max()
        maxSpeed2 = (np.sqrt(self.u2**2.0 + self.v2**2.0)).max()

        CFL1 = maxSpeed1 * self.dt * self.nx/self.Lx
        CFL2 = maxSpeed2 * self.dt * self.nx/self.Lx
    
        print( \
            "step = {:.2e}, clock = {:.2e} s, ".format(self.step, tc) + \
            "t = {:.2e} s, KE = {:.2e}, ".format(self.t, KE) + \
            "CFL1 = {:.3f}, CFL2 = {:.3f}".format(CFL1, CFL2) \
        )

        self.timer = time.time()

    def describe_model(self):
        """ Describe the current model state """

        print("\nThis is a doubly-periodic spectral model for \n" + \
                "{:s} \n".format(self.physics) + \
                "with the following attributes:\n\n" + \
                "   Domain             : {:.2e} X {:.2e} m\n".format(self.Lx, self.Ly) + \
                "   Resolution         : {:d} X {:d}\n".format(self.nx, self.ny) + \
                "   Deformation radius : {:.2e} m\n".format(self.defRadius) + \
                "   Layer depth ratio  : {:.2f} \n".format(self.delta) + \
                "   Timestep           : {:.2e} s\n".format(self.dt) + \
                "   Current time       : {:.2e} s\n\n".format(self.t) + \
                "The FFT scheme uses {:d} thread(s).\n".format(self.nThreads) \
        )

