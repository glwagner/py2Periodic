import doublyPeriodic
import numpy as np; from numpy import pi 
import time

class model(doublyPeriodic.model):
    def __init__(self, name = "twoLayerQuasigeostrophicExample", 
            # Grid parameters
            nx = 256, ny = None, Lx = 1e6, Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0e2,                     # Numerical timestep
            step = 0, 
            timeStepper = "RK4",            # Time-stepping method
            nThreads = 1,                   # Number of threads for FFTW
            useFilter = False,
            # 
            # Two-layer parameters:
            ## Physical constants
            f0 = None,
            beta = 2.0e-11,
            defRadius = 1.5e4, 
            ## Layer-wise parameters
            H1 = 1.0e3,
            H2 = 1.0e3,
            U1 = 1.0e-1,
            U2 = 0.0,
            ## Friction parameters
            bottomDrag = 0.0,
            visc = 1.0e0,
            viscOrder = 4.0,
            ## Flag to activate bathymetry
            flatBottom = True,
        ):

        # Initialize super-class.
        doublyPeriodic.model.__init__(self, 
            physics = "two-layer quasi-geostrophic flow",
            nVars = 2, 
            realVars = True,
            # Persistant doublyPeriodic initialization arguments 
            nx = nx, ny = ny, Lx = Lx, Ly = Ly, t = t, dt = dt, step = step,
            timeStepper = timeStepper, nThreads = nThreads, useFilter = useFilter,
        )
            
        # Parameters specific to the Physical Problem
        self.name = name
        self.f0 = f0
        self.beta = beta
        self.defRadius = defRadius
        self.H1 = H1
        self.H2 = H2
        self.U1 = U1
        self.U2 = U2
        self.bottomDrag = bottomDrag
        self.visc = visc
        self.viscOrder = viscOrder
        self.flatBottom = flatBottom
            
        # Initialize variables and parameters specific to the problem
        self._init_parameters()
        self._init_linear_coeff()

        # Initialize time-stepper once linear coefficient is determined
        self._init_time_stepper()

        # Set the initial condition to default.
        self.set_physical_soln( \
            1.0e-1*np.random.standard_normal(self.physSolnShape))
        self.update_state_variables()
        
    # Methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def describe_physics(self):
        print("""
            This model solves the two-layer quasi-geostrophic equations \n
            with a variable-order hyperdissipation operator. There are \n
            two prognostic variables: the quasi-geostrophic potential \n
            vorticity in each layer.
        """)

    def _init_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """
        self.linearCoeff[:, :, 0] = self.__jk*self.U1 \
            + self.visc*(self.k**2.0 + self.l**2.0)**(self.viscOrder/2.0)

        self.linearCoeff[:, :, 1] = self.__jk*self.U2 \
            + self.visc*(self.k**2.0 + self.l**2.0)**(self.viscOrder/2.0)

    def _init_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """

        # Layer depth ratio
        self.delta = self.H1/self.H2

        # Scaled, squared deformation wavenumbers
        self.F1 = self.defRadius**(-2.0) / (1 + self.delta)
        self.F2 = self.delta * self.F1

        # Background meridional PV gradients
        self.Q1y = self.beta + self.F1*(self.U1 - self.U2)
        self.Q2y = self.beta - self.F2*(self.U1 - self.U2)

        # A square wavenumber and various products
        self.KK = self.k**2.0 + self.l**2.0

        self.__jk = 1j*self.k
        self.__jl = 1j*self.l

        self.__jkQ1y = 1j*self.k*self.Q1y
        self.__jkQ2y = 1j*self.k*self.Q2y

        self.__bottomDragKK = self.bottomDrag*self.KK

        # "One over" the determinant of the PV-streamfunction inversion matrix
        detM = self.KK*(self.KK + self.F1 + self.F2)
        detM[0, 0] = float('Inf')
        self.__oneOverDetM = 1.0/detM

        # Streamfunctions
        self.psi1h = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.psi2h = np.zeros(self.physVarShape, np.dtype('complex128'))

        # Vorticities and velocities
        self.q1  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.q2  = np.zeros(self.physVarShape, np.dtype('float64'))

        self.u1  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.u2  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v1  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v2  = np.zeros(self.physVarShape, np.dtype('float64'))

    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side """

        q1h = soln[:, :, 0]
        q2h = soln[:, :, 1]

        # Get self.psi1h and self.psi2h
        self._get_streamfunctions(q1h, q2h)

        # Vorticity and velocity in physical space.
        self.q1 = self.ifft2(q1h)
        self.q2 = self.ifft2(q2h)

        self.u1 = -self.ifft2(self.__jl*self.psi1h)
        self.v1 =  self.ifft2(self.__jk*self.psi1h)

        self.u2 = -self.ifft2(self.__jl*self.psi2h)
        self.v2 =  self.ifft2(self.__jk*self.psi2h)

        # Right Hand Side of the q1-equation
        self.RHS[:, :, 0] = -self.__jk*self.fft2( self.u1*self.q1 ) \
                                - self.__jl*self.fft2( self.v1*self.q1 ) \
                                - self.__jkQ1y*self.psi1h

        # Right Hand Side of the q2-equation
        self.RHS[:, :, 1] = -self.__jk*self.fft2( self.u2*self.q2 ) \
                                - self.__jl*self.fft2( self.v2*self.q2 ) \
                                - self.__jkQ2y*self.psi2h \
                                + self.__bottomDragKK*self.psi2h

        self._dealias_RHS()

    def update_state_variables(self):
        """ Update diagnostic variables to current model state """

        q1h = self.soln[:, :, 0]
        q2h = self.soln[:, :, 1]

        # Get streamfunctions
        self._get_streamfunctions(q1h, q2h) 
            
        # Vorticities and velocities
        self.q1 = self.ifft2(q1h)
        self.q2 = self.ifft2(q2h)

        self.u1 = -self.ifft2(self.__jl*self.psi1h)
        self.v1 =  self.ifft2(self.__jk*self.psi1h)

        self.u2 = -self.ifft2(self.__jl*self.psi2h)
        self.v2 =  self.ifft2(self.__jk*self.psi2h)

    def _get_streamfunctions(self, q1h, q2h):
        """ Calculate the streamfunctions psi1h and psi2h given the input 
            PV fields q1h and q2h """

        self.psi1h = -self.__oneOverDetM * ((self.KK + self.F2)*q1h + self.F1*q2h) 
        self.psi2h = -self.__oneOverDetM * (self.F2*q1h + (self.KK + self.F1)*q2h)
    
    def set_bathymetry(self, h):
        """ Set model bathymetry """
        pass

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
            *1.0/2.0*( (self.k**2.0+self.l**2.0)*np.abs(self.psi1h)**2.0 ).sum()
        KE2 = (self.Lx*self.Ly)/(self.nx*self.ny) \
            *1.0/2.0*( (self.k**2.0+self.l**2.0)*np.abs(self.psi2h)**2.0 ).sum()
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
                " Domain             : {:.2e} X {:.2e} m\n".format( \
                    self.Lx, self.Ly) + \
                " Grid               : {:d} X {:d}\n".format(self.nx, self.ny) + \
                " (Hyper)viscosity   : {:.2e} m^{:d}/s\n".format( \
                    self.visc, int(self.viscOrder)) + \
                " Deformation radius : {:.2e} m\n".format(self.defRadius) + \
                " Layer depth ratio  : {:.2f} \n".format(self.delta) + \
                " Comp. threads      : {:d} \n".format(self.nThreads) \
        )
