import numpy as np
import time

from ..doublyPeriodic import doublyPeriodicModel
from numpy import pi 

class model(doublyPeriodicModel):
    def __init__(self, name = None,
            # Grid parameters
            nx = 256, ny = None, Lx = 1e6, Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0,                       # Numerical timestep
            step = 0, 
            timeStepper = "RK4",            # Time-stepping method
            nThreads = 1,                   # Number of threads for FFTW
            useFilter = False,
            # 
            # Two-layer parameters:
            ## Physical constants
            f0 = None,
            beta = 2e-11,
            defRadius = 1.5e4, 
            ## Layer-wise parameters
            H1 = 1e3,
            H2 = 1e3,
            U1 = 1e-1,
            U2 = 0.0,
            ## Friction parameters
            bottomDrag = 0.0,
            visc = 1e0,
            viscOrder = 4.0,
        ):

        # Parameters specific to the Physical Problem
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

        # Initialize super-class.
        doublyPeriodicModel.__init__(self, name = name,
            physics = "two layer quasi-geostrophic flow",
            nVars = 2, 
            realVars = True,
            # Persistent doublyPeriodic initialization arguments 
            nx = nx, ny = ny, Lx = Lx, Ly = Ly, t = t, dt = dt, step = step,
            timeStepper = timeStepper, nThreads = nThreads, useFilter = useFilter,
        )
            
        # Set the initial condition to default.
        self.set_physical_soln( \
            1e-1*np.random.standard_normal(self.physSolnShape))
        self.update_state_variables()

        # Initialize default diagnostics
        self.add_diagnostic('CFL', lambda self: self._calc_CFL(),
            description="Maximum CFL number")

        self.add_diagnostic('KE', lambda self: self._calc_KE(),
            description="Total kinetic energy")

        self.add_diagnostic('maxRo', lambda self: self._calc_max_Ro(),
            description="Maximum Rossby number")
        

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

        self.linearCoeff[:, :, 0] = -self._jk*self.U1 \
            - self.visc*(self.k**2.0 + self.l**2.0)**(self.viscOrder/2.0)

        self.linearCoeff[:, :, 1] = -self._jk*self.U2 \
            - self.visc*(self.k**2.0 + self.l**2.0)**(self.viscOrder/2.0)


    def _init_problem_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """

        # Layer depth ratio
        self.delta = self.H1/self.H2

        # Scaled, squared deformation wavenumbers
        self.F1 = self.defRadius**(-2.0) / (1.0 + self.delta)
        self.F2 = self.delta * self.F1

        # Background meridional PV gradients
        self.Q1y = self.beta + self.F1*(self.U1 - self.U2)
        self.Q2y = self.beta - self.F2*(self.U1 - self.U2)

        # A square wavenumber and various products
        self.Ksq = self.k**2.0 + self.l**2.0

        self._jk = 1j*self.k
        self._jl = 1j*self.l

        self._jkQ1y = 1j*self.k*self.Q1y
        self._jkQ2y = 1j*self.k*self.Q2y

        self._bottomDragKsq = self.bottomDrag*self.Ksq

        # "One over" the determinant of the PV-streamfunction inversion matrix
        detM = self.Ksq*(self.Ksq + self.F1 + self.F2)
        detM[0, 0] = float('Inf')
        self._oneOverDetM = 1.0/detM

        # Streamfunctions
        self.psi1h = np.zeros(self.specVarShape, np.dtype('complex128'))
        self.psi2h = np.zeros(self.specVarShape, np.dtype('complex128'))

        # Vorticities and velocities
        self.q1  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.q2  = np.zeros(self.physVarShape, np.dtype('float64'))

        self.u1  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.u2  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v1  = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v2  = np.zeros(self.physVarShape, np.dtype('float64'))

        # Topographic PV
        self.qTop = np.zeros(self.physVarShape, np.dtype('float64'))


    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side """

        q1h = soln[:, :, 0]
        q2h = soln[:, :, 1]

        # Get self.psi1h and self.psi2h
        self._get_streamfunctions(q1h, q2h)

        # Vorticity and velocity in physical space.
        self.q1 = self.ifft2(q1h) 
        self.q2 = self.ifft2(q2h)

        self.u1 = self.ifft2(-self._jl*self.psi1h)
        self.v1 = self.ifft2(self._jk*self.psi1h)

        self.u2 = self.ifft2(-self._jl*self.psi2h)
        self.v2 = self.ifft2(self._jk*self.psi2h)

        # "Premature optimization is the root of all evil"
        #   - Donald Knuth

        # Add topographic contribution to PV
        self.q2 += self.qTop

        # Right Hand Side of the q1-equation
        self.RHS[:, :, 0] = -self._jk*self.fft2( self.u1*self.q1 ) \
                                - self._jl*self.fft2( self.v1*self.q1 ) \
                                - self._jkQ1y*self.psi1h

        # Right Hand Side of the q2-equation
        self.RHS[:, :, 1] = -self._jk*self.fft2( self.u2*self.q2 ) \
                                - self._jl*self.fft2( self.v2*self.q2 ) \
                                - self._jkQ2y*self.psi2h \
                                + self._bottomDragKsq*self.psi2h

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

        self.u1 = self.ifft2(-self._jl*self.psi1h)
        self.v1 = self.ifft2(self._jk*self.psi1h)

        self.u2 = self.ifft2(-self._jl*self.psi2h)
        self.v2 = self.ifft2(self._jk*self.psi2h)


    def _get_streamfunctions(self, q1h, q2h):
        """ Calculate the streamfunctions psi1h and psi2h given the input 
            PV fields q1h and q2h """

        self.psi1h = self._oneOverDetM*( -(self.Ksq + self.F2)*q1h - self.F1*q2h )
        self.psi2h = self._oneOverDetM*( -self.F2*q1h - (self.Ksq + self.F1)*q2h )
    

    def set_topography(self, h):
        """ Set the topographic PV given an input bathymetry """

        # TODO: Add a warning if f0 is None.
        self.qTop = -self.f0*h / self.H2


    def set_q1_and_q2(self, q1, q2):
        """ Update the model state by setting q1 and q2 and calculating 
            state variables """

        self.soln[:, :, 0] = self.fft2(q1)
        self.soln[:, :, 1] = self.fft2(q2)

        self._dealias_soln()
        self.update_state_variables()


    def set_q1(self, q1):
        """ Update the model state by setting q1 and calculating 
            state variables """

        self.soln[:, :, 0] = self.fft2(q1)
        self._dealias_soln()
        self.update_state_variables()


    def set_q2(self, q2):
        """ Update the model state by setting q2 and calculating 
            state variables """

        self.soln[:, :, 1] = self.fft2(q2)
        self._dealias_soln()
        self.update_state_variables()


    def _print_status(self):
        """ Print model status """
        tc = time.time() - self.timer

        # Update model state and calculate diagnostics
        self.update_state_variables() 
        self.evaluate_diagnostics()
            
        print( \
            "step = {:.2e}, clock = {:.2e} s, ".format(self.step, tc) \
            "t = {:.2e} s, KE = {:.2e}, "\
                .format(self.t, self.diagnostics['KE']['value']) \
            "CFL = {:.3f}".format(self.diagnostics['CFL']['value']) \
        )

        self.timer = time.time()


    def describe_model(self):
        """ Describe the current model state """

        print( \
            "\nThis is a doubly-periodic spectral model for \n" + \
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


    # Diagnostic-calculating functions  - - - - - - - - - - - - - - - - - - - -
    def _calc_CFL(self): 
        """ Calculate the maximum CFL number in the model """

        maxSpeed1 = (np.sqrt(self.u1**2.0 + self.v1**2.0)).max()
        maxSpeed2 = (np.sqrt(self.u2**2.0 + self.v2**2.0)).max()

        CFL1 = maxSpeed1 * self.dt * self.nx/self.Lx
        CFL2 = maxSpeed2 * self.dt * self.nx/self.Lx

        return np.array((CFL1, CFL2)).max()


    def _calc_KE(self): 
        """ Calculate the total kinetic energy in the two-layer flow """

        KE1 = (self.Lx*self.Ly)/(self.nx*self.ny) \
            *1.0/2.0*( (self.k**2.0+self.l**2.0)*np.abs(self.psi1h)**2.0 ).sum()

        KE2 = (self.Lx*self.Ly)/(self.nx*self.ny) \
            *1.0/2.0*( (self.k**2.0+self.l**2.0)*np.abs(self.psi2h)**2.0 ).sum()

        return KE1 + KE2


    def _calc_max_Ro(self): 
        """ Calculate the maximum Rossby number in the two-layer flow """

        if self.f0 is None:
            return None

        else:
            maxRo1 = np.abs(self.q1/self.f0).max()
            maxRo2 = np.abs(self.q2/self.f0).max()
            
            return np.array((maxRo1, maxRo2)).max()
