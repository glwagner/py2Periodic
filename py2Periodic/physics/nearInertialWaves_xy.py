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
            # Near-inertial equation params: rotating and gravitating Earth 
            f0 = 1.0, 
            kappa = 64.0, 
            # Friction: 4th order hyperviscosity
            waveVisc = 1.0e-4,
            waveViscOrder = 2.0,
            meanVisc = 1.0e-4,
            meanViscOrder = 2.0,
        ):

        # Physical parameters specific to the Physical Problem
        self.f0 = f0
        self.kappa = kappa
        self.meanVisc = meanVisc
        self.meanViscOrder = meanViscOrder
        self.waveVisc = waveVisc
        self.waveViscOrder = waveViscOrder

        # Initialize super-class.
        doublyPeriodicModel.__init__(self, name = name,
            physics = "two-dimensional turbulence and the" + \
                            " near-inertial wave equation",
            nVars = 2, 
            realVars = False,
            # Persistent doublyPeriodic initialization arguments 
            nx = nx, ny = ny, Lx = Lx, Ly = Ly, t = t, dt = dt, step = step,
            timeStepper = timeStepper, nThreads = nThreads, useFilter = useFilter,
        )
            
        ## Default vorticity initial condition: Gaussian vortex
        rVortex = self.Lx/20
        q0 = 0.1*self.f0 * np.exp( \
            - ( (self.XX-self.Lx/2.0)**2.0 + (self.YY-self.Ly/2.0)**2.0 ) \
            / (2*rVortex**2.0) \
                       )
        self.set_q(q0)

        # Default wave initial condition: uniform velocity.
        A0 = np.ones(self.physVarShape)
        self.set_A(A0)
        
    # Methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def describe_physics(self):
        print("""
            This model solves the linearized near-inertial wave equation, also \n
            known as the YBJ equation, and the \n
            two-dimensional vorticity equation simulataneously. \n
            Arbitrary-order hyperdissipation can be specified for both. \n
            There are two prognostic variables: wave amplitude, and mean vorticity.
        """)

    def _set_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """
        # Two-dimensional turbulent part.
        self.linearCoeff[:, :, 0] = -self.meanVisc \
            * (self.KK**2.0 + self.LL**2.0)**(self.meanViscOrder/2.0)

        waveDissipation = -self.waveVisc \
            * (self.KK**2.0 + self.LL**2.0)**(self.waveViscOrder/2.0)

        waveDispersion = -1j*self.f0/(2.0*self.kappa**2.0) \
                            * ( self.KK**2.0 + self.LL**2.0)

        self.linearCoeff[:, :, 1] = waveDissipation + waveDispersion
       
    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side of PDE """
        # Views for clarity:
        qh = soln[:, :, 0]
        Ah = soln[:, :, 1]

        # Physical-space things
        self.q = np.real(self.ifft2(qh))
        self.A = self.ifft2(Ah)

        # Calculate streamfunction
        self.psih = -qh / self.divideSafeKay2

        # Mean velocities
        self.U = -np.real(self.ifft2(self.jLL*self.psih))
        self.V =  np.real(self.ifft2(self.jKK*self.psih))

        # Views to clarify calculation of A's RHS
        U = self.U
        V = self.V
        q = self.q
        A = self.A
        
        # Right hand side for q
        self.RHS[:, :, 0] = -self.jKK*self.fft2(U*q) - self.jLL*self.fft2(V*q) 

        # Right hand side for A, in steps:
        ## 1. Advection term, 
        self.RHS[:, :, 1] = -self.jKK*self.fft2(U*A) - self.jLL*self.fft2(V*A) \
                                -1j/2.0*self.fft2(q*A)
        self._dealias_RHS()
         
    def _init_problem_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """
        # Divide-safe square wavenumber
        self.divideSafeKay2 = self.KK**2.0 + self.LL**2.0
        self.divideSafeKay2[0, 0] = float('Inf')
    
        # Vorticity and wave-field amplitude
        self.q = np.zeros(self.physVarShape, np.dtype('float64'))
        self.A = np.zeros(self.physVarShape, np.dtype('complex128'))

        # Streamfunction transform
        self.psih = np.zeros(self.specVarShape, np.dtype('complex128'))
    
        # Mean and wave velocity components 
        self.U = np.zeros(self.physVarShape, np.dtype('float64'))
        self.V = np.zeros(self.physVarShape, np.dtype('float64'))
        
    def update_state_variables(self):
        """ Update diagnostic variables to current model state """
        # Views for clarity:
        qh = self.soln[:, :, 0]
        Ah = self.soln[:, :, 1]
        
        # Streamfunction
        self.psih = - qh / self.divideSafeKay2 

        # Physical-space PV and velocity components
        self.q = np.real(self.ifft2(qh))
        self.A = self.ifft2(Ah)

        self.U = -np.real(self.ifft2(self.jLL*self.psih))
        self.V =  np.real(self.ifft2(self.jKK*self.psih))

    def set_q(self, q):
        """ Set model vorticity """
        self.soln[:, :, 0] = self.fft2(q)
        self.soln = self._dealias_array(self.soln)
        self.update_state_variables()

    def set_A(self, A):
        """ Set model wave field amplitude"""
        self.soln[:, :, 1] = self.fft2(A)
        self.soln = self._dealias_array(self.soln)
        self.update_state_variables()

    def plot_current_state(self):
        """ Create a simple plot that shows the state of the model."""

        # Figure out how to do this efficiently.
        import matplotlib.pyplot as plt

        self.update_state_variables()

        # Initialize colorbar dictionary
        colorbarProperties = { 
            'orientation' : 'vertical',
            'shrink'      : 0.8,
            'extend'      : 'neither',
        }

        self.fig = plt.figure('Hydrostatic wave equation',
                            figsize=(8, 4))

        ax1 = plt.subplot(121)
        plt.pcolormesh(self.xx, self.yy, self.q, cmap='RdBu_r')
        plt.axis('square')

        ax2 = plt.subplot(122)
        plt.pcolormesh(self.xx, self.yy, np.sqrt(self.uu**2.0+self.vv**2.0))
        plt.axis('square')

    def describe_model(self):
        """ Describe the current model state """

        print("\nThis is a doubly-periodic spectral model for \n" + \
                "{:s} \n".format(self.physics) + \
                "with the following attributes:\n\n" + \
                "   Domain       : {:.2e} X {:.2e} m\n".format(self.Lx, self.Ly) + \
                "   Resolution   : {:d} X {:d}\n".format(self.nx, self.ny) + \
                "   Timestep     : {:.2e} s\n".format(self.dt) + \
                "   Current time : {:.2e} s\n\n".format(self.t) + \
                "The FFT scheme uses {:d} thread(s).\n".format(self.nThreads))
