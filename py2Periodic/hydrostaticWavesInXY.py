import numpy as np
from numpy import pi, exp, sqrt, cos, sin
from doublyPeriodic import doublyPeriodicModel
import time

class model(doublyPeriodicModel):
    def __init__(
            self,
            # Parameters general to the doubly-periodic model - - - - - - - - -  
            ## Grid parameters
            nx = 128,
            Lx = 2.0*pi,
            ny = None,
            Ly = None, 
            ## Solver parameters
            t  = 0.0,  
            dt = 1.0e-1,                    # Numerical timestep
            step = 0, 
            timeStepper = "ETDRK4",         # Time-stepping method
            nThreads = 1,                   # Number of threads for FFTW
            ## Printing and saving
            dnSave = 1e2,                   # Interval to save (in timesteps)
            makingPlots = False,
            #
            # Parameters specific to two-dimensional turbulence - - - - - - - - 
            name = "hydrostaticWaveEquationExample", 
            nVars = 2, 
            realVars = False,
            ## Parameters! 
            ### Rotating and gravitating Earth parameters
            f0 = 1.0, 
            sigma = sqrt(5),
            kappa = 8.0, 
            ### Friction: 4th order hyperviscosity
            waveVisc = 1.0e-8,
            meanVisc = 1.0e-8,
            waveViscOrder = 4.0,
            meanViscOrder = 4.0,
        ):

        # Initialize super-class.
        doublyPeriodicModel.__init__(self, 
            physics = "two-dimensional turbulence and the" + \
                            " hydrostatic wave equation",
            # Grid parameters
            nx = nx,
            ny = ny,
            Lx = Lx,
            Ly = Ly,
            nVars = nVars, 
            realVars = realVars,
            # Solver parameters
            t  = t,   
            dt = dt,                        # Numerical timestep
            step = step,                    # Current step
            timeStepper = timeStepper,      # Time-stepping method
            nThreads = nThreads,            # Number of threads for FFTW
            # Simple I/O
            dnSave    = dnSave,             # Interval to save (in timesteps)
        )

        # Physical parameters specific to the Physical Problem
        self.name = name
        self.f0 = f0
        self.sigma = sigma
        self.kappa = kappa
        self.meanVisc = meanVisc
        self.waveVisc = waveVisc
        self.meanViscOrder = meanViscOrder
        self.waveViscOrder = waveViscOrder
            
        # Initial routines
        ## Initialize variables and parameters specific to this problem
        self._init_parameters()
        self._set_linear_coeff()
        self._init_time_stepper()

        # Default initial condition.
        soln = np.zeros_like(self.soln)

        ## Default vorticity initial condition: Gaussian vortex
        rVortex = self.Lx/20
        q0 = 0.1*self.f0 * exp( \
            - ( (self.XX-self.Lx/2.0)**2.0 + (self.YY-self.Ly/2.0)**2.0 ) \
            / (2*rVortex**2.0) \
                       )
        soln[:, :, 0] = q0

        ## Default wave initial condition: plane wave. Find closest
        ## plane wave that satisfies specified dispersion relation.
        kExact = sqrt(self.alpha)*self.kappa
        kApprox = 2.0*pi/self.Lx*np.round(self.Lx*kExact/(2.0*pi)) 

        # Set initial wave velocity to 1
        A00 = -self.alpha*self.f0 / (1j*self.sigma*kApprox)
        A0 = A00*exp(1j*kApprox*self.XX)
        soln[:, :, 1] = A0

        self.set_physical_soln(soln)
        self.update_state_variables()
        
    # Methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def describe_physics(self):
        print("""
            This model solves the hydrostatic wave equation and the \n
            two-dimensional vorticity equation simulataneously. \n
            Arbitrary-order hyperdissipation can be specified for both. \n
            There are two prognostic variables: wave amplitude, and mean vorticity.
        """)

    def _set_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """
        # Two-dimensional turbulent part.
        self.linearCoeff[:, :, 0] = self.meanVisc \
            * (self.KK**2.0 + self.LL**2.0)**(self.meanViscOrder/2.0)

        waveDissipation = self.waveVisc \
            * (self.KK**2.0 + self.LL**2.0)**(self.waveViscOrder/2.0)

        waveDispersion = self.alpha*self.kappa**2.0 - self.KK**2.0 - self.LL**2.0

        self.linearCoeff[:, :, 1] = waveDissipation \
            + self.invE*1j*self.alpha*self.sigma*waveDispersion
       
    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side of the equation """
        # Views for clarity:
        qh = soln[:, :, 0]
        Ah = soln[:, :, 1]

        # Physical-space PV and velocitiy components
        self.q = np.real(self.ifft2(qh))

        # Derivatives of A in physical space
        self.Ax = self.ifft2(self.jKK*Ah)
        self.Ay = self.ifft2(self.jLL*Ah)
        self.Axx = -self.ifft2(self.KK**2.0*Ah)
        self.Ayy = -self.ifft2(self.LL**2.0*Ah)
        self.Axy = -self.ifft2(self.LL*self.KK*Ah)
        self.EA  = -self.ifft2( self.alpha/2.0*Ah*( \
                        self.KK**2.0 + self.LL**2.0 \
                        + (4.0+3.0*self.alpha)*self.kappa**2.0 ))

        # Calculate self.ph
        self.ph = -qh / self.divideSafeKay2

        # Mean velocities
        self.U = np.real(self.ifft2(-self.jLL*self.ph))
        self.V = np.real(self.ifft2( self.jKK*self.ph))

        # Views to clarify calculation of A's RHS
        U = self.U
        V = self.V
        q = self.q
        Ax = self.Ax
        Ay = self.Ay
        EA = self.EA
        Axx = self.Axx
        Ayy = self.Ayy
        Axy = self.Axy
        f0 = self.f0
        sigma = self.sigma
        kappa = self.kappa

        # Right hand side for q
        self.RHS[:, :, 0] = -self.jKK*self.fft2(U*q) \
                                -self.jLL*self.fft2(V*q)

        # Right hand side for A, in steps:
        ## 1. Advection term, 
        self.RHS[:, :, 1] = -self.invE*( \
            self.jKK*self.fft2(U*EA) + self.jLL*self.fft2(V*EA) )

        ## 2. Refraction term
        self.RHS[:, :, 1] += -self.invE/f0*( \
              self.jKK*self.fft2( q * (1j*sigma*Ax - f0*Ay) ) \
            + self.jLL*self.fft2( q * (1j*sigma*Ay + f0*Ax) ) \
                                           )

        ## 3. 'Middling' difference Jacobian term.
        self.RHS[:, :, 1] += self.invE*(2j*sigma/f0**2.0)*( \
              self.jKK*self.fft2(   V*(1j*sigma*Axy - f0*Ayy)   \
                                  - U*(1j*sigma*Ayy + f0*Axy) ) \
            + self.jLL*self.fft2(   U*(1j*sigma*Axy + f0*Axx)   \
                                  - V*(1j*sigma*Axx - f0*Axy) ) \
                                        )

        self._dealias_RHS()
         
    def _init_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """

        # Frequency parameter
        self.alpha = (self.sigma**2.0 - self.f0**2.0) / self.f0**2.0

        # Divide-safe square wavenumber
        self.divideSafeKay2 = self.KK**2.0 + self.LL**2.0
        self.divideSafeKay2[0, 0] = float('Inf')
    
        # Inversion of the operator E
        E = -self.alpha/2.0 * \
                ( self.KK**2.0 + self.LL**2.0 + self.kappa**2.0*(4.0+3.0*self.alpha) )
        self.invE = 1.0 / E

        # Prognostic variables  - - - - - - - - - - - - - - - - - - - - - - - -  
        ## Vorticity and wave-field amplitude
        self.q = np.zeros(self.physVarShape, np.dtype('float64'))
        self.A = np.zeros(self.physVarShape, np.dtype('complex128'))

        # Diagnostic variables  - - - - - - - - - - - - - - - - - - - - - - - -  
        ## Streamfunction transform
        self.ph = np.zeros(self.specVarShape, np.dtype('complex128'))
    
        ## Mean and wave velocity components 
        self.U = np.zeros(self.physVarShape, np.dtype('float64'))
        self.V = np.zeros(self.physVarShape, np.dtype('float64'))
        self.uu = np.zeros(self.physVarShape, np.dtype('float64'))
        self.vv = np.zeros(self.physVarShape, np.dtype('float64'))

        ## Derivatives of wave field amplitude
        self.Ax = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.Ay = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.EA = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.Axx = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.Ayy = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.Axy = np.zeros(self.physVarShape, np.dtype('complex128'))

    # Visible methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def update_state_variables(self):
        """ Update diagnostic variables to current model state """
        # Views for clarity:
        qh = self.soln[:, :, 0]
        Ah = self.soln[:, :, 1]
        
        # Streamfunction
        self.ph = - qh / self.divideSafeKay2 

        # Physical-space PV and velocity components
        self.A = self.ifft2(Ah)
        self.q = np.real(self.ifft2(qh))

        self.U = -np.real(self.ifft2(self.jLL*self.ph))
        self.V =  np.real(self.ifft2(self.jKK*self.ph))

        # Wave velocities
        uh = -1.0/(self.alpha*self.f0)*( \
            1j*self.sigma*self.jKK*Ah - self.f0*self.jLL*Ah )

        vh = -1.0/(self.alpha*self.f0)*( \
            1j*self.sigma*self.jLL*Ah + self.f0*self.jKK*Ah )

        self.uu = np.real( self.ifft2(uh) + np.conj(self.ifft2(uh)) )
        self.vv = np.real( self.ifft2(vh) + np.conj(self.ifft2(vh)) )
        self.sp = sqrt(self.uu**2.0 + self.vv**2.0)

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
        plt.pcolormesh(self.xx, self.yy, sqrt(self.uu**2.0+self.vv**2.0))
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
