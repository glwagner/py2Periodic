import doublyPeriodic
import numpy as np; from numpy import pi 
import time

class model(doublyPeriodic.model):
    def __init__(self, name = "hydrostaticWaveEquationExample", 
            # Grid parameters
            nx = 256, ny = None, Lx = 1e6, Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0e-1,                    # Numerical timestep
            step = 0, 
            timeStepper = "ETDRK4",         # Time-stepping method
            nThreads = 1,                   # Number of threads for FFTW
            useFilter = False,
            #
            # Hydrostatic Wave Eqn params: rotating and gravitating Earth 
            f0 = 1.0, 
            sigma = np.sqrt(5),
            kappa = 8.0, 
            # Friction: 4th order hyperviscosity
            waveVisc = 1.0e-12,
            meanVisc = 1.0e-8,
            waveViscOrder = 4.0,
            meanViscOrder = 4.0,
        ):

        # Initialize super-class.
        doublyPeriodic.model.__init__(self, 
            physics = "two-dimensional turbulence and the" + \
                            " hydrostatic wave equation",
            nVars = 2, 
            realVars = False,
            # Persistent doublyPeriodic initialization arguments 
            nx = nx, ny = ny, Lx = Lx, Ly = Ly, t = t, dt = dt, step = step,
            timeStepper = timeStepper, nThreads = nThreads, useFilter = useFilter,
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
        rVortex = self.Lx/10.0
        (x0, y0) = (self.Lx/2.0, self.Ly/2.0)
        q0 = -0.05*self.f0 * np.exp( -( (self.x-x0)**2.0 + (self.y-y0)**2.0 ) \
            / (2*rVortex**2.0) \
        )
        soln[:, :, 0] = q0

        ## Default wave initial condition: plane wave. Find closest
        ## plane wave that satisfies specified dispersion relation.
        kExact = np.sqrt(self.alpha)*self.kappa
        kApprox = 2.0*pi/self.Lx*np.round(self.Lx*kExact/(2.0*pi)) 

        # Set initial wave velocity to 1
        A00 = -self.alpha*self.f0 / (1j*self.sigma*kApprox)
        A0 = A00*np.exp(1j*kApprox*self.x)
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
        self.linearCoeff[:, :, 0] = -self.meanVisc \
            * (self.k**2.0 + self.l**2.0)**(self.meanViscOrder/2.0)

        waveDispersion = self.k**2.0 + self.l**2.0 - self.alpha*self.kappa**2.0
        waveDissipation = -self.waveVisc \
            * (self.k**2.0 + self.l**2.0)**(self.waveViscOrder/2.0)

        self.linearCoeff[:, :, 1] = waveDissipation \
            + self._invE*1j*self.alpha*self.sigma*waveDispersion
       
    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side of PDE """

        qh = soln[:, :, 0]
        Ah = soln[:, :, 1]

        self.q = np.real(self.ifft2(qh))

        # Derivatives of A in physical space
        self.Ax  =  self.ifft2(self._jk*Ah)
        self.Ay  =  self.ifft2(self._jl*Ah)
        self.Axx = -self.ifft2(self.k**2.0*Ah)
        self.Ayy = -self.ifft2(self.l**2.0*Ah)
        self.Axy = -self.ifft2(self.l*self.k*Ah)
        self.EA  = -self.ifft2( self.alpha/2.0*Ah*( \
                        self.k**2.0 + self.l**2.0 \
                        + (4.0+3.0*self.alpha)*self.kappa**2.0 ))

        # Calculate streamfunction
        self.psih = -qh / self._divSafeKsq

        # Mean velocities
        self.U = np.real(self.ifft2(-self._jl*self.psih))
        self.V = np.real(self.ifft2( self._jk*self.psih))

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
        self.RHS[:, :, 0] = -self._jk*self.fft2(U*q) \
                                -self._jl*self.fft2(V*q)

        # Right hand side for A, in steps:
        ## 1. Advection term, 
        self.RHS[:, :, 1] = -self._invE*( \
            self._jk*self.fft2(U*EA) + self._jl*self.fft2(V*EA) )

        ## 2. Refraction term
        self.RHS[:, :, 1] += -self._invE/f0*( \
              self._jk*self.fft2( q * (1j*sigma*Ax - f0*Ay) ) \
            + self._jl*self.fft2( q * (1j*sigma*Ay + f0*Ax) ) \
        )

        ## 3. 'Middling' difference Jacobian term.
        self.RHS[:, :, 1] += self._invE*(2j*sigma/f0**2.0)*( \
              self._jk*self.fft2(   V*(1j*sigma*Axy - f0*Ayy)   \
                                  - U*(1j*sigma*Ayy + f0*Axy) ) \
            + self._jl*self.fft2(   U*(1j*sigma*Axy + f0*Axx)   \
                                  - V*(1j*sigma*Axx - f0*Axy) ) \
        )

        self._dealias_RHS()
         
    def _init_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """

        # Frequency parameter
        self.alpha = (self.sigma**2.0 - self.f0**2.0) / self.f0**2.0

        # Wavenumbers and products
        self._jk = 1j*self.k
        self._jl = 1j*self.l

        self._divSafeKsq = self.k**2.0 + self.l**2.0
        self._divSafeKsq[0, 0] = float('Inf')
    
        # Inversion of the operator E
        E = -self.alpha/2.0 * \
                ( self.k**2.0 + self.l**2.0 + self.kappa**2.0*(4.0+3.0*self.alpha) )
        self._invE = 1.0 / E

        # Vorticity and wave-field amplitude
        self.q = np.zeros(self.physVarShape, np.dtype('float64'))
        self.A = np.zeros(self.physVarShape, np.dtype('complex128'))

        # Streamfunction transform
        self.psih = np.zeros(self.specVarShape, np.dtype('complex128'))
    
        # Mean and wave velocity components 
        self.U = np.zeros(self.physVarShape, np.dtype('float64'))
        self.V = np.zeros(self.physVarShape, np.dtype('float64'))
        self.u = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v = np.zeros(self.physVarShape, np.dtype('float64'))

        # Derivatives of wave field amplitude
        self.Ax = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.Ay = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.EA = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.Axx = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.Ayy = np.zeros(self.physVarShape, np.dtype('complex128'))
        self.Axy = np.zeros(self.physVarShape, np.dtype('complex128'))

    def update_state_variables(self):
        """ Update diagnostic variables to current model state """

        qh = self.soln[:, :, 0]
        Ah = self.soln[:, :, 1]
        
        # Streamfunction
        self.psih = -qh / self._divSafeKsq 

        # Physical-space PV and velocity components
        self.A = self.ifft2(Ah)
        self.q = np.real(self.ifft2(qh))

        self.U = -np.real(self.ifft2(self._jl*self.psih))
        self.V =  np.real(self.ifft2(self._jk*self.psih))

        # Wave velocities
        uh = -1.0/(self.alpha*self.f0)*( \
            1j*self.sigma*self._jk*Ah - self.f0*self._jl*Ah )

        vh = -1.0/(self.alpha*self.f0)*( \
            1j*self.sigma*self._jl*Ah + self.f0*self._jk*Ah )

        self.u = np.real( self.ifft2(uh) + np.conj(self.ifft2(uh)) )
        self.v = np.real( self.ifft2(vh) + np.conj(self.ifft2(vh)) )

    def set_q(self, q):
        """ Set model vorticity """

        self.soln[:, :, 0] = self.fft2(q)
        self._dealias_soln()
        self.update_state_variables()

    def set_A(self, A):
        """ Set model wave-field amplitude """

        self.soln[:, :, 1] = self.fft2(A)
        self._dealias_soln()
        self.update_state_variables()

    def describe_model(self):
        """ Describe the current model state """

        print("\nThis is a doubly-periodic spectral model for \n" + \
                "{:s} \n".format(self.physics) + \
                "with the following attributes:\n\n" + \
                " Domain             : {:.2e} X {:.2e} m\n".format( \
                    self.Lx, self.Ly) + \
                " Grid               : {:d} X {:d}\n".format(self.nx, self.ny) + \
                " Wave hypervisc     : {:.2e} m^{:d}/s\n".format( \
                    self.waveVisc, int(self.waveViscOrder)) + \
                " Mean hypervisc     : {:.2e} m^{:d}/s\n".format( \
                    self.meanVisc, int(self.meanViscOrder)) + \
                " Frequency param    : {:.2f}\n".format(self.alpha) + \
                " Comp. threads      : {:d} \n".format(self.nThreads) \
        )
