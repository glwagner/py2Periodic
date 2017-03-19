from __future__ import division
import numpy as np
from numpy import pi, exp, sqrt, cos, sin
from doublyPeriodic import doublyPeriodicModel
import time

class hydrostaticWaveModel(doublyPeriodicModel):

    """ This class describes a 2D, doubly-periodic turbulence model 
        in velocity-vorticity formulation. """

    def __init__(
            self,
            # Parameters general to the doubly-periodic model - - - - - - - - -  
            ## Grid parameters
            nx = 128,
            Lx = 2.0*pi,
            ny = None,
            Ly = None, 
            ## Timestepping parameters
            t  = 0.0,  
            dt = 1.0e1,                    # Numerical timestep
            ## Computational parameters
            nThreads = 1,                   # Number of threads for FFTW
            dealias = True, 
            ## Printing and saving
            dnSave = 1e2,                   # Interval to save (in timesteps)
            ## Plotting
            makingPlots = False,
            # Parameters specific to two-dimensional turbulence - - - - - - - - 
            name = "hydrostaticWaveEquationExample", 
            physics = "two-dimensional turbulence and the" + \
                            " hydrostatic wave equation",
            nVars = 2, 
            realVars = False,
            ## Parameters! 
            ### Rotating and gravitating Earth parameters
            f0 = 1.0, 
            sigma = 3.0,
            kappa = 8.0, 
            ### Friction: 4th order hyperviscosity
            waveVisc = 1.0e-8,
            meanVisc = 1.0e-16,
            waveViscOrder = 4.0,
            meanViscOrder = 4.0,
        ):

        # The default domain is square
        if ny is None: ny = nx
        if Ly is None: Ly = Lx

        # Initialize super-class.
        doublyPeriodicModel.__init__(self, 
            name = name,
            # Grid parameters
            nx = nx,
            ny = ny,
            Lx = Lx,
            Ly = Ly,
            nVars = nVars, 
            realVars = realVars,
            # Timestepping parameters
            t  = t,   
            dt = dt,                        # Numerical timestep
            step = 0,                       # Current step
            # Computational parameters
            nThreads = nThreads,            # Number of threads for FFTW
            dealias  = dealias,
            # Simple I/O
            dnSave    = dnSave,             # Interval to save (in timesteps)
            # Plotting
            makingPlots = makingPlots,
        )

        # Move these around.
        self.nSave = 0
      
        # Physical parameters specific to the Physical Problem
        self.physics = physics
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
        self._init_linear_coeff()
        self._init_time_stepper()

        # Default initial condition.
        soln = np.zeros_like(self.soln)

        ## Default vorticity initial condition: Gaussian vortex
        rVortex = self.Lx/20
        q0 = 0.05*self.f0 * exp( \
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
        
    # Hidden methods  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _init_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """

        self.linearCoeff = np.zeros(self.physicalShape, np.dtype('complex128'))

        # Two-dimensional turbulence
        self.linearCoeff[:, :, 0] = self.meanVisc \
            * (self.KK**2.0 + self.LL**2.0)**(self.meanViscOrder/2.0)

        # Hyperviscous operator for waves
        waveDissipation = self.waveVisc \
            * (self.KK**2.0 + self.LL**2.0)**(self.waveViscOrder/2.0)

        # Dispersion operator
        waveDispersion = self.alpha*self.kappa**2.0 - self.KK**2.0 - self.LL**2.0

        #self.linearCoeff[:, :, 1] = waveDissipation \
        #    - self.invE*1j*self.alpha*self.sigma*waveDispersion

        self.linearCoeff[:, :, 1] = self.invE*1j*self.alpha*self.sigma*waveDispersion
            
       
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
        self.ph = -qh / self.kay2

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

        #self.RHS[:, :, 1] = np.zeros_like(self.RHS[:, :, 0])
        self._dealias_imag_RHS(self.RHS)
         
    def _init_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """

        # Frequency parameter
        self.alpha = (self.sigma**2.0 - self.f0**2.0) / self.f0**2.0

        # Divide-safe square wavenumber
        self.kay2 = self.KK**2.0 + self.LL**2.0
        self.kay2[0, 0] = float('Inf')
    
        # Inversion of the operator E
        E = -self.alpha/2.0 * \
                ( self.KK**2.0 + self.LL**2.0 + self.kappa**2.0*(4.0+3.0*self.alpha) )
        self.invE = 1.0 / E

        # Prognostic variables  - - - - - - - - - - - - - - - - - - - - - - - -  
        ## Solution
        self.soln = np.zeros(self.physicalShape, np.dtype('complex128'))
        
        ## Vorticity and wave-field amplitude
        self.q = np.zeros((self.ny, self.nx), np.dtype('float64'))
        self.A = np.zeros((self.ny, self.nx), np.dtype('complex128'))

        # Diagnostic variables  - - - - - - - - - - - - - - - - - - - - - - - -  
        ## Streamfunction transform
        self.ph = np.zeros((self.ny, self.nx), np.dtype('complex128'))
    
        ## Mean and wave velocity components 
        self.U = np.zeros((self.ny, self.nx), np.dtype('float64'))
        self.V = np.zeros((self.ny, self.nx), np.dtype('float64'))
        self.uu = np.zeros((self.ny, self.nx), np.dtype('float64'))
        self.vv = np.zeros((self.ny, self.nx), np.dtype('float64'))

        ## Derivatives of wave field amplitude
        self.Ax = np.zeros((self.ny, self.nx), np.dtype('complex128'))
        self.Ay = np.zeros((self.ny, self.nx), np.dtype('complex128'))
        self.EA = np.zeros((self.ny, self.nx), np.dtype('complex128'))
        self.Axx = np.zeros((self.ny, self.nx), np.dtype('complex128'))
        self.Ayy = np.zeros((self.ny, self.nx), np.dtype('complex128'))
        self.Axy = np.zeros((self.ny, self.nx), np.dtype('complex128'))

    # Visible methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def update_state_variables(self):
        """ Update diagnostic variables to current model state """

        # Views for clarity:
        qh = self.soln[:, :, 0]
        Ah = self.soln[:, :, 1]
        
        # Streamfunction
        self.ph = - qh / self.kay2 

        # Physical-space PV and velocity components
        self.A = self.ifft2(Ah)
        self.q = np.real(self.ifft2(qh))

        self.u = -np.real(self.ifft2(self.jLL*self.ph))
        self.v =  np.real(self.ifft2(self.jKK*self.ph))

        # Wave velocities
        uh = -1.0/(self.alpha*self.f0)*( \
            1j*self.sigma*self.jKK*Ah - self.f0*self.jLL*Ah )

        vh = -1.0/(self.alpha*self.f0)*( \
            1j*self.sigma*self.jLL*Ah + self.f0*self.jKK*Ah )

        self.uu = np.real( self.ifft2(uh) + np.conj(self.ifft2(uh)) )
        self.vv = np.real( self.ifft2(vh) + np.conj(self.ifft2(vh)) )

    def set_physical_soln(self, soln):
        """ Initialize model with a physical space solution """ 
        q = soln[:, :, 0]
        A = soln[:, :, 1]

        self.soln[:, :, 0] = self.fft2(q)
        self.soln[:, :, 1] = self.fft2(A)
        
        #self.soln = self._dealias_imag(self.soln)

    def set_spectral_soln(self, soln):
        """ Initialize model with a spectral space solution """ 
        self.soln = soln
        self.soln = self._dealias_imag(self.soln)

    def run_nSteps(self, nSteps=1e2, dnLog=float('Inf')):
        """ Step forward nStep times """

        # Initialize run
        step0 = self.step
        self._start_timer()

        # Step forward
        while (self.step < step0+nSteps):
            
            self._step_forward()

            if (self.step % dnLog == 0.0):
                self._print_status()

            self.t += self.dt
            self.step += 1

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

    # Time steppers for the doublyPeriodicModel class - - - - - - - - - - - - - 
    ## Forward Euler
    def _step_forward_forward_euler(self):
        """ March system forward in time using forward Euler scheme """
    
        self._calc_right_hand_side(self.soln, self.t)
        self.soln += self.dt*(self.RHS - self.linearCoeff*self.soln)

    def _init_time_stepper_forward_euler(self):
        """ Initialize and allocate vars for forward Euler time-marching """

        if self.realVars:
            self.RHS = np.zeros(self.realSpectralShape, np.dtype('complex128'))
        else:
            self.RHS = np.zeros(self.physicalShape, np.dtype('complex128'))


    ## RKW3
    def _init_time_stepper_RKW3(self):
        """ Initialize and allocate vars for RK3W time-marching """

        self.a1, self.a2, self.a3 = 29./96., -3./40., 1./6.
        self.b1, self.b2, self.b3 = 37./160., 5./24., 1./6.
        self.c1, self.c2, self.c3 = 8./15., 5./12., 3./4.
        self.d1, self.d2 = -17./60., -5./12.

        self.L0 = -self.dt*self.linearCoeff
        self.L1 = ( (1. + self.a1*self.L0)/(1. - self.b1*self.L0) )     
        self.L2 = ( (1. + self.a2*self.L0)/(1. - self.b2*self.L0) )
        self.L3 = ( (1. + self.a2*self.L0)/(1. - self.b3*self.L0) )

        # Allocate nonlinear terms
        if self.realVars:
            self.RHS = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL1 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL2 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
        else:
            self.RHS = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL1 = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL2 = np.zeros(self.physicalShape, np.dtype('complex128'))

    def _step_forward_RKW3(self):
        """ March the system forward in time using a RK3W-theta scheme """

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

    ## RK4
    def _step_forward_RK4(self):
        """ March the system forward using a ETDRK4 scheme """

        self._calc_right_hand_side(self.soln, self.t)
        self.NL1 = self.RHS.copy() - self.linearCoeff*self.soln

        t1 = self.t + self.dt/2
        self.soln1 = self.soln + self.dt/2.0*self.NL1

        self._calc_right_hand_side(self.soln1, t1) 
        self.NL2 = self.RHS.copy() - self.linearCoeff*self.soln1

        self.soln1 = self.soln + self.dt/2.0*self.NL2
        self._calc_right_hand_side(self.soln1, t1) 
        self.NL3 = self.RHS.copy() - self.linearCoeff*self.soln1

        t1 = self.t + self.dt
        self.soln1 = self.soln + self.dt*self.NL3
        self._calc_right_hand_side(self.soln1, t1) 
        self.NL4 = self.RHS.copy() - self.linearCoeff*self.soln1

        self.soln += self.dt*(   1/6*self.NL1 + 1/3*self.NL2 \
                               + 1/3*self.NL3 + 1/6*self.NL4 )

    def _init_time_stepper_RK4(self):
        """ Initialize and allocate vars for RK4 time-marching """

        if self.realVars:
            # Allocate intermediate solution variable
            self.soln1 = np.zeros(self.realSpectralShape, np.dtype('complex128'))

            # Allocate nonlinear terms
            self.RHS = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL1 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL2 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL3 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL4 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
        else:
            # Allocate intermediate solution variable
            self.soln1 = np.zeros(self.physicalShape, np.dtype('complex128'))

            # Allocate nonlinear terms
            self.RHS = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL1 = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL2 = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL3 = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL4 = np.zeros(self.physicalShape, np.dtype('complex128'))

    ## ETDRK4
    def _step_forward_ETDRK4(self):
        """ March the system forward using an ETDRK4 scheme """

        self._calc_right_hand_side(self.soln, self.t)
        self.NL1 = self.RHS.copy()

        t1 = self.t + self.dt/2
        self.soln1 = self.linearExp*self.soln + self.zeta*self.NL1
        self._calc_right_hand_side(self.soln1, t1)
        self.NL2 = self.RHS.copy()

        self.soln2 = self.linearExp*self.soln + self.zeta*self.NL2
        self._calc_right_hand_side(self.soln2, t1)
        self.NL3 = self.RHS.copy()

        t1 = self.t + self.dt
        self.soln1 = self.linearExp*self.soln1 + self.zeta*(2*self.NL3-self.NL1)
        self._calc_right_hand_side(self.soln1, t1)
        self.NL4 = self.RHS.copy()

        # The final step
        self.soln = self.linearExp*self.soln \
                    +   self.alph * self.NL1 \
                    + 2*self.beta * (self.NL2 + self.NL3) \
                    +   self.gamm * self.NL4

    def _init_time_stepper_ETDRK4(self):
        """ Initialize and allocate vars for RK4 time-marching """

        linearCoeffDt = self.dt*self.linearCoeff
        
        # Calculate coefficients with circular line integral in complex plane
        nCirc = 32          
        rCirc = 1.0       
        circ = rCirc*exp(2j*pi*(np.arange(1, nCirc+1)-1/2)/nCirc) 

        # Circular contour around the point to be calculated
        zc = -linearCoeffDt[..., np.newaxis] \
                + circ[np.newaxis, np.newaxis, np.newaxis, ...]

        # Four coefficients, zeta, alpha, beta, and gamma
        self.zeta = self.dt*( \
                        (exp(zc/2.0) - 1.0) / zc \
                            ).mean(axis=-1)

        self.alph = self.dt*( \
                      (-4.0 - zc + exp(zc)*(4.0-3.0*zc+zc**2.0)) / zc**3.0 \
                            ).mean(axis=-1)

        self.beta = self.dt*( \
                      (2.0 + zc + exp(zc)*(-2.0+zc) ) / zc**3.0 \
                            ).mean(axis=-1)

        self.gamm = self.dt*( \
                      (-4.0 - 3.0*zc - zc**2.0 + exp(zc)*(4.0-zc)) / zc**3.0 \
                            ).mean(axis=-1)
                              
        # Pre-calculate an exponential     
        self.linearExp = exp(-self.dt*self.linearCoeff/2)

        if self.realVars:
            # Allocate intermediate solution variable
            self.soln1 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.soln2 = np.zeros(self.realSpectralShape, np.dtype('complex128'))

            # Allocate nonlinear terms
            self.RHS = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL1 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL2 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL3 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
            self.NL4 = np.zeros(self.realSpectralShape, np.dtype('complex128'))
        else:
            # Allocate intermediate solution variable
            self.soln1 = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.soln2 = np.zeros(self.physicalShape, np.dtype('complex128'))

            # Allocate nonlinear terms
            self.RHS = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL1 = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL2 = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL3 = np.zeros(self.physicalShape, np.dtype('complex128'))
            self.NL4 = np.zeros(self.physicalShape, np.dtype('complex128'))

    #_init_time_stepper = _init_time_stepper_forward_euler
    #_step_forward = _step_forward_forward_euler
    #_init_time_stepper = _init_time_stepper_RKW3
    #_step_forward = _step_forward_RKW3
    #_init_time_stepper = _init_time_stepper_RK4
    #_step_forward = _step_forward_RK4
    _init_time_stepper = _init_time_stepper_ETDRK4
    _step_forward = _step_forward_ETDRK4

