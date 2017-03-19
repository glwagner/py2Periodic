from __future__ import division
import numpy as np
from numpy import pi, exp, sqrt, cos, sin
from spectralModels import doublyPeriodicModel
#from .timeSteppers import *
import time

class model(doublyPeriodicModel):

    """ This class describes a 2D, doubly-periodic turbulence model 
        in velocity-vorticity formulation. """

    def __init__(
            self,
            # Parameters general to the doubly-periodic model - - - - - - - - -  
            ## Grid parameters
            nx = 256,
            Lx = 2.0*pi, 
            ny = None,
            Ly = None, 
            ## Timestepping parameters
            t  = 0.0,  
            dt = 1.0e-1,                    # Numerical timestep
            ## Computational parameters
            nThreads = 1,                   # Number of threads for FFTW
            dealias = True, 
            ## Printing and saving
            dnSave = 1e2,                   # Interval to save (in timesteps)
            # Parameters specific to two-dimensional turbulence - - - - - - - - 
            name = "generic2DTurbulenceModel", 
            physics = "two-dimensional turbulence", 
            nVars = 1, 
            realVars = True,
            ## Laplacian viscosity
            nu = 1.0e-4,
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
        )

        # Move these around.
        self.nSave = 0
      
        # Physical parameters specific to the Physical Problem
        self.nu = nu
        self.physics = physics

        # Initial routines
        
        ## Initialize variables and parameters specific to this problem
        self._init_parameters()
        self._init_linear_coeff()
        self._init_time_stepper()

        ## Initialize the solution
        self.set_physical_sol(np.random.standard_normal(self.physicalShape))

    # Hidden methods  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _init_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """

        self.linearCoeff = np.zeros(self.realSpectralShape, np.dtype('complex128'))

        self.linearCoeff[:, :, 0] = self.nu*(self.rKK**2.0 + self.rLL**2.0)

    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side of the equation """

        # For clarity:
        qh = soln[:, :, 0]

        # Get streamfunction
        self.ph = -qh / self.kay2

        # Physical-space PV and velocitiy components
        self.q = self.irfft2(qh)
        self.u = self.irfft2(-self.rjLL*self.ph) 
        self.v = self.irfft2( self.rjKK*self.ph)

        # Advection and diffusion
        self.RHS[:, :, 0] = -self.rjKK*self.rfft2(self.u*self.q) \
                                -self.rjLL*self.rfft2(self.v*self.q)

        self._dealias_real_RHS(self.RHS)

    def _update_diagnostic_variables(self):
        """ Update diagnostic variables to current model state """

        # For convenience:
        qh = self.soln[:, :, 0]

         # Get streamfunction
        self.ph = -qh / self.kay2

        # Physical-space PV and velocitiy components
        self.q = self.irfft2(qh)
        self.u = self.irfft2(-self.rjLL*self.ph) 
        self.v = self.irfft2( self.rjKK*self.ph)
             
    def _init_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """

        # Divide-safe square wavenumber
        self.kay2 = self.rKK**2 + self.rLL**2
        self.kay2[0, 0] = float('Inf')
            
        # Prognostic variables  - - - - - - - - - - - - - - - - - - - - - - - -  
        ## Vorticity (real)
        self.q  = np.zeros(self.physicalShape, np.dtype('float64'))

        # Diagnostic variables  - - - - - - - - - - - - - - - - - - - - - - - -  
        ## Streamfunction (real)
        self.p  = np.zeros(self.physicalShape, np.dtype('float64'))
        self.ph = np.zeros(self.realSpectralShape, np.dtype('complex128'))
    
        ## Velocity components (real)
        self.u  = np.zeros(self.physicalShape, np.dtype('float64'))
        self.v  = np.zeros(self.physicalShape, np.dtype('float64'))
    
    def _take_snapshot(self):
        """ Save a snapshot """
        pass

    def _save_to_disk(self):
        """ Save to disk """
        self.nSave += 1
    
    def _create_outfile(self):
        """ Create the file to store output """
        pass

    # Visible methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def run_nSteps(self, nSteps=1e2, dnLog=1e2):
        """ Step forward nStep times """

        # Initialize run
        step0 = self.step
        self._start_timer()

        # Step forward
        while (self.step <= step0+nSteps):
            
            self._step_forward()
            self._update_diagnostic_variables()

            if (self.step % dnLog == 0.0):
                self._print_status()

            self.t += self.dt
            self.step += 1

    def run_for_time(self, nSteps=1e2):
        """ Step forward nStep time-step """

        # Initialize run
        step0 = self.step
        self._start_timer()

        # Step forward
        while (self.step <= step0+nSteps):
            
            self._step_forward()

            self.t += self.dt
            self.step += 1

        tc = time.time() - self.timer

        print("Elapsed time = {:3f}".format(tc))
        
        return tc

    def run_and_take_snapshots(self, nSteps=1e2, dnSnap=1e1, dnLog=float('Inf')):
        """ Step forward nStep time-step """

        # Initialize run
        step0 = self.step
        self._start_timer()

        # Step forward
        while (self.step <= step0+nSteps):
            
            self._step_forward()
            self._update_diagnostic_variables()

            if (self.step % self.dnLog == 0.0):
                self._print_status()

            if (self.step % self.dnSnap == 0.0):
                self._print_snap_status()
                self._take_snapshot()

            self.t += self.dt
            self.step += 1

    def run_and_save(self, nSteps=1e2, dnSave=1e1, dnLog=float('Inf')):
        """ Step forward nStep time-step """

        # Initialize outfile
        self._create_outfile()

        # Initialize run
        self.step = 0
        self._start_timer()

        # Step forward
        while (self.step <= nSteps):
            
            self._step_forward()

            if (self.step % self.dnLog == 0.0):
                self._print_status()

            if (self.step % self.dnSave == 0.0):
                self._print_save_status()
                self._save_snapshot()

            self.t += self.dt
            self.step += 1

    def describe_model(self):
        """ Describe the current model state """

        print("\nThis is a doubly-periodic spectral model for " + \
                "{:s} ".format(self.physics) + \
                "with the following attributes:\n\n" + \
                "   Domain       : {:.2e} X {:.2e} m\n".format(self.Lx, self.Ly) + \
                "   Resolution   : {:d} X {:d}\n".format(self.nx, self.ny) + \
                "   Timestep     : {:.2e} s\n".format(self.dt) + \
                "   Current time : {:.2e} s\n\n".format(self.t) + \
                "The FFT scheme uses {:d} thread(s).\n".format(self.nThreads))

    def set_physical_sol(self, soln):
        """ Initialize vorticity """ 
        self.q  = soln[:, :, 0]
        self.soln[:, :, 0] = self.rfft2(self.q)

    def set_spectral_sol(self, qh):
        """ Initialize vorticity """ 
        self.soln[:, :, 0] = qh
        self.q  = np.real(self.irfft2(qh))

    # Time steppers for the doublyPeriodicModel class - - - - - - - - - - - - - 
    ## Forward Euler
    def _step_forward_forward_euler(self):
        """ March system forward in time using forward Euler scheme """
    
        self._calc_right_hand_side(self.soln, self.t)
        self.soln += self.dt*((self.RHS).copy() - self.linearCoeff*self.soln)

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

        self.NL1 = self._calc_right_hand_side(self.soln, self.t)
        self.soln  = (self.L1*self.soln + self.c1*self.dt*self.NL1).copy()

        self.NL2 = self.NL1.copy()
        self.NL1 = self._calc_right_hand_side(self.soln, self.t)
        self.soln = (self.L2*self.soln + self.c2*self.dt*self.NL1 \
                    + self.d1*self.dt*self.NL2).copy()

        self.NL2 = self.NL1.copy()
        self.NL1 = self._calc_right_hand_side(self.soln, self.t)

        self.soln = (self.L3*self.soln + self.c3*self.dt*self.NL1 \
                    + self.d2*self.dt*self.NL2).copy()

    ## RK4
    def _step_forward_RK4(self):
        """ March the system forward using a ETDRK4 scheme """

        self._calc_right_hand_side(self.soln, self.t)
        self.NL1 = self.RHS.copy() - self.linearCoeff*self.soln
                    
        t1 = self.t + self.dt/2
        self.soln1 = self.soln + self.dt/2*self.NL1, 
        self._calc_right_hand_side(self.soln1, t1) 
        self.NL2 = self.RHS.copy() - self.linearCoeff*self.soln1

        self.soln1 = self.soln + self.dt/2*self.NL2
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

