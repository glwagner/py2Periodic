import time
import numpy as np
from numpy import pi
import pyfftw
pyfftw.interfaces.cache.enable() 

try:   
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

class doublyPeriodicModel(object):

    # Initialize  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def __init__(
            self,
            name = "genericTwoDimensionalTurbulence", 
            # Grid parameters
            nx = 256,
            Lx = 2.0*pi, 
            ny = None,
            Ly = None, 
            nVars = 1,
            realVars = False,
            # Timestepping parameters
            t  = 0.0,  
            dt = 1.0e-2,                # Fixed numerical time-step.
            step = 0,                   # Initial or current step of the model
            # Computational parameters
            nThreads = 1,               # Number of threads for FFTW
            dealias  = True,
            # Simple I/O
            dnLog     = 1e2,
            dnSave    = 1e2,            # Interval to save (in timesteps)
            savingData = False, 
            # Plotting
            makingPlots = False,
            # Single physical parameter: viscosity
            nu = 1.0e-2, 
        ):

        # For convenience, use a default square, uniformly-gridded domain when 
        # user specifies only nx or Lx
        if ny is None: ny = nx
        if Ly is None: Ly = Lx

        # Assign initial parameters
        self.name = name
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.nVars = nVars
        self.realVars = realVars
        self.t = t 
        self.dt = dt
        self.step = step
        self.nThreads = nThreads
        self.dealias = dealias
        self.savingData = savingData
        self.makingPlots = makingPlots
        self.dnLog = dnLog
        self.dnSave = dnSave
        self.nu = nu

        # Finish initializing the doubly-periodic model
        self._init_solution()
        self._init_physical_grid()
        self._init_spectral_grid()
        self._init_fft()

    # Hidden methods  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _init_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """
        # The default model is 2D turbulence with Laplacian viscosity.
        self.linearCoeff = np.zeros(self.spectralShape, self.spectralType)
        self.linearCoeff[:, :, 0] = nu*(self.KK**2.0 + self.LL**2.0)

    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side of the equation """

        # Views for clarity:
        qh = self.soln[:, :, 0]

        # Streamfunction
        self.ph = - qh / self.kay2 

        self.u = -np.real(self.ifft2(self.jLL*self.ph))
        self.v =  np.real(self.ifft2(self.jKK*self.ph))

        # Right hand side describes advection of q
        self.RHS[:, :, 0] = -self.jKK*self.fft2(u*q) - self.jLL*self.fft2(v*q) 

    def _init_parameters(self):
        """ Pre-allocate parameters in memory in addition to the solution """

        # Possible shapes for variables. "realSpectralShape" is the shape of 
        # a real physical variable in spectral space.
        self.physicalShape = (self.ny, self.nx, self.nVars)

        if self.realVars:
            self.spectralShape = (self.ny, self.nx//2+1, self.nVars)
            self.physicalType = np.dtype('float64')
        else:
            self.spectralShape = self.physicalShape
            self.physicalType = np.dtype('complex128')

        # Divide-safe square wavenumber
        self.kay2 = self.KK**2.0 + self.LL**2.0
        self.kay2[0, 0] = float('Inf')

        # Prognostic variables  - - - - - - - - - - - - - - - - - - - - - - - -  
        # Initialize the solution.
        self.soln = np.zeros(self.spectralShape, self.spectralType)

        ## Vorticity and wave-field amplitude
        self.q = np.zeros((self.ny, self.nx), np.dtype('float64'))

        # Diagnostic variables  - - - - - - - - - - - - - - - - - - - - - - - -  
        ## Streamfunction transform
        self.ph = np.zeros((self.ny, self.nx), np.dtype('complex128'))
    
        ## Mean and wave velocity components 
        self.u = np.zeros((self.ny, self.nx), np.dtype('float64'))
        self.v = np.zeros((self.ny, self.nx), np.dtype('float64'))
            
    def _init_physical_grid(self):
        """ Initialize the physical grid """

        # Grid spacing
        self.dx = self.Lx/self.nx
        self.dy = self.Ly/self.ny

        self.xx = np.arange(0.0, self.Lx, self.dx)
        self.yy = np.arange(0.0, self.Ly, self.dy)

        # Two-dimensional grid space arrays
        self.XX, self.YY = np.meshgrid(self.xx, self.yy)

    def _init_spectral_grid(self):
        """ Initialize the spectral grid"""

        # Grid spacing
        self.dk = 2.0*pi/self.Lx
        self.dl = 2.0*pi/self.Ly

        self.ll = self.dl*np.append(np.arange(0.0, self.ny/2),
                                        np.arange(-self.ny/2, 0.0) )
    
        if self.realVars:
            self.kk = self.dk*np.arange(0.0, self.nx/2+1)
        else:
            self.kk = self.dk*np.append(np.arange(0.0, self.nx/2),
                                        np.arange(-self.nx/2, 0.0) )

        # Two-dimensional wavenumber arrays
        self.KK, self.LL = np.meshgrid(self.kk, self.ll)

        # Pre-computed products of 1j and wavenumber arrays
        self.jKK = 1j*self.KK
        self.jLL = 1j*self.LL

    def _init_fft(self):
        """ Initialize the fast Fourier transform routine. """

        if self.realVars:
            self.rfft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.rfft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
            self.irfft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.irfft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
        else:
            self.fft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.fft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
            self.ifft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.ifft2(x, threads=self.nThreads, \
                            planner_effort='FFTW_ESTIMATE'))
            
    def _print_status(self):
        """ Print model status """
        tc = time.time() - self.timer
        print("t = {:.3f} s, step = {:d}, ".format(self.t, self.step) + \
                "tComp = {:.3f} s".format(tc))
        self.timer = time.time()

    def _dealias_RHS(self):
        """ Dealias the RHS """
        if self.realVars:
            self.RHS[self.ny//3:2*self.ny//3, :, :] = 0.0
            self.RHS[:, self.nx//3:, :] = 0.0
        else:
            self.RHS[self.ny//3:2*self.ny//3, :, :] = 0.0
            self.RHS[:, self.nx//3:2*self.nx//3, :] = 0.0

    def _dealias_array(self, array):
        """ Dealias the Fourier transform of a real array """
        if self.realVars:
            array[self.nx//3:2*self.nx//3, :, :] = 0.0
            array[:, self.ny//3:, :] = 0.0
        else:
            array[self.ny//3:2*self.ny//3, :, :] = 0.0
            array[:, self.nx//3:2*self.nx//3, :] = 0.0
        
        return array

    # Visible methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def set_physical_soln(self, soln):
        """ Initialize model with a physical space solution """ 
        q = soln[:, :, 0]

        self.soln[:, :, 0] = self.fft2(q)
        self.soln = self._dealias(self.soln)

        self.update_state_variables()

    def set_spectral_soln(self, soln):
        """ Initialize model with a spectral space solution """ 
        self.soln = soln
        self.soln = self._dealias(self.soln)

        self.update_state_variables()

    def update_state_variables(self):
        """ Update diagnostic variables to current model state """

        # Views for clarity:
        qh = self.soln[:, :, 0]

        # Streamfunction
        self.ph = - qh / self.kay2 

        self.u = -np.real(self.ifft2(self.jLL*self.ph))
        self.v =  np.real(self.ifft2(self.jKK*self.ph))

    def __describe_model(self):
        """ Describe the current model state """

        print("\nThis is a doubly-periodic spectral model with the following " + \
                "attributes:\n\n" + \
                "   Domain       : {:.2e} X {:.2e} m\n".format(self.Lx, self.Ly) + \
                "   Resolution   : {:d} X {:d}\n".format(self.nx, self.ny) + \
                "   Timestep     : {:.2e} s\n".format(self.dt) + \
                "   Current time : {:.2e} s\n\n".format(self.t) + \
                "The FFT scheme uses {:d} thread(s).\n".format(self.nThreads))

    # Name mangle
    describe_model = __describe_model
