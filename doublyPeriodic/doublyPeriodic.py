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
            name = "genericDoublyPeriodicModel", 
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

        # Finish initializing the doubly-periodic model
        self._init_solution()
        self._init_physical_grid()
        self._init_spectral_grid()
        self._init_fft()

    # Hidden methods  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _init_solution(self):
        """ Initialize the spectral-space solution """

        # Possible shapes for variables 
        self.physicalShape = (self.ny, self.nx, self.nVars)
        self.realSpectralShape = (self.ny, self.nx//2+1, self.nVars)

        if self.realVars:
            self.soln = np.zeros(self.realSpectralShape, np.dtype('complex128'))
        else:
            self.soln = np.zeros(self.physicalShape, np.dtype('float64'))
            
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
        self.kk = self.dk*np.append(np.arange(0.0, self.nx/2),
                                        np.arange(-self.nx/2, 0.0) )

        # Additional wavenumber grid to optimize solutions for 'r'eal variables
        self.rkk = self.dk*np.arange(0.0, self.nx/2+1)

        # Two-dimensional wavenumber arrays
        self.rKK, self.rLL = np.meshgrid(self.rkk, self.ll)
        self.KK, self.LL = np.meshgrid(self.kk, self.ll)

        # Pre-computed products of 1j and wavenumber arrays
        self.jKK = 1j*self.KK
        self.jLL = 1j*self.LL

        self.rjKK = 1j*self.rKK
        self.rjLL = 1j*self.rLL
         
    def _init_fft(self):
        """ Initialize the fast Fourier transform routine. """

        self.fft2 = (lambda x :
                pyfftw.interfaces.numpy_fft.fft2(x, threads=self.nThreads, \
                        planner_effort='FFTW_ESTIMATE'))
        self.ifft2 = (lambda x :
                pyfftw.interfaces.numpy_fft.ifft2(x, threads=self.nThreads, \
                        planner_effort='FFTW_ESTIMATE'))
        self.rfft2 = (lambda x :
                pyfftw.interfaces.numpy_fft.rfft2(x, threads=self.nThreads, \
                        planner_effort='FFTW_ESTIMATE'))
        self.irfft2 = (lambda x :
                pyfftw.interfaces.numpy_fft.irfft2(x, threads=self.nThreads, \
                        planner_effort='FFTW_ESTIMATE'))

        #self.fft2  = (lambda x : np.fft.fft2(x))
        #self.ifft2 = (lambda x : np.fft.ifft2(x))
        #self.rfft2  = (lambda x : np.fft.rfft2(x))
        #self.irfft2 = (lambda x : np.fft.irfft2(x))

    def _print_status(self):
        """ Print model status """
        tc = time.time() - self.timer
        print("t = {:.3f} s, step = {:d}, ".format(self.t, self.step) + \
                "tComp = {:.3f} s".format(tc))
        self._start_timer()

    def _print_snapshot_status(self):
        """ Print model status """
        tc = time.time() - self.timer
        print("Snapshot taken at t = {:.3f} s, step = {:d}, " + \
                "tComp = {:.3f} s".format(self.t, self.step, tc))
        self._start_timer()

    def _dealias_real_RHS(self, array):
        """ Dealias the RHS for real variables """
        self.RHS[self.ny//3:2*self.ny//3, :, :] = 0.0
        self.RHS[:, self.nx//3:, :] = 0.0

    def _dealias_imag_RHS(self, array):
        """ Dealias the RHS for real variables """
        self.RHS[self.ny//3:2*self.ny//3, :, :] = 0.0
        self.RHS[:, self.nx//3:2*self.nx//3, :] = 0.0

    def _dealias_real(self, array):
        """ Dealias the Fourier transform of a real array """
        
        array[self.nx//3:2*self.nx//3, :, :] = 0.0
        array[:, self.ny//3:, :] = 0.0

        return array

    def _dealias_imag(self, array):
        """ Dealias the Fourier transform of an imaginary array """
        
        array[self.ny//3:2*self.ny//3, :, :] = 0.0
        array[:, self.nx//3:2*self.nx//3, :] = 0.0

        return array

    def _start_timer(self):
        """ Store current time """
        self.timer = time.time()

    # Visible methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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
