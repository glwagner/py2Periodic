import numpy as np
import time as timeTools
import matplotlib.pyplot as plt

from ..doublyPeriodic import doublyPeriodicModel
from numpy import pi 

class model(doublyPeriodicModel):
    def __init__(self, name = None,
            # Grid parameters
            nx = 128, ny = None, Lx = 2.0*pi, Ly = None, 
            # Solver parameters
            t  = 0.0,  
            dt = 1.0e-2,                   # Numerical timestep
            step = 0,                      # Initial or current step of the model
            timeStepper = "forwardEuler",  # Time-stepping method
            nThreads = 1,                  # Number of threads for FFTW
            useFilter = False,             # Use exp filter rather than dealias
            # 
            # Two-dimensional turbulence parameters: arbitrary-order viscosity
            visc = 1.0e-4,
            viscOrder = 2.0,
        ):

        # Scalar attributes specific to the Physical Problem
        self.visc = visc
        self.viscOrder = viscOrder

        doublyPeriodicModel.__init__(self, name = name, 
            physics = "two-dimensional turbulence",
            nVars = 1,
            realVars = True,
            # Persistant doublyPeriodic initialization arguments 
            nx = nx, ny = ny, Lx = Lx, Ly = Ly, t = t, dt = dt, step = step,
            timeStepper = timeStepper, nThreads = nThreads, useFilter = useFilter,
        )

        # Set a default initial condition
        self.set_physical_soln(
            0.1*np.random.standard_normal(self.physSolnShape))

        self.update_state_variables()

        # Initialize default diagnostics
        self.add_diagnostic('CFL', lambda self: self._calc_CFL(),
            description="Maximum CFL number")

        self.add_diagnostic('KE', lambda self: self._calc_KE(),
            description="Total kinetic energy")
        
    # Methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def describe_physics(self):
        print("""
            This model solves the two-dimensional Navier-Stokes equation in \n
            streamfunction-vorticity formulation, with a variable-order \n
            hyperdissipation operator. There is a single prognostic \n
            variable: vorticity in Fourier space.
        """)


    def _init_linear_coeff(self):
        """ Calculate the coefficient that multiplies the linear left hand
            side of the equation """

        self.linearCoeff[:, :, 0] = \
            -self.visc*(self.k**2.0 + self.l**2.0)**(self.viscOrder/2.0)
       

    def _calc_right_hand_side(self, soln, t):
        """ Calculate the nonlinear right hand side """

        qh = soln[:, :, 0]

        # Transform of streamfunction and physical vorticity and velocity
        self.psih = -qh.copy()
        self.psih *= self.__oneOverKay2

        self.q = self.ifft2(qh)

        self.Uh = -self.psih.copy()
        self.Uh *= self.__jl

        self.u = self.ifft2(self.Uh)

        self.Uh = self.psih.copy()
        self.Uh *= self.__jk
        self.v = self.ifft2(self.Uh)

        self.Uq = self.u.copy()
        self.Uq *= self.q

        self.RHS[:, :, 0] = -self.fft2(self.Uq)
        self.RHS[:, :, 0] *= self.__jk

        self.Uq = self.v.copy()
        self.Uq *= self.q

        self.Uh = self.fft2(self.Uq)
        self.Uh *= self.__jl

        self.RHS[:, :, 0] -= self.Uh

        self._dealias_RHS()
    

    def _init_problem_parameters(self):
        """ Pre-allocate parameters in memory """

        self.__jk = 1j*self.k
        self.__jl = 1j*self.l

        # Divide-safe square wavenumber magnitude
        self.__divideSafeKay2 = self.k**2.0 + self.l**2.0
        self.__divideSafeKay2[0, 0] = float('Inf')
        self.__oneOverKay2 = 1.0 / self.__divideSafeKay2

        # Transformed streamfunction and physical vorticity and velocity
        self.psih = np.zeros(self.specVarShape, np.dtype('complex128'))
        self.Uh   = np.zeros(self.specVarShape, np.dtype('complex128'))
        self.q    = np.zeros(self.physVarShape, np.dtype('float64'))
        self.u    = np.zeros(self.physVarShape, np.dtype('float64'))
        self.v    = np.zeros(self.physVarShape, np.dtype('float64'))
        self.Uq   = np.zeros(self.physVarShape, np.dtype('float64'))
            

    def update_state_variables(self):
        """ Update diagnostic variables to current model state """

        qh = self.soln[:, :, 0]

        # Transform of streamfunction and physical vorticity and velocity
        self.psih = - qh * self.__oneOverKay2
        self.q = self.ifft2(qh)
        self.u = -self.ifft2(self.__jl*self.psih)
        self.v =  self.ifft2(self.__jk*self.psih)


    def set_q(self, q):
        self.soln[:, :, 0] = self.fft2(q)
        self._dealias_soln()
        self.update_state_variables()


    def visualize_model_state(self, show=False):
        """ Visualize the model state """

        self.update_state_variables() 
        self.evaluate_diagnostics()

        fig = plt.figure('Vorticity'); plt.clf()
        ax = plt.subplot(111)

        (maxVorticity, scale) = (np.max(np.abs(self.q)), 0.8)
        (cmin, cmax) = (-scale*maxVorticity, scale*maxVorticity)

        plt.pcolormesh(self.x, self.y, self.q, cmap='RdBu_r', 
            vmin=cmin, vmax=cmax)
        plt.axis('square')

        plt.xlabel('$x$', labelpad=5.0); 
        plt.ylabel('$y$', labelpad=12.0)
        plt.colorbar()

        message = '$t = {:.2e}$'.format(self.t)
        title = '$q$ ($\mathrm{s^{-1}}$)'
        position = ax.get_position()

        plt.text(0.00, 1.03, message, transform=ax.transAxes) 
        plt.text(1.00, 1.03, title, transform=ax.transAxes,
            HorizontalAlignment='right') 

        if show:
            plt.pause(0.01)
        else:
            plt.savefig('{}/{}_{:09d}'.format(
                self.plotDirectory, self.runName, self.step))


    def _print_status(self):
        """ Print model status """
        tc = timeTools.time() - self.timer

        # Update model state and calculate diagnostics
        self.update_state_variables() 
        self.evaluate_diagnostics()
            
        print( \
            "step = {:.2e}, clock = {:.2e} s, ".format(self.step, tc) +
            "t = {:.2e} s, ".format(self.t) +
            "KE = {:.2e}, ".format(self.diagnostics['KE']['value']) +
            "CFL = {:.3f}".format(self.diagnostics['CFL']['value'])
        )

        self.timer = timeTools.time()


    def describe_model(self):
        print("\nThis is a doubly-periodic spectral model for \n" + \
                "{:s} \n".format(self.physics) + \
                "with the following attributes:\n\n" + \
                " Domain           : {:.2e} X {:.2e} m\n".format( \
                    self.Lx, self.Ly) + \
                " Grid             : {:d} X {:d}\n".format(self.nx, self.ny) + \
                " (Hyper)viscosity : {:.2e} m^{:d}/s\n".format( \
                    self.visc, int(self.viscOrder)) + \
                " Comp. threads    : {:d} \n".format(self.nThreads) \
        )


    def random_energy_spectrum(self, q0rms=0.01, kPeak=16):
        """ Generate a random initial condition with an rms 
            Rossby number of q0rms and an energy spectrum peaked
            at the non-dimensional wavenumber kPeak"""

        # Dimensional peak wave number
        dimKPeak = kPeak * 2.0*pi/self.Lx

        # Random phases
        phase = 2.0*pi*np.random.rand(self.nl, self.nk)

        q0h = -np.exp(1j*phase)*(self.k**2.0+self.l**2.0)**(3.0/2.0) \
            / (1.0 + np.sqrt(self.k**2.0+self.l**2.0)/dimKPeak)**8.0

        q0 = self.ifft2(q0h)
        q0 *= q0rms / np.sqrt( (q0**2.0).mean() )
        
        return q0

    # Diagnostic-calculating functions  - - - - - - - - - - - - - - - - - - - -
    def _calc_CFL(self): 
        """ Calculate the maximum CFL number in the model """

        maxSpeed = (np.sqrt(self.u**2.0 + self.v**2.0)).max()
        CFL = maxSpeed * self.dt * self.nx/self.Lx

        return CFL


    def _calc_KE(self): 
        """ Calculate the total kinetic energy """

        KE = np.sum( self.Lx*self.Ly*(self.k**2.0+self.l**2.0)
                        * np.abs(self.psih)**2.0
        )

        return KE
