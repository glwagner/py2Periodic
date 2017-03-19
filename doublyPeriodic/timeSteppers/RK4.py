from __future__ import division
import numpy as np
from numpy import pi, exp, sqrt, cos, sin

# RK4
def _step_forward_RK4(self):
    """ March the system forward using a ETDRK4 scheme """

    self._calc_right_hand_side(self.soln, self.t)
    self.NL1 = self.RHS - self.linearCoeff*self.soln

    t1 = self.t + self.dt/2
    self.soln1 = self.soln + self.dt/2.0*self.NL1

    self._calc_right_hand_side(self.soln1, t1) 
    self.NL2 = self.RHS - self.linearCoeff*self.soln1

    self.soln1 = self.soln + self.dt/2.0*self.NL2
    self._calc_right_hand_side(self.soln1, t1) 
    self.NL3 = self.RHS - self.linearCoeff*self.soln1

    t1 = self.t + self.dt
    self.soln1 = self.soln + self.dt*self.NL3
    self._calc_right_hand_side(self.soln1, t1) 
    self.NL4 = self.RHS - self.linearCoeff*self.soln1

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
