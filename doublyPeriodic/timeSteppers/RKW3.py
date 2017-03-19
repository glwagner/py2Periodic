from __future__ import division
import numpy as np
from numpy import pi, exp, sqrt, cos, sin

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
