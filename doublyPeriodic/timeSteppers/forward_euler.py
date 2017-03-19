from __future__ import division
import numpy as np
from numpy import pi, exp, sqrt, cos, sin

# Forward Euler
def _step_forward(self):
    """ March system forward in time using forward Euler scheme """

    self._calc_right_hand_side(self.soln, self.t)
    self.soln += self.dt*(self.RHS - self.linearCoeff*self.soln)

def _init(self):
    """ Initialize and allocate vars for forward Euler time-marching """

    if self.realVars:
        self.RHS = np.zeros(self.realSpectralShape, np.dtype('complex128'))
    else:
        self.RHS = np.zeros(self.physicalShape, np.dtype('complex128'))
