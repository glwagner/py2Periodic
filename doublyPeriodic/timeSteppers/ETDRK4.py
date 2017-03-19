from __future__ import division
import numpy as np
from numpy import pi, exp, sqrt, cos, sin

# ETDRK4
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
