import numpy as np

# Time steppers for the doublyPeriodicModel class.

## Forward Euler
def _step_forward_forward_euler(self):
    """ March system forward in time using forward Euler scheme """
    self._calc_right_hand_side(self.soln, self.t)
    self.soln += self.dt*(self.RHS.copy() - self.linearCoeff*self.soln)

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
    zc = linearCoeffDt[..., np.newaxis] \
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
