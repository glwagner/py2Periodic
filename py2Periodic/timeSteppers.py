import numpy as np 
from numpy import pi

# Time steppers for the doublyPeriodicModel class


## Forward Euler
def _describe_time_stepper_forwardEuler(self):
    print("""
        The forward Euler time-stepping method is a simple 1st-order 
        explicit method with poor stability characteristics. It is \n
        described, among other places, in Bewley's Numerical Renaissance.\n
          """)


def _init_time_stepper_forwardEuler(self):
    pass


def _step_forward_forwardEuler(self, dt=None):
    """ Step the solution forward in time using the forward Euler scheme """
    if dt is None: dt=self.dt

    self._calc_right_hand_side(self.soln, self.t)
    self.soln += dt*(self.RHS + self.linearCoeff*self.soln)

    self.t += dt
    self.step += 1



## 4th-order Runge-Kutta (RK4)
def _describe_time_stepper_RK4(self):
    print("""
        RK4 is the classical explicit 4th-order Runge-Kutta time-stepping \n
        method. It uses a series of substeps/estimators to achieve 4th-order \n 
        accuracy over each individual time-step, at the cost of requiring \n
        relatively more evaluations of the nonlinear right hand side. \n
        It is described, among other places, in Bewley's Numerical \n
        Renaissance.
          """)


def _init_time_stepper_RK4(self):
    """ Initialize and allocate vars for RK4 time-stepping """

    # Allocate intermediate solution variable
    self.__soln1 = np.zeros(self.specSolnShape, np.dtype('complex128'))

    # Allocate nonlinear terms
    self.__RHS1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
    self.__RHS2 = np.zeros(self.specSolnShape, np.dtype('complex128'))
    self.__RHS3 = np.zeros(self.specSolnShape, np.dtype('complex128'))


def _step_forward_RK4(self, dt=None):
    """ Step the solution forward in time using the RK4 scheme """

    if dt is None: dt=self.dt

    # Substep 1
    self._calc_right_hand_side(self.soln, self.t)
    self.__RHS1 = self.RHS + self.linearCoeff*self.soln

    # Substep 2
    t1 = self.t + dt/2.0
    self.__soln1 = self.soln + dt/2.0*self.__RHS1

    self._calc_right_hand_side(self.__soln1, t1) 
    self.__RHS2 = self.RHS + self.linearCoeff*self.__soln1

    # Substep 3
    self.__soln1 = self.soln + dt/2.0*self.__RHS2

    self._calc_right_hand_side(self.__soln1, t1) 
    self.__RHS3 = self.RHS + self.linearCoeff*self.__soln1

    # Substep 4
    t1 = self.t + dt
    self.__soln1 = self.soln + dt*self.__RHS3

    self._calc_right_hand_side(self.__soln1, t1) 
    self.RHS += self.linearCoeff*self.__soln1

    # Final step
    self.soln += dt*(   1.0/6.0*self.__RHS1 + 1.0/3.0*self.__RHS2 \
                      + 1.0/3.0*self.__RHS3 + 1.0/6.0*self.RHS )
    self.t += dt
    self.step += 1



## 4th Order Runge-Kutta Exponential Time Differenceing (ETDRK4)
def _describe_time_stepper_ETDRK4(self):
    print("""
        ETDRK4 is a 4th-order Runge-Kutta exponential time-differencing \n
        method described by Cox and Matthews (2002). The prefactors are \n
        computed by contour integration in the complex plane, as described \n
        by Kassam and Trefethen (2005).
          """)

    
def _init_time_stepper_ETDRK4(self):
    """ Initialize and allocate vars for ETDRK4 time-stepping """

    # Calculate coefficients with circular line integral in complex plane
    nCirc = 32          
    rCirc = 1.0       
    circ = rCirc*np.exp(2j*pi*(np.arange(1, nCirc+1)-1/2)/nCirc) 

    # Circular contour around the point to be calculated
    linearCoeffDt = self.dt*self.linearCoeff
    zc = linearCoeffDt[..., np.newaxis] \
            + circ[np.newaxis, np.newaxis, np.newaxis, ...]

    # Four coefficients, zeta, alpha, beta, and gamma
    self.__zeta = self.dt*( \
                    (np.exp(zc/2.0) - 1.0) / zc \
                        ).mean(axis=-1)

    self.__alph = self.dt*( \
                  (-4.0 - zc + np.exp(zc)*(4.0-3.0*zc+zc**2.0)) / zc**3.0 \
                        ).mean(axis=-1)

    self.__beta = self.dt*( \
                  (2.0 + zc + np.exp(zc)*(-2.0+zc) ) / zc**3.0 \
                        ).mean(axis=-1)

    self.__gamm = self.dt*( \
                  (-4.0 - 3.0*zc - zc**2.0 + np.exp(zc)*(4.0-zc)) / zc**3.0 \
                        ).mean(axis=-1)
                          
    # Pre-calculate an exponential     
    self.__linearExpDt     = np.exp(self.dt*self.linearCoeff)
    self.__linearExpHalfDt = np.exp(self.dt/2.0*self.linearCoeff)

    # Allocate intermediate solution variable
    self.__soln1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
    self.__soln2 = np.zeros(self.specSolnShape, np.dtype('complex128'))

    # Allocate nonlinear terms
    self.__NL1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
    self.__NL2 = np.zeros(self.specSolnShape, np.dtype('complex128'))
    self.__NL3 = np.zeros(self.specSolnShape, np.dtype('complex128'))


def _step_forward_ETDRK4(self):
    """ Step the solution forward in time using the ETDRK4 scheme """

    self._calc_right_hand_side(self.soln, self.t)
    self.__NL1 = self.RHS.copy()

    t1 = self.t + self.dt/2
    self.__soln1 = self.__linearExpHalfDt*self.soln + self.__zeta*self.__NL1
    self._calc_right_hand_side(self.__soln1, t1)
    self.__NL2 = self.RHS.copy()

    self.__soln2 = self.__linearExpHalfDt*self.soln + self.__zeta*self.__NL2
    self._calc_right_hand_side(self.__soln2, t1)
    self.__NL3 = self.RHS.copy()

    t1 = self.t + self.dt
    self.__soln2 = self.__linearExpHalfDt*self.__soln1 \
        + self.__zeta*(2.0*self.__NL3-self.__NL1)
    self._calc_right_hand_side(self.__soln2, t1)

    # The final step
    self.soln = self.__linearExpDt*self.soln \
                +     self.__alph * self.__NL1 \
                + 2.0*self.__beta * (self.__NL2 + self.__NL3) \
                +     self.__gamm * self.RHS
    self.t += self.dt
    self.step += 1



## 3rd-order Adams-Bashforth (AB3)
def _describe_time_stepper_AB3(self):
    print("""
        AB3 is the 3rd-order explicity Adams-Bashforth scheme, which employs 
        solutions from prior time-steps to achieve higher-order accuracy   \n
        over forward Euler. AB3 is faster, but has a smaller linear \n
        stability region compared to RK4.
          """)


def _init_time_stepper_AB3(self):
    """ Initialize and allocate vars for AB3 time-stepping """

    # Allocate right hand sides to be stored from previous steps
    self.__RHSm1 = np.zeros(self.specSolnShape, np.dtype('complex128'))
    self.__RHSm2 = np.zeros(self.specSolnShape, np.dtype('complex128'))


def _step_forward_AB3(self):
    """ Step the solution forward in time using the AB3 scheme """

    # While RHS_{n-2} is unfilled, step forward with foward Euler.
    if not self.__RHSm2.any():
        self._calc_right_hand_side(self.soln, self.t)
        self.soln += self.dt*(self.RHS + self.linearCoeff*self.soln)
    else:
        self._calc_right_hand_side(self.soln, self.t)
        self.RHS += self.linearCoeff*self.soln

        self.soln +=   23.0/12.0 * self.dt * self.RHS \
                     - 16.0/12.0 * self.dt * self.__RHSm1 \
                     +  5.0/12.0 * self.dt * self.__RHSm2

    # Store RHS for use in future time-steps.
    self.__RHSm2 = self.__RHSm1.copy()
    self.__RHSm1 = self.RHS.copy()

    self.t += self.dt
    self.step += 1
