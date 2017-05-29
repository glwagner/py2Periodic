import numpy as np 
import numexpr as ne 
from numpy import pi

# Time steppers for the doublyPeriodicModel class

class methods(object):

    class forwardEuler(object):
        """ The forward Euler time-stepping method is a simple 1st-order 
            explicit method with poor stability characteristics. It is
            described, among other places, in Bewley's Numerical Renaissance. """

        def __init__(self, model):
            self.model = model

        def step_forward(self):
            """ Step the solution forward in time """

            m = self.model

            m._calc_right_hand_side(m.soln, m.t)
            m.soln += m.dt*(m.RHS + m.linearCoeff*m.soln)

            m.t += m.dt
            m.step += 1


    class forwardEuler_ne(object):
        """ The forward Euler time-stepping method is a simple 1st-order 
            explicit method with poor stability characteristics. It is
            described, among other places, in Bewley's Numerical Renaissance. """

        def __init__(self, model):
            self.model = model

        def step_forward(self):
            """ Step the solution forward in time """

            # Set up views for numexpr
            m = self.model

            dt = m.dt
            RHS = m.RHS
            soln = m.soln
            linearCoeff = m.linearCoeff

            m._calc_right_hand_side(m.soln, m.t)
            ne.evaluate("soln + dt*(RHS+linearCoeff*soln)", out=soln)

            m.t += dt
            m.step += 1


    class RK4_ne(object):
        """ RK4 is the classical explicit 4th-order Runge-Kutta time-stepping
            method. It uses a series of substeps/estimators to achieve 4th-order
            accuracy over each individual time-step, at the cost of requiring
            relatively more evaluations of the nonlinear right hand side.
            It is described, among other places, in Bewley's Numerical
            Renaissance. """


        def __init__(self, model):

            self.model = model

            # Allocate intermediate solution variable
            self.soln1 = np.zeros(model.specSolnShape, np.dtype('complex128'))

            # Allocate nonlinear terms
            self.RHS1 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.RHS2 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.RHS3 = np.zeros(model.specSolnShape, np.dtype('complex128'))


        def step_forward(self):
            """ Step the solution forward in time """

            # Set up views for numexpr
            m = self.model

            RHS         = m.RHS
            linearCoeff = m.linearCoeff
            soln        = m.soln
            dt          = m.dt
            dt2         = m.dt/2.0

            RHS1  = self.RHS1
            RHS2  = self.RHS2
            RHS3  = self.RHS3
            soln1 = self.soln1


            # Substep 1
            m._calc_right_hand_side(soln, m.t)
            ne.evaluate("RHS + linearCoeff*soln", out=RHS1)

            # Substep 2
            t1 = m.t + dt2

            ne.evaluate("soln + dt2*RHS1", out=soln1)
            m._calc_right_hand_side(soln1, t1) 
            ne.evaluate("RHS + linearCoeff*soln1", out=RHS2)

            # Substep 3
            ne.evaluate("soln + dt2*RHS2", out=soln1)
            m._calc_right_hand_side(soln1, t1) 
            ne.evaluate("RHS + linearCoeff*soln1", out=RHS3)

            # Substep 4
            t1 = m.t + dt

            ne.evaluate("soln + dt*RHS3", out=soln1)
            m._calc_right_hand_side(soln1, t1) 
            ne.evaluate("RHS + linearCoeff*soln1", out=RHS)

            ne.evaluate("soln + dt*(RHS1/6.0 + RHS2/3.0 + RHS3/3.0 + RHS/6.0)", 
                out=soln)
            m.t += m.dt
            m.step += 1


    class RK4(object):
        """ RK4 is the classical explicit 4th-order Runge-Kutta time-stepping
            method. It uses a series of substeps/estimators to achieve 4th-order
            accuracy over each individual time-step, at the cost of requiring
            relatively more evaluations of the nonlinear right hand side.
            It is described, among other places, in Bewley's Numerical
            Renaissance. """


        def __init__(self, model):

            self.model = model

            # Allocate intermediate solution variable
            self.soln1 = np.zeros(model.specSolnShape, np.dtype('complex128'))

            # Allocate nonlinear terms
            self.RHS1 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.RHS2 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.RHS3 = np.zeros(model.specSolnShape, np.dtype('complex128'))


        def step_forward(self):
            """ Step the solution forward in time """

            # Set up views for numexpr
            m = self.model

            # Substep 1
            m._calc_right_hand_side(m.soln, m.t)
            self.RHS1 = m.RHS + m.linearCoeff*m.soln

            # Substep 2
            t1 = m.t + m.dt/2.0
            self.soln1 = m.soln + m.dt/2.0*self.RHS1

            m._calc_right_hand_side(self.soln1, t1) 
            self.RHS2 = m.RHS + m.linearCoeff*self.soln1

            # Substep 3
            self.soln1 = m.soln + m.dt/2.0*self.RHS2

            m._calc_right_hand_side(self.soln1, t1) 
            self.RHS3 = m.RHS + m.linearCoeff*self.soln1

            # Substep 4
            t1 = m.t + m.dt
            self.soln1 = m.soln + m.dt*self.RHS3

            m._calc_right_hand_side(self.soln1, t1) 
            m.RHS += m.linearCoeff*self.soln1

            # Final step
            m.soln += m.dt*(   
                1.0/6.0*self.RHS1 + 1.0/3.0*self.RHS2
              + 1.0/3.0*self.RHS3 + 1.0/6.0*m.RHS
            )
            m.t += m.dt
            m.step += 1





    class ETDRK4_ne(object):
        """ ETDRK4 is a 4th-order Runge-Kutta exponential time-differencing
            method described by Cox and Matthews (2002). The prefactors are
            computed by contour integration in the complex plane, as described
            by Kassam and Trefethen (2005). """

        def __init__(self, model):
            self.model = model

            # Calculate coefficients with circular line integral in complex plane
            nCirc = 32          
            rCirc = 1.0       
            circ = rCirc*np.exp(2j*pi*(np.arange(1, nCirc+1)-1/2)/nCirc) 

            # Circular contour around the point to be calculated
            linearCoeffDt = model.dt*model.linearCoeff
            zc = linearCoeffDt[..., np.newaxis] \
                    + circ[np.newaxis, np.newaxis, np.newaxis, ...]

            # Four coefficients, zeta, alpha, beta, and gamma
            self.zeta = model.dt*( \
                            (np.exp(zc/2.0) - 1.0) / zc \
                                ).mean(axis=-1)

            self.alph = model.dt*( \
                          (-4.0 - zc + np.exp(zc)*(4.0-3.0*zc+zc**2.0)) / zc**3.0 \
                                ).mean(axis=-1)

            self.beta = model.dt*( \
                          (2.0 + zc + np.exp(zc)*(-2.0+zc) ) / zc**3.0 \
                                ).mean(axis=-1)

            self.gamm = model.dt*( \
                          (-4.0 - 3.0*zc - zc**2.0 + np.exp(zc)*(4.0-zc)) / zc**3.0 \
                                ).mean(axis=-1)
                                  
            # Pre-calculate an exponential     
            self.linearExpDt     = np.exp(model.dt*model.linearCoeff)
            self.linearExpHalfDt = np.exp(model.dt/2.0*model.linearCoeff)

            # Allocate intermediate solution variable
            self.soln1 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.soln2 = np.zeros(model.specSolnShape, np.dtype('complex128'))

            # Allocate nonlinear terms
            self.NL1 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.NL2 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.NL3 = np.zeros(model.specSolnShape, np.dtype('complex128'))

        def step_forward(self):
            """ Step the solution forward in time """
        
            # Views for numexpr
            m    = self.model
            dt   = m.dt
            dt2  = m.dt/2.0
            RHS  = m.RHS
            soln = m.soln

            soln1           = self.soln1
            soln2           = self.soln2
            linearExpDt     = self.linearExpDt
            linearExpHalfDt = self.linearExpHalfDt
            NL1             = self.NL1
            NL2             = self.NL2
            NL3             = self.NL3
            zeta            = self.zeta
            alph            = self.alph
            beta            = self.beta
            gamm            = self.gamm
            

            m._calc_right_hand_side(soln, m.t)
            NL1 = m.RHS.copy()

            ne.evaluate("linearExpHalfDt*soln + zeta*NL1", out=soln1)
            m._calc_right_hand_side(soln1, m.t+dt2)
            NL2 = m.RHS.copy()

            ne.evaluate("linearExpHalfDt*soln + zeta*NL2", out=soln2)
            m._calc_right_hand_side(soln2, m.t+dt2)
            NL3 = m.RHS.copy()

            ne.evaluate("linearExpHalfDt*soln1 + zeta*(2.0*NL3-NL1)", out=soln2)
            m._calc_right_hand_side(soln2, m.t+dt)

            # The final step
            ne.evaluate("linearExpDt*soln + alph*NL1 "
                        " + 2.0*beta*(NL2+NL3) + gamm*RHS", out=soln)
                        
            m.t += dt
            m.step += 1

            #self.soln1 = self.linearExpHalfDt*m.soln + self.zeta*self.NL1
            #self.soln2 = self.linearExpHalfDt*m.soln + self.zeta*self.NL2
            #m.soln = self.linearExpDt*m.soln \
            #            +     self.alph * self.NL1 \
            #            + 2.0*self.beta * (self.NL2 + self.NL3) \
            #            +     self.gamm * m.RHS


    class ETDRK4(object):
        """ ETDRK4 is a 4th-order Runge-Kutta exponential time-differencing
            method described by Cox and Matthews (2002). The prefactors are
            computed by contour integration in the complex plane, as described
            by Kassam and Trefethen (2005). """

        def __init__(self, model):
            self.model = model

            # Calculate coefficients with circular line integral in complex plane
            nCirc = 32          
            rCirc = 1.0       
            circ = rCirc*np.exp(2j*pi*(np.arange(1, nCirc+1)-1/2)/nCirc) 

            # Circular contour around the point to be calculated
            linearCoeffDt = model.dt*model.linearCoeff
            zc = linearCoeffDt[..., np.newaxis] \
                    + circ[np.newaxis, np.newaxis, np.newaxis, ...]

            # Four coefficients, zeta, alpha, beta, and gamma
            self.zeta = model.dt*( \
                            (np.exp(zc/2.0) - 1.0) / zc \
                                ).mean(axis=-1)

            self.alph = model.dt*( \
                          (-4.0 - zc + np.exp(zc)*(4.0-3.0*zc+zc**2.0)) / zc**3.0 \
                                ).mean(axis=-1)

            self.beta = model.dt*( \
                          (2.0 + zc + np.exp(zc)*(-2.0+zc) ) / zc**3.0 \
                                ).mean(axis=-1)

            self.gamm = model.dt*( \
                          (-4.0 - 3.0*zc - zc**2.0 + np.exp(zc)*(4.0-zc)) / zc**3.0 \
                                ).mean(axis=-1)
                                  
            # Pre-calculate an exponential     
            self.linearExpDt     = np.exp(model.dt*model.linearCoeff)
            self.linearExpHalfDt = np.exp(model.dt/2.0*model.linearCoeff)

            # Allocate intermediate solution variable
            self.soln1 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.soln2 = np.zeros(model.specSolnShape, np.dtype('complex128'))

            # Allocate nonlinear terms
            self.NL1 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.NL2 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.NL3 = np.zeros(model.specSolnShape, np.dtype('complex128'))

        def step_forward(self):
            """ Step the solution forward in time """
        
            m = self.model

            m._calc_right_hand_side(m.soln, m.t)
            self.NL1 = m.RHS.copy()

            t1 = m.t + m.dt/2
            self.soln1 = self.linearExpHalfDt*m.soln + self.zeta*self.NL1
            m._calc_right_hand_side(self.soln1, t1)
            self.NL2 = m.RHS.copy()

            self.soln2 = self.linearExpHalfDt*m.soln + self.zeta*self.NL2
            m._calc_right_hand_side(self.soln2, t1)
            self.NL3 = m.RHS.copy()

            t1 = m.t + m.dt
            self.soln2 = self.linearExpHalfDt*self.soln1 \
                + self.zeta*(2.0*self.NL3-self.NL1)
            m._calc_right_hand_side(self.soln2, t1)

            # The final step
            m.soln = self.linearExpDt*m.soln \
                        +     self.alph * self.NL1 \
                        + 2.0*self.beta * (self.NL2 + self.NL3) \
                        +     self.gamm * m.RHS
            m.t += m.dt
            m.step += 1


    class AB3(object):
        """ AB3 is the 3rd-order explicity Adams-Bashforth scheme, which employs 
            solutions from prior time-steps to achieve higher-order accuracy
            over forward Euler. AB3 is faster, but has a smaller linear
            stability region compared to RK4. """

        def __init__(self, model):

            self.model = model

            # Allocate right hand sides to be stored from previous steps
            self.RHSm1 = np.zeros(model.specSolnShape, np.dtype('complex128'))
            self.RHSm2 = np.zeros(model.specSolnShape, np.dtype('complex128'))


        def step_forward(self):
            """ Step the solution forward in time using the AB3 scheme """

            m = self.model

            # While RHS_{n-2} is unfilled, step forward with foward Euler.
            if not self.RHSm2.any():
                m._calc_right_hand_side(m.soln, m.t)
                m.soln += m.dt*(m.RHS + m.linearCoeff*m.soln)
            else:
                m._calc_right_hand_side(m.soln, m.t)
                m.RHS += m.linearCoeff*m.soln

                m.soln +=   23.0/12.0 * m.dt * m.RHS \
                             - 16.0/12.0 * m.dt * self.RHSm1 \
                             +  5.0/12.0 * m.dt * self.RHSm2

            # Store RHS for use in future time-steps.
            self.RHSm2 = self.RHSm1.copy()
            self.RHSm1 = m.RHS.copy()

            m.t += m.dt
            m.step += 1
