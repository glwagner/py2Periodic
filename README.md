# py2Periodic

This code solves several PDEs on doubly-periodic two-dimensional domains, 
and provides a framework for generating solvers with new physics.

This is achieved by introducing a "base-class, sub-class" relationship between
the numerical and physical aspects of the solver. The physics-agnostic attributes 
of the base-class, described in `doublyPeriodic.py`, point to physical and 
spectral grids, time-stepping routine and parameters, and the 2D Fourier 
transform method. The sub-class-specific attributes define the linear and 
nonlinear parts of the equation, as well as additional state or diagnostic 
variables required for the model's solution. Examples of sub-classes 
are `twoDimensionalTurbulence.py` and `hydrostaticWaveEquationInXY.py`.
