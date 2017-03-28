# py2Periodic

This code is under rapid and somewhat chaotic development.

When it is complete, this code will solve several PDEs on doubly-periodic two-dimensional 
domains, and provide a low-level framework for rapidly generating solvers with new physics.

This is achieved by introducing a "base-class, sub-class" relationship between
the numerical and physical aspects of the solver. The physics-agnostic attributes 
of the base-class, described in `doublyPeriodic.py`, point to physical and 
spectral grids, time-stepping routine and parameters, and the 2D Fourier 
transform method. The sub-class-specific attributes define the linear and 
nonlinear parts of the equation, as well as additional state or diagnostic 
variables required for the model's solution. Examples of sub-classes 
are `twoDimensionalTurbulence.py` and `hydrostaticWaveEquationInXY.py`.
