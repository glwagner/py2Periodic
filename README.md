# Overview. 

**Note:** "py2Periodic" is under rapid and somewhat chaotic development.

py2Periodic provides a framework for solving PDEs with a 
pseudospectral method on doubly-periodic domains. In short, 
the framework defines a base-class, sub-class relationship between
the numerical and physical aspects of the problems it solves, 
respectively. The intention is to provide the user with the ability
to rapidly implement and customize new physical models. Moreover, because
all the physical models implemented with py2Periodic, code that sets up,
runs, and analyzes different physical problems can be reused or easily
adapated.

# Installation

At the moment the only way to use py2Periodic is to clone the git 
repository and attempt to use the examples as is. The project
will become a proper package in the future. The current requirements of
py2Periodic are:

* ``numpy``
* ``matplotlib``
* ``pyFFTW``
* ``h5py``

# Documentation

Little documentation exists right now. 

## The base-class: ``doublyPeriodic.py``
Note, however, that the base-class
code is contained in ``doublyPeriodic.py`` in the root directory
``py2Periodic``. The base-class definition defines the 
physical and numerical grids, the Fourier transform, various helper
functions for introducing scalar diagnostics to be calculated, 
routines that run the model, create plots, and save model output.

Time-steppers for the model, which are implemented as 
attributes instantiated and assigned to the model class at 
model instantiation, are defined in ``timeSteppers.py`` in the root
directory.

## Physical problem sub-classes

Physical problems are implemented as sub-classes of 
``doublyPeriodic.model``. The physical problems currently implemented are

* Two-dimensional turbulence (``twoDimTurbulence.py``)
* Linearized, hydrostatic, single-mode Boussinesq equations in xy 
(``linearizedBoussinesq_xy.py``)
* Two-layer quasi-geostrophic flow with optional 
bathymetry (``twoLayerQG.py``)
* Two-layer quasi-geostrophic flow with optional bathymetry with 
tracers in each layer (``twoLayerTracers.py``)
* The 'YBJ' model for a single near-inertial wave mode in 
two-dimensional turbulence (``nearInertialWaves_xy.py``)
* The single-mode hydrostatic wave equation modeling 
hydrostatic internal waves in two-dimensional turbulence (``hydrostaticWaveEqn_xy.py``)

## Tests and examples

Tests exist for all of the models. In addition, there is a single, 
somewhat underdeveloped example in ``/examples/waveIsotropization`` 
that first generates and saves a two-dimensional turbulent vorticity
field, and then introduces a wave field into the vorticity field in
both the hydrostatic wave equation model and the linearized
Boussinesq model. The wave field subsequently is scrambled and 
isotropized and, crucially, the two models generally agree.
