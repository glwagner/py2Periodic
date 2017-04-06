from numpy import pi

paramsInFile = {
    'nx'         : 256, 
    'Lx'         : 2.0*pi, 
    'dt'         : 1.0e-2,
    'visc'       : 1.0e-5, 
    'viscOrder'  : 4.0, 
    'nThreads'   : 4, 
    'timeStepper': 'forwardEuler',
}
