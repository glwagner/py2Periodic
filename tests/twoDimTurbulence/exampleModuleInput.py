from numpy import pi

paramsInFile = {
    'nx'         : 128, 
    'Lx'         : 2.0*pi, 
    'dt'         : 5.0e-2,
    'visc'       : 1.0e-7, 
    'viscOrder'  : 4.0, 
    'nThreads'   : 2, 
    'timeStepper': 'RK4',
}
