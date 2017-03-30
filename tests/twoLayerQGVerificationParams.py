from numpy import pi

# Parameters for a simple test of the twoLayerQG physics
testParams = { 
    'Lx'         : 1.0e6, 
    'f0'         : 1.0e-4, 
    'beta'       : 2.0e-11,
    'defRadius'  : 1.5e4, 
    'H1'         : 500, 
    'H2'         : 2000, 
    'U1'         : 2.0e-1, 
    'U2'         : 1.0e-1, 
    'drag'       : 1.0e-6, 
    'visc'       : 1.0e7, 
    'viscOrder'  : 4.0, 
    'nx'         : 128, 
    'dt'         : 2.0e4, 
    'timeStepper': 'RK4', 
    'nThreads'   : 4, 
}

# Params for attempting a pyqg validation
pyqgParams = { 
    'Lx'         : 1.0e6, 
    'f0'         : 4.176e-3, 
    'beta'       : 1.5e-11,
    'defRadius'  : 1.5e4, 
    'H1'         : 500, 
    'H2'         : 2000, 
    'U1'         : 2.5e-2, 
    'U2'         : 0.0, 
    'drag'       : 5.787e-7, 
    'visc'       : 1.0e6, 
    'viscOrder'  : 4.0, 
    'nx'         : 128, 
    'dt'         : 8000, 
    'timeStepper': 'RK4', 
    'nThreads'   : 4, 
}

# Params for comparing twoLayerQG to Glenn's QG code
H1 = 1.4610e4
H2 = 5.0*H1
glennsParams = { 
    'Lx'         : 1.0e3, 
    'f0'         : 8.64, 
    'beta'       : 1.728e-3,
    'defRadius'  : 40.0, 
    'H1'         : H1, 
    'H2'         : H2, 
    'U1'         : 25.0, 
    'U2'         : 5.0, 
    'drag'       : 0.1, 
    'visc'       : 1.0e-4, 
    'viscOrder'  : 4.0, 
    'nx'         : 128, 
    'dt'         : 1.0/128.0,
    'timeStepper': 'ETDRK4', 
    'nThreads'   : 4, 
}
