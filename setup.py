from setuptools import setup

setup(
    name = 'py2Periodic',
    version = '0.1',
    description = "A family of doubly-periodic pseudospectral Python models", 
    url = 'http://github.com/glwagner/py2Periodic',
    author = 'Gregory L. Wagner',
    author_email = 'wagner.greg@gmail.com',
    license = 'MIT',
    packages = ['py2Periodic'],
    install_requires = [
        'numpy', 
        'matplotlib', 
        'pyfftw', 
    ],
    zip_safe = False,
)
