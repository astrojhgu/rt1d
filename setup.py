#!/usr/bin/env python

from distutils.core import setup
#from setuptools import setup

setup(name='rt1d',
      version='1.0',
      description='1D Radiative Transfer',
      author='Jordan Mirocha',
      author_email='mirochaj@gmail.com',
      url='https://bitbucket.org/mirochaj/rt1d',
      packages=['rt1d', 'rt1d.analysis', 'rt1d.static', 'rt1d.physics', 
                'rt1d.sources', 'rt1d.evolve', 'rt1d.util', 'rt1d.run'],
      #install_requires = ['numpy', 'scipy', 'h5py'],            
     )
     

     