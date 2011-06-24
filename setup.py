#!/usr/bin/env python

from distutils.core import setup

setup(name='rt1d',
      version='1.0',
      description='1D Radiative Transfer',
      author='Jordan Mirocha',
      author_email='mirochaj@gmail.com',
      url='https://bitbucket.org/mirochaj/rt1d',
      packages=['rt1d', 'rt1d.mods', 'rt1d.analysis'],
     )