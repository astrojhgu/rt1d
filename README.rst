====
rt1d
====

rt1d is a 1D radiative transfer code developed for the purpose of studying 
ionization (hydrogen and helium) and thermal evolution of gas in the vicinity 
of stars, accreting black holes, or really any source of ultraviolet and/or 
X-ray photons you can think of. A paper including some discussion of its 
inner-workings can be found 
`here <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.

Be warned: this code is still under active development -- use at your own risk! 
Correctness of results is not guaranteed.

Getting started
---------------
To clone a copy and install: ::

    hg clone https://bitbucket.org/mirochaj/rt1d rt1d
    cd rt1d
    python setup.py install

or visit the 'Downloads' page for a tarball.


Dependencies
------------

Currently, rt1d depends on h5py, numpy, and scipy.  The built-in analysis 
modules also use on matplotlib, though if you'd rather make your plots with 
something else, this dependence is not necessary.

Dependencies (Optional)
-----------------------
If you have the Python 
`progressbar <https://code.google.com/p/python-progressbar>`_ installed, rt1d 
will use it. Don't be alarmed if the time-to-completion estimate you're given 
is absurd at first -- the time-step at the beginning of radiative transfer 
simulations is very small (characteristic ionization timescale very
short).  The example problem given below should run in about a minute on a 
single CPU.

Some of the rt1d/test scripts and built-in analysis routines use a module for 
making multi-panel plots, which can be found 
`here <https://bitbucket.org/mirochaj/multiplot>`_.

The code is written such that running with an arbitrary chemical composition 
is possible (in principle) using `dengo <https://bitbucket.org/MatthewTurk/dengo>`_ 
(written by Matthew Turk and Devin Silvia). Its sub-dependencies can all be 
installed using pip: ::

    pip install sympy
    pip install periodic
    pip install ChiantiPy
    
The chianti database itself can be downloaded 
`here <http://www.chiantidatabase.org/download/CHIANTI_7.1_data.tar.gz>`_. To 
make use of it with ChiantiPy, you must define a new environment variable, 
which for me (in bash) looks like: ::

    export XUVTOP=$WORK/mods/chianti

Once this is all done, you should be off to the races.

The dengo stuff is still pretty unstable, though rt1d/tests/test_chemistry_metals.py
should work.

Example
-------

rt1d is meant to be modular. For instance, rather than running a full 
radiative transfer simulation on a grid, you can also run non-equilibrium 
chemistry tests without any radiation at all.

However, there are also modules one can use to avoid writing rt1d scripts. 
They are located in the rt1d/run directory.  In a Python terminal (or script), 
it's as easy as typing:

>>>
>>> import rt1d
>>> sim = rt1d.run.Simulation()
>>> sim.run()
>>>
  
The 'sim' object is an instance of the Simulation class, which contains the 
data as well as instances of all major classes used in the calculation (e.g. 
Grid, Radiation, ChemicalNetwork, etc.). A second command to actually run
the simulation is implemented so that if you like, you can initialize the 
grid, solver, and/or radiation sources without starting a calculation. This
is useful for inspecting the properties of sources, chemical reaction
networks, debugging, etc.

This example (all default parameter values) simulates the expansion of an 
ionization front around a monochromatic source of 13.6 eV photons in an isothermal, 
hydrogen only medium (test #1 from the Radiative Transfer Comparison Project; 
`Iliev et al. 2006 <http://adsabs.harvard.edu/abs/2006MNRAS.371.1057I>`_).

To do some simple analysis of the output, open up a python (or ipython) 
session and use built-in analysis routines, or look at the raw data itself:

>>>
>>> anl = rt1d.analyze.Simulation(sim.checkpoints) 
>>> 
>>> # Some built-in analysis routines
>>> anl.PlotIFrontEvolution()               # Plot the I-front radius vs. time
>>> anl.IonizationProfile(t = [1, 10, 100]) # Plot neutral/ionized fractions vs. radius at 1, 10, 100 Myr
>>> 
>>> # Look at min and max values of the neutral fraction in data dump 50
>>> print sim.data[50]['h_1'].min(), sim.data[50]['h_1'].max()
>>>

More examples on the way.

