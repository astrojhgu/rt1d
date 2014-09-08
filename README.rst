====
rt1d
====

rt1d is a 1D radiative transfer code developed for the purpose of studying 
ionization (hydrogen and helium) and thermal evolution of gas in the vicinity 
of stars, accreting black holes, or really any source of ultraviolet and/or 
X-ray photons you can think of. A paper including some discussion of its 
inner-workings can be found here: 
`Mirocha et al. (2012) <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.

On its own, it is perhaps most valuable as a testbed for new physics modules and/or numerical
solvers. It can also be used in conjunction with `glorb <https://bitbucket.org/mirochaj/glorb>`_ to study
the ionization and temperature evolution of the IGM during (and preceding) reionization.

Be warned: this code is still under active development -- use at your own risk! 
Correctness of results is not guaranteed.

The `documentation <http://rt1d.readthedocs.org/en/latest/index.html>`_ is still a work in progress.

Getting started
---------------
To clone a copy and install: ::

    hg clone https://bitbucket.org/mirochaj/rt1d rt1d
    cd rt1d
    python setup.py install

Dependencies
------------
Currently, rt1d depends on numpy, h5py, and scipy.

Dependencies (Optional)
-----------------------
If you plan to run with hydrogen and helium, we'll need to be able to 
interpolate in N >= 3 dimensions. Go
`here <https://bitbucket.org/mirochaj/mathutils>`_ to grab the routines I
wrote to do this.

The built-in analysis modules also use on matplotlib, though if you'd rather
make your plots with something else, this dependence is not necessary.

If you have the Python 
`progressbar <https://code.google.com/p/python-progressbar>`_ installed, rt1d 
will use it. Don't be alarmed if the time-to-completion estimate you're given 
is absurd at first -- the time-step at the beginning of radiative transfer 
simulations is very small (characteristic ionization timescale very
short). The example problem given below should run in about a minute on a 
single CPU.

Some of the rt1d/test scripts and built-in analysis routines use a module for 
making multi-panel plots, which can be found 
`here <https://bitbucket.org/mirochaj/multiplot>`_.

If you'd like to find the optimal discrete SED for a given continuous spectrum,
you'll need `ndmin <https://bitbucket.org/mirochaj/ndmin>`_.

Example
-------
rt1d is meant to be modular. Rather than running a full radiative transfer
calculation on a grid, you can also run non-equilibrium chemistry tests
without any radiation at all.

If you'd like to write your own rt1d scripts, it would be useful to have a look
at rt1d/run/Simulate.py, which houses a convenience class (Simulation) that 
initializes a calculation from a parameter file or via keyword arguments:

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
>>> anl.PlotIFrontEvolution()           # Plot the I-front radius vs. time
>>> anl.IonizationProfile(t=[1,10,100]) # Plot neutral/ionized fractions vs. radius at 1, 10, 100 Myr
>>> 
>>> # Look at min and max values of the neutral fraction in data dump 50
>>> print sim.checkpoints.data['dd0050']['h_1'].min()
>>> print sim.checkpoints.data['dd0050']['h_1'].max()
>>>

To see what pre-defined problem types are available, have a look at
``rt1d/util/ProblemTypes.py``, or for a list of all available input parameters,
see ``rt1d/util/SetDefaultParameterValues.py``.

More examples on the way.

