Getting Started
===============
In this section, we'll demonstrate how all the pieces of a radiative transfer
calculation are initialized: the grid, the chemical network, the radiation
source, and the radiation field itself.

rt1d has convenience functions for initializing all this stuff for you (see the
Simulation class), so by the time you're done with this section, you'll know
exactly what happens every time you invoke: ::

    >>> import rt1d
    >>> sim = rt1d.run.Simulation()
    
Include a figure here showing a cartoon of what happens when rt1d is run.    

The Grid
--------
The grid object is at the core of any calculation. It carries information
about the physical properties of the medium in which we're propagating
radiation, like the chemical species it contains, its initial temperature and
ionization state, and plenty of convenience functions for accessing various
quantities.

To get going, let's initialize a simple grid: ::
    
    >>> import rt1d
    >>>
    >>> # Initialize 64 cell grid, ignore inner 1% (to avoid singularity at r=0)
    >>> grid = rt1d.static.Grid(dims=64, start_radius=0.01)
    >>>
    >>> # Simplest scenario: isothermal, hydrogen only
    >>> grid.set_physics(isothermal=True)
    >>> grid.set_chemistry(Z=1)                 # by atomic number
    >>> grid.set_density(1e-24)                 # in g / cm^3
    >>> grid.set_temperature(1e4)               # in K
    >>> grid.set_ionization(state='neutral')
    >>>
    >>> # The initial conditions are stored in a dictionary called `data' - let's see what's inside
    >>> for key in grid.data:
    >>>     print key, grid.data[key][0]
    >>>

Note properties of the Grid class - they are useful!    
    
The Chemical Network
--------------------
Though the grid contains information about the chemical species to be evolved
in the calculation, it contains no information about *how* they'll be evolved. 
That's where the Chemistry and ChemicalNetwork classes comes into play. The
It creates the evolution 
equations that will be solved, and contains links to routines that calculate
various rate coefficients

    >>> chem = rt1d.Chemistry(grid)

The Radiation Source
--------------------
    
    
Now we have a grid object that contains all we need to start doing some RT.  

Let's initialize a monochromatic source of UV photons, emitted at 
:math:`h\nu = 13.6 \ \text{eV}`, at a rate of 
:math:`\dot{Q} = 5\times 10^{48} \ \text{s}^{-1}`: ::

    >>> rs = rt1d.sources.RadiationSource(spectrum_E=[13.6], spectrum_LE=[1.0], source_qdot=5e48)



    
    
    
    
    
    
    
    
    