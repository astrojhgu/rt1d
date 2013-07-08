Built-In Analysis Tools
=======================
rt1d has a built-in analysis module which contains routines for performing standard
operations like tracking ionization fronts, making radial profiles of ionization,
temperature, etc. To use it, do: ::

    >>> import rt1d
    >>> sim = rt1d.run.Simulation()
    >>> sim.run()
    >>>
    >>> # ...once simulation is complete
    >>> anl = rt1d.analyze.Simulation(sim.checkpoints)
    >>> anl.TemperatureProfile(t[10, 20])
    
or, if you'd like to examine the properties of a radiation source, type instead: ::

    >>> src = rt1d.analyze.Source(sim.rs)
    >>> src.PlotSpectrum()
    

