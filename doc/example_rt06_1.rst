RT06 Test #1
============================================
Test #1 from the Radiative Transfer Comparison Project.

:: 

    import rt1d
    
    sim = rt1d.run.Simulation(problem_type=1)
    sim.run()
    
    anl = rt1d.analyze.Simulation(sim.checkpoints)
    anl.PlotIonizationFrontEvolution()
