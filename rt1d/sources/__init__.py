from .RadiationSourceIdealized import RadiationSource
#from .RadiationSourceFromFile import RadiationSourceFromFile

def load_spectrum(pf):
    """
    Create attributes we need, normalize, etc.
    """
            
    fn = pf['spectrum_file']    
    tau = self.SourcePars['lifetime'] * self.pf['time_units']
            
    # Read spectrum - expect hdf5 with (at least) E, L_E, and time_yr datasets.
    f = h5py.File(fn)
    pf['spectrum_E'] = f['E'].value
    pf['spectrum_LE'] = f['L_E'].value
    
    self.t = self.Age = f['time_yr'].value * s_per_yr
    self.maxAge = np.max(self.Age)
    self.Nt = len(self.t)
    
    
    self.Emin = np.min(self.E)
    self.Emax = np.max(self.E)
    self.Nfreq = len(self.E)
    
    # Threshold indices
    self.i_Eth = np.zeros(3)
    for i, absorber in enumerate(self.grid.absorbers):
        energy = self.grid.ioniz_thresholds[absorber]
        loc = np.argmin(np.abs(energy - self.E))
        
        if self.E[loc] < energy:
            loc += 1
        
        self.i_Eth[i] = loc
    
    return pf
