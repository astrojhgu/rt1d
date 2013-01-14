"""

WriteData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec 31 14:57:19 2012

Description: 

"""

import h5py, os
import numpy as np
from ..physics.Cosmology import Cosmology
from ..physics.Constants import s_per_myr

try:
    import h5py
    h5 = True
except ImportError:
    print 'Module h5py not found. Will read/write to ASCII instead of HDF5.'
    h5 = False
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    print "Module mpi4py not found.  No worries, we'll just run in serial."
    rank = 0
    size = 1

GlobalDir = os.getcwd()

class CheckPoints:
    def __init__(self, pf = None, grid = None, dtDataDump = 5., 
        time_units = s_per_myr):
        
        self.pf = pf
        self.data = {}
        self.grid = grid
        self.dtDD = dtDataDump * time_units
        self.time_units = time_units
        
        self.basename = 'dd'
        self.fill = 4
        
        if self.grid is not None:
            self.store_ics(grid.data)
        
    def store_ics(self, data):
        nothing = self.update(data, 0., 1)
    
    def store_kwargs(self, t, kwargs):
        if not self.write_now(t):    
            return
        
        dd = int(self.dd(t))
        for kwarg in kwargs:
            self.data[dd][kwarg] = kwargs[kwarg]
        
    def update(self, data, t, dt):
        """
        Store data or don't.  If (t + dt) passes our next checkpoint,
        return new dt.
        """
        
        if self.write_now(t):
            tmp = data.copy()
            tmp.update({'time': t})
            self.data[int(self.dd(t))] = tmp
            del tmp
            
        return self.new_dt(t, dt)
        
    def write_now(self, t):
        if t % self.dtDD == 0:
            return True
        
        return False    
        
    def new_dt(self, t, dt):   
        last_dd = int(self.dd(t))
        next_dd = last_dd + 1
         
        # If dt won't take us all the way to the next DD, don't modify dt
        if self.dd(t + dt) <= next_dd:
            return dt
            
        return next_dd * self.dtDD - t
        
    def dd(self, t):
        return t / self.dtDD     
            
    def name(self, t):
        ct = int(self.dd(t))
        return '%s%s' % (self.basename, str(ct).zfill(self.fill))
        
    def dump(self, fn):
        """ Write out data to file. """
    
        f = h5py.File(fn, 'w')
        
            
        

class WriteData:
    def __init__(self, grid, fn = 'rt1d.sim', **kwargs):
        self.fn = fn
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
        self.cosm = Cosmology(pf)
        self.OutputFormat = pf['OutputFormat'] and h5
        
    @property
    def f(self):
        """ Open hdf5 file object. """    
        
        if not hasattr(self, 'f_hdf5'):
            if self.OutputFormat == 1:
                self.f_hdf5 = h5py.File('%s.hdf5' % self.fn, 'a')
            else:
                self.f_hdf5 = open('%s.dat' % self.fn, 'a')
        
        return self.f_hdf5    
        
    def WriteAllData(self, data, wct, t, dt):
        """
        Write all data to hdf5 file.
        """
        
        if self.OutputFormat == 0:
            self.WriteASCII(data, wct, t, dt)
        else:
            self.WriteHDF5(data, wct, t, dt)
            
        self.WriteParameterFile(wct, t, dt)    
        
    def WriteHDF5(self, data, wct, t, dt):
        """
        Write all data to hdf5 file.
        """
                                        
        f = h5py.File("{0}/{1}/{2}.h5".format(GlobalDir, self.pf['OutputDirectory'].rstrip('/'), 
            self.GetDataDumpName(wct)), 'w') 

        pf_grp = f.create_group("parameters")
        data_grp = f.create_group("data")
        
        for par in self.pf.keys(): 
            if par == "CurrentTime": 
                pf_grp.create_dataset(par, data = t / self.pf['TimeUnits'])
            elif par == "CurrentRedshift":
                pf_grp.create_dataset(par, data = self.cosm.TimeToRedshiftConverter(0, t, self.pf['InitialRedshift']))
            elif par == "CurrentTimestep": 
                pf_grp.create_dataset(par, data = dt / self.pf['TimeUnits'])
            else: 
                pf_grp.create_dataset(par, data = self.pf[par])
        
        for field in data: 
            if data[field].shape[0] > 0:
                data_grp.create_dataset(field, data = data[field])
        
        f.close()
        
        if rank == 0: 
            print "\nWrote %s/%s.h5\n" % (self.pf['OutputDirectory'], 
                    self.GetDataDumpName(wct))

    def WriteASCII(self, data, wc, t, dt):
        """
        Write all data to ASCII file.
        """    
        
        raise ValueError('WriteASCII not yet implemented.')
        
    def GetDataDumpName(self, wct):
        """
        Return name of data dump to be written
        """

        return "{0}{1:04d}".format(self.pf['DataDumpName'], wct)

    def WriteParameterFile(self, wct, t, dt):
        """
        Write out parameter file to ASCII format.
        """
                
        f = open("{0}/{1}/{2}".format(GlobalDir, self.pf['OutputDirectory'], 
            self.GetDataDumpName(wct)), 'w')
        
        names = self.pf.keys()
        names.sort()
        
        print >> f, "{0} = {1}".format('ProblemType'.ljust(35, ' '), self.pf['ProblemType'])
        
        for par in names:
            
            # ProblemType must be the first parameter
            if par == 'ProblemType': 
                continue
            
            if par == "CurrentTime": 
                val = t / self.pf['TimeUnits']
            elif par == "CurrentRedshift":
                val = self.cosm.TimeToRedshiftConverter(0, t, self.pf['InitialRedshift'])
            elif par == "CurrentTimestep": 
                val = dt / self.pf['TimeUnits']
            else: 
                val = self.pf[par]
            
            print >> f, "{0} = {1}".format(par.ljust(35, ' '), val)
            
        f.close()    
        
        