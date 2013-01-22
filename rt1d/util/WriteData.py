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
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    print "Module mpi4py not found.  No worries, we'll just run in serial."
    rank = 0
    size = 1

GlobalDir = os.getcwd()

class CheckPoints:
    def __init__(self, pf = None, grid = None, time_units = s_per_myr,
        dtDataDump = 5., logdtDataDump = None, stop_time = 100,
        initial_timestep = None, source_lifetime = None):
        self.pf = pf
        self.data = {}
        self.grid = grid
        self.time_units = time_units
        self.stop_time = stop_time * time_units
        self.source_lifetime = source_lifetime * time_units
        self.initial_timestep = initial_timestep * time_units
        
        self.basename = 'dd'
        self.fill = 4
            
        if dtDataDump is not None:   
            self.DDtimes = np.linspace(0, self.stop_time, 
                stop_time / dtDataDump + 1)
        else:
            self.DDtimes = np.array([0, self.stop_time])

        self.logdtDD = logdtDataDump
        if logdtDataDump is not None:
            self.logti = np.log10(initial_timestep)
            self.logtf = np.log10(stop_time)
            self.logDDt = time_units * np.logspace(self.logti, self.logtf, 
                (self.logtf - self.logti) / self.logdtDD + 1)[0:-1]
                
            self.DDtimes = np.sort(np.concatenate((self.DDtimes, self.logDDt)))
                                
        self.DDtimes = uniquify(self.DDtimes)                        
                                
        self.allDD = np.linspace(0, len(self.DDtimes) - 1., len(self.DDtimes))
        self.NDD = len(self.allDD)                            
        
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
        if t in self.DDtimes:
            return True
            
        return False    
        
    def new_dt(self, t, dt):
        """
        Compute next timestep based on when our next data dump is, and
        when the source turns off (if ever).
        """
        
        last_dd = int(self.dd(t))
        next_dd = last_dd + 1
        
        if t == self.source_lifetime:
            return self.initial_timestep
        
        src_on_now = t < self.source_lifetime
        src_on_next = (t + dt) < self.source_lifetime
                        
        # If dt won't take us all the way to the next DD, don't modify dt
        if self.dd(t + dt) <= next_dd:
            if (src_on_now and src_on_next) or (not src_on_now):
                return dt    
            
        if next_dd <= self.NDD:    
            next_dt = self.DDtimes[next_dd] - t
        else:
            next_dt = self.stop_time - t
        
        src_still_on = (t + next_dt) < self.source_lifetime
            
        if src_on_now and src_still_on or (not src_on_now):
            return next_dt
        
        return self.source_lifetime - t 
                
    def dd(self, t):
        """ What data dump are we at currently? Doesn't have to be integer. """
        return np.interp(t, self.DDtimes, self.allDD, 
            right = self.NDD)

    def name(self, t):
        ct = int(self.dd(t))
        return '%s%s' % (self.basename, str(ct).zfill(self.fill))
        
    def dump(self, fn):
        """ Write out data to file. """
    
        f = h5py.File(fn, 'w')
        
        pf = f.create_group('parameters')
        for key in self.pf:
            pf.create_dataset(key, data = self.pf[key])
        
        for dd in self.data.keys():
            grp = f.create_group('dd%s' % str(dd).zfill(4))
            
            for key in self.data[dd]:
                grp.add_dataset(key, data = self.data[dd][key])
                
            del grp
            
        f.close()        

def uniquify(l):   
    """
    Return a revised version of 'list' containing only unique elements.  
    This routine will preserve the order of the original list.
    """
    
    def ID(x): 
        return x 
    
    seen = {} 
    result = [] 
    for item in l: 
        marker = ID(item) 
        if marker in seen: 
            continue 
        
        seen[marker] = 1 
        result.append(item) 
    
    return np.array(result)
          
        
        