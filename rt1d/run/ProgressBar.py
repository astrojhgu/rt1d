"""

ProgressBar.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec 27 16:27:51 2012

Description: Wrapper for Python progressbar.

"""

try:
    import progressbar
    pb = True
except ImportError:
    pb = False
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1    

class ProgressBar:
    def __init__(self, maxval, name='rt1d'):
        self.maxval = maxval
        
        if pb and rank == 0:
            self.widget = ["%s: " % name, progressbar.Percentage(), ' ', \
              progressbar.Bar(marker = progressbar.RotatingMarker()), ' ', \
              progressbar.ETA(), ' ']
    
    def start(self):
        if rank == 0:
            self.pbar = progressbar.ProgressBar(widgets=self.widget, 
                maxval=self.maxval).start()
        
    def update(self, value):
        if hasattr(self, 'pbar'):
            self.pbar.update(value)
        
    def finish(self):
        if hasattr(self, 'pbar'):
            self.pbar.finish()