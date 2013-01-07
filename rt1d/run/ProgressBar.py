"""

ProgressBar.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec 27 16:27:51 2012

Description: 

"""

try:
    import progressbar
    pb = True
except ImportError:
    pb = False

class ProgressBar:
    def __init__(self, maxval, name = 'rt1d'):
        self.maxval = maxval
        
        if pb:
            self.widget = ["%s: " % name, progressbar.Percentage(), ' ', \
              progressbar.Bar(marker = progressbar.RotatingMarker()), ' ', \
              progressbar.ETA(), ' ']
            self._start()
    
    def _start(self):
        self.pbar = progressbar.ProgressBar(widgets = self.widget, 
            maxval = self.maxval).start()
        
    def update(self, value):
        if hasattr(self, 'pbar'):
            self.pbar.update(value)
        
    def finish(self):
        if hasattr(self, 'pbar'):
            self.pbar.finish()