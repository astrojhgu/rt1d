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
    widget = ["rt1d: ", progressbar.Percentage(), ' ', \
              progressbar.Bar(marker = progressbar.RotatingMarker()), ' ', \
              progressbar.ETA(), ' ']
except ImportError:
    pb = False

class ProgressBar:
    def __init__(self, maxval):
        self.maxval = maxval
        
        if pb:
            self._start()
    
    def _start(self):
        self.pbar = progressbar.ProgressBar(widgets = widget, 
            maxval = self.maxval).start()
        
    def update(self, value):
        if hasattr(self, 'pbar'):
            self.pbar.update(value)
        
    def finish(self):
        if hasattr(self, 'pbar'):
            self.pbar.finish()