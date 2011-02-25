"""
SolveRateEquations.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-31.

Description: Subset of my homemade odeint routine made special for rt1d.
    
Notes: 
    -The function we supply to the routine 'integrate' must follow the form f(y, x, args),
     where args are any additional parameters required by it.     
    
Coupled ODE Example:
     >>> from odeint import *
     >>> ode = odeint(integrator = 4, stepper = 1, pbar = True)
     >>> def f(y, x, args): return np.array([-y[1], -y[0]])
     >>>     
     >>> x, y = ode.integrate(f, (0, 1), 0, 10.)    # y is a tuple now 
     
     # Analytic solution:
     # y1ofx = -0.5 * np.exp(-x) * (np.exp(2. * x) - 1.)
     # y2ofx = 0.5 * np.exp(-x) * (np.exp(2. * x) - 1.)
         
"""

import numpy as np

class SolveRateEquations:
    def __init__(self, stepper = 1, hmin = 0, hmax = 0.1, rtol = 1e-8, atol = 1e-8, 
        Dfun = None, maxiter = 100):
        """
        This 'odeint' class is the driver for ODE integration via the implicit Euler
        method, which is done below by the 'integrate' routine.  
        
            stepper: 0 (Off), 1 (Step Doubling)
            
            hmin: Minimum time/space step allowed when stepper > 0.
            
            hmax: Maximum time/space step allowed when stepper > 0, otherwise sets size of fixed
                  time/space step.
                  
            rtol: Maximum allowed relative error when using adaptive stepping.
            atol: Maximum allowed absolute error when using adaptive stepping
                *Both are used, limit really set by which one is smaller.
                
        """
        self.stepper = stepper
        self.rtol = rtol
        self.atol = atol
        self.hmax = hmax
        self.hmin = hmin    
        self.maxiter = maxiter    # Max number of iterations for root finding          

        self.solve = self.ImplicitEuler     
        
        # Set adaptive timestepping method
        if self.stepper == 1: self.adapt = self.StepDoubling     
        
    def integrate(self, f, y0, x0, xf, Dfun, *args):
        """
        This routine does all the work.
        
            f: function(y, x, *args) to be solved.  Order of y and x is to be consistent with scipy convention.
            
            x0, y0: Initial conditions on x and y.  y0 may be a tuple if equations are coupled.
            
            xf: Endpoint of integration in independent variable x.
            
        """

        h = self.hmax
        x = [x0]
        if type(y0) is tuple: y = [np.array(y0)]
        else: y = [y0]
                
        # Widget for progressbar.
        if self.pbar: widget = ["odeint: ", Percentage(), ' ', Bar(marker = RotatingMarker()), ' ']        
                
        i = 1
        while x[i - 1] < xf: 
            xnext = x[i - 1] + h
            ynext = self.solve(f, y[i - 1], x[i - 1], h, Dfun, args)
            
            # Update progress bar
            if self.pbar:                    
                try: 
                    pbar = ProgressBar(widgets = widget, maxval = xf).start()
                    pbar.update(xnext)
                except AssertionError: pass
            
            adapted = False                    
                                                                                
            # If adaptive stepping is turned on
            if self.stepper > 0:
                dabs, drel = self.adapt(f, y[i - 1], x[i - 1], ynext, xnext, h, args)
                                                                                               
                # Special treatment if system of ODE's.  Limit set by worst integration in set.
                try:
                    for k, err in enumerate(dabs):
                        if (abs(dabs[k]) > self.atol) or (abs(drel[k]) > self.rtol):
                            h = max(self.hmin, h / 2.)
                            adapted = True
                            break
                
                # If only a single ODE.
                except TypeError:
                    if (abs(dabs) > self.atol) or (abs(drel) > self.rtol): 
                        h = max(self.hmin, h / 2.)
                        adapted = True
            
            # Ensure we end exactly at xf.        
            if xnext > xf: h = (xf - x[i - 1])  
            
            # If we've gotten this far without adaptively stepping, make h = self.hmax once again
            if adapted is False: h = self.hmax
            
            # If we didn't meet our error requirement, repeat loop with different h
            if adapted and h != self.hmin: continue               
            
            x.append(xnext)        
            y.append(ynext)            
            i += 1
        
        # If we're dealing with coupled equations, re-organize return list.    
        if type(y0) is tuple: y = zip(*y)
                            
        return np.array(x), np.array(y)    
           
    def ImplicitEuler(self, f, yi, xi, h, Dfun, args = ()):
        """
        Integrate ODE using backward (implicit) Euler method.  Must apply
        minimization technique separately for each yi, hence the odd array
        manipulation and loop.
        """                
         
        yip1 = []
        for i, element in enumerate(yi):
            newf = lambda y: y - h * f(np.array([y] * len(yi)), xi + h)[i] - yi[i]
            yip1.append(self.Newton(newf, yi[i], Dfun))
                
        rtn = yi + h * f(np.array(yip1), xi + h, args)

        return rtn
        
    def StepDoubling(self, f, yi, xi, yip1, xip1, h, args = ()):    
        """
        Calculate y_n+1 in two ways - first via a single step spanning 2h, and second
        using two steps spanning h each.  The difference gives an estimate of the 
        truncation error, which we can use to adapt our step size in self.integrate.
        """
        
        ynp2_os = self.solve(f, yi, xi, 2. * h, args) # y_n+2 using one step
        ynp2_ts = self.solve(f, yip1, xip1, h, args)  # y_n+2 using two steps
        
        return ynp2_ts - ynp2_os, (ynp2_ts - ynp2_os) / ynp2_ts
        
    def Newton(self, f, x_guess, Dfun, args = ()):
        """
        Find the roots of the function f using the Newton-Raphson method.       
        """    
    
        xnow = x_guess
        xpre = x_guess + self.atol     # Sort of arbitrary
        
        i = 0
        err = 1
        while err > self.atol:
            
            # If the function's derivative is not provided, estimate it.
            fp = None
            if Dfun is not None: fp = Dfun(xnow, args)
            if fp is None: fp = (f(xpre) - f(xnow)) / (xpre - xnow)
            
            # Calculate new estimate of the root
            dx = f(xnow) / fp
            xpre = xnow
            xnow -= dx
                     
            # Calculate deviation between this estimate and last            
            err = abs(xpre - xnow)

            # If we've reached the maximum number of iterations, break
            if i >= self.maxiter: 
                print "Maximum number of iterations reached."
                break
            else: i += 1
            
        return xnow
        
        
        
        
        
        
        
