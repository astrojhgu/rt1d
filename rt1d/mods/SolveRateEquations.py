"""
SolveRateEquations.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-31.

Description: Subset of my homemade odeint routine made special for rt1d.  
         
"""

import numpy as np
import pylab as pl

tiny_number = 1e-30

class SolveRateEquations:
    def __init__(self, pf, guesses, stepper = 1, hmin = 0, hmax = 0.1, rtol = 1e-8, atol = 1e-8, 
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

        self.MultiSpecies = pf["MultiSpecies"]
        self.Isothermal = pf["Isothermal"]
        
        self.stepper = stepper
        self.rtol = rtol
        self.atol = atol
        self.hmax = hmax
        self.hmin = hmin    
        self.maxiter = maxiter    # Max number of iterations for root finding          

        self.solve = self.ImplicitEuler

        if pf["RootFinder"] == 0: self.rootfinder = self.Newton
        elif pf["RootFinder"] == 1: self.rootfinder = self.Bisection
        elif pf["RootFinder"] == 2: self.rootfinder = self.FalsePosition
        
        # Set adaptive timestepping method
        if self.stepper == 1: self.adapt = self.StepDoubling     
        
        # Guesses
        self.guesses = np.array(guesses)
                
    def integrate(self, f, y0, x0, xf, Dfun, hpre, *args):
        """
        This routine does all the work.
        
            f: function(y, x, *args) to be solved.  Order of y and x is to be consistent with scipy convention.
            
            x0, y0: Initial conditions on x and y.  y0 may be a tuple if equations are coupled.
            
            xf: Endpoint of integration in independent variable x.
            
        """
        
        if hpre is None: h = self.hmax
        else: h = hpre
        
        x = [x0]
        if type(y0) is tuple: y = [np.array(y0)]
        else: y = [y0]

        i = 1
        while x[i - 1] < xf: 
            xnext = x[i - 1] + h
            ynext = self.solve(f, y[i - 1], x[i - 1], h, Dfun, args)
                                                                                                                                                               
            # If anything is negative or NAN, our timestep is too big.  Reduce it, and repeat step.
            finite = np.isfinite(ynext)
            positive = np.greater_equal(ynext, 0.)
            if not np.all(finite) or not np.all(positive): 
                h = max(self.hmin, h / 2.)
                continue
                                        
            # Adaptive time-stepping
            adapted = False
            if self.stepper > 0 and (h != self.hmin or i == 1):
                drel = self.adapt(f, y[i - 1], x[i - 1], ynext, xnext, h, Dfun, args)

                # Limit determined by worst integration in set of equations.
                for k, err in enumerate(drel):
                    if (err > self.rtol):
                        h = max(self.hmin, h / 2.)
                        adapted = True
                        break
                        
            # If we've gotten this far without adaptively stepping, increase h for the next timestep
            if adapted is False: h = min(self.hmax, 2. * h)
            
            # Ensure we end exactly at xf.        
            if xnext > xf: h = (xf - x[i - 1])  
            
            # If we didn't meet our error requirement, repeat loop with different h
            if adapted and h != self.hmin: continue             
            
            x.append(xnext)        
            y.append(ynext)            
            i += 1
                            
        # If we're dealing with coupled equations, re-organize return list.    
        if type(y0) is tuple: y = zip(*y)
                                                        
        return np.array(x), np.array(y), h  
           
    def ImplicitEuler(self, f, yi, xi, h, Dfun, args):
        """
        Integrate ODE using backward (implicit) Euler method.  Must apply
        minimization technique separately for each yi, hence the odd array
        manipulation and loop.
        """                

        yip1 = []
        for i, element in enumerate(yi):

            # If isothermal or Hydrogen only, do not change temperature or helium values
            if (self.MultiSpecies == 0 and (i == 1 or i == 2)) or (self.Isothermal and i == 3):
                yip1.append(yi[i])

            else:
                newargs = list(args)
                newargs.append(i)
                
                def ynext(y):
                                                            
                    if i == 0: return y - h * f(np.array([y, yi[1], yi[2], yi[3]]), xi + h, newargs)[i] - yi[i]
                    if i == 1: return y - h * f(np.array([yi[0], y, yi[2], yi[3]]), xi + h, newargs)[i] - yi[i]
                    if i == 2: return y - h * f(np.array([yi[0], yi[1], y, yi[3]]), xi + h, newargs)[i] - yi[i]
                    if i == 3: return y - h * f(np.array([yi[0], yi[1], yi[2], y]), xi + h, newargs)[i] - yi[i]
                
                if yi[i] == 0.: guess = self.guesses[i]
                else: guess = yi[i]
                
                yip1.append(self.rootfinder(ynext, guess))
                                              
        rtn = yi + h * f(np.array(yip1), xi + h, args)
        if self.MultiSpecies == 0:
            rtn[1] = yip1[1]
            rtn[2] = yip1[2]
        if self.Isothermal:
            rtn[3] = yip1[3]

        return rtn

    def StepDoubling(self, f, yi, xi, yip1, xip1, h, Dfun, args):    
        """
        Calculate y_n+1 in two ways - first via a single step spanning 2h, and second
        using two steps spanning h each.  The difference gives an estimate of the 
        truncation error, which we can use to adapt our step size in self.integrate.
        """
                
        ynp2_os = self.solve(f, yi, xi, 2. * h, Dfun, args) # y_n+2 using one step        
        ynp2_ts = self.solve(f, yip1, xip1, h, Dfun, args)  # y_n+2 using two steps
                
        err_abs = np.abs(ynp2_ts - ynp2_os)
        err_rel = np.zeros_like(err_abs)
        for i, element in enumerate(ynp2_ts):
            if element > 0: err_rel[i] = err_abs[i] / element
        
        return err_rel
        
    def Newton(self, f, y_guess):
        """
        Find the roots of the function f using the Newton-Raphson method.
        
        Let's remind ourselves what f and y_guess are:
            y_guess is [nHII, nHeII, nHeIII, T], right?
        
        """    

        ynow = y_guess    

        i = 0
        err = 1
        while err > self.rtol:
            y1 = ynow
            y2 = max(ynow - 1e-3 * ynow, 0)
            fp = (f(y1) - f(y2)) / (y1 - y2)
                                                                        
            # Calculate new estimate of the root
            dy = f(ynow) / fp
            ypre = ynow
            ynow -= dy   
                                                                     
            # Calculate deviation between this estimate and last            
            err = abs(ypre - ynow) / ypre
            
            # If we've reached the maximum number of iterations, break
            if i >= self.maxiter: 
                print "Maximum number of iterations reached."
                break
            else: i += 1               
                                                                                                                                                 
        return max(ynow, tiny_number)

    def Bisection(self, f, y_guess):
        """
        Find root of function using bisection method.
        """

        y1, y2 = self.Bracket(f, y_guess)
    
        # Narrow bracketed range with bisection until tolerance is met
        i = 0
        while abs(y2 - y1) > self.atol:
            midpt = np.mean([y1, y2])
            fmid = f(midpt)
    
            if np.sign(fmid) < 0: y1 = midpt
            else: y2 = midpt
            
            if fmid == 0.0: break
            
        return y2
        
    def FalsePosition(self, f, y_guess):
        """
        Find root using false position method.  Should converge faster than bisection.
        Secand method might beat this, but not guaranteed to keep solution bracketed.
        """ 
        
        # Find points that bracket root
        y1, y2 = self.Bracket(f, y_guess)
        
        # Narrow bracketed range with bisection until tolerance is met
        i = 0
        broke = False
        while abs(y2 - y1) > self.atol:
            f1 = f(y1)
            f2 = f(y2)
                
            midpt = np.interp(0, [f1, f2], [y1, y2])
            fmid = f(midpt)
            
            if np.sign(fmid) < 0: y1 = midpt
            else: y2 = midpt
                        
            if (y1 == midpt) or (y2 == midpt): 
                broke = True
                break
                
            if i >= self.maxiter: 
                print "Maximum number of iterations reached."
                break
            else: i += 1    
                                
        if broke == True: y2 = self.Bisection(f, y_guess)
            
        return y2   # Don't want the negative function value (in general)
        
    def Bracket(self, f, y_guess):
        """
        Bracket root by finding points where function goes from positive to negative.
        """
        
        f1 = f(y_guess)
        f2 = f(y_guess + 0.01 * y_guess)
        df = f2 - f1
        
        # Determine whether increasing or decreasing y_guess will lead us to zero
        if (f1 > 0 and df < 0) or (f1 < 0 and df > 0): sign = 1
        else: sign = -1
        
        # Find root bracketing points
        ypre = y_guess
        ynow = y_guess + sign * 0.01 * y_guess
        fpre = f1
        fnow = f(ynow)
        while (np.sign(fnow) == np.sign(fpre)):
            ypre = ynow
            ynow += sign * 0.1 * ynow
            fpre = f(ypre)
            fnow = f(ynow)
                    
        y1 = min(ynow, ypre)
        y2 = max(ynow, ypre)
        
        if not np.all([np.sign(fpre), np.sign(fnow)]): 
            y1 -= self.atol
            y2 += self.atol
                                
        return y1, y2
    
        
        
        
