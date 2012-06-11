"""
SolveRateEquations.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-31.

Description: Subset of my homemade odeint routine made special for rt1d.  
         
"""

import copy
import numpy as np
from .Constants import k_B, m_H
from .Cosmology import Cosmology

class SolveRateEquations:
    def __init__(self, pf):
        """
        This 'odeint' class is the driver for ODE integration via the implicit Euler
        method, which is done below by the 'integrate' routine.  
        
            stepper: 0 (Off), 1 (Step Doubling)
            
            hmin: Minimum time/space step allowed when stepper > 0.
            
            hmax: Maximum time/space step allowed when stepper > 0, otherwise sets size of fixed
                  time/space step.
                  
        """

        self.pf = pf
        self.cosm = Cosmology(pf)
                
        self.AdaptiveODEStep = pf.ODEAdaptiveStep
        self.hmax = pf.ODEMaxStep * pf.TimeUnits
        self.hmin = pf.ODEMinStep * pf.TimeUnits   
        self.maxiter = pf.ODEMaxIter    # Max number of iterations for root finding          

        self.solve = self.ImplicitEuler
        self.rootfinder = self.Newton
        
        self.xmin = pf.MinimumSpeciesFraction # shorthand
        
        # Set adaptive timestepping method
        if self.AdaptiveODEStep == 1: 
            self.stepper = self.StepDoubling     
        
        # Initial guesses for ODE solver
        if pf.CosmologicalExpansion:
            self.guesses = lambda z: np.array([self.cosm.MeanHydrogenNumberDensity(z), 
                                               self.cosm.MeanHeliumNumberDensity(z),
                                               self.cosm.MeanHeliumNumberDensity(z),
                                               3. * self.cosm.Tgas(z) * k_B * self.cosm.MeanBaryonNumberDensity(z) / 2.])
        else:
            if pf.DensityProfile == 0:
                tmp = [pf.DensityUnits * (1. - self.cosm.Y) / m_H, 
                       self.cosm.Y * pf.DensityUnits / 4. / m_H, 
                       self.cosm.Y * pf.DensityUnits / 4. / m_H]
            elif pf.DensityProfile == 1:
                tmp = [self.cosm.MeanHydrogenNumberDensity(pf.InitialRedshift), 
                       self.cosm.MeanHeliumNumberDensity(pf.InitialRedshift),
                       self.cosm.MeanHeliumNumberDensity(pf.InitialRedshift)]
        
            if pf.TemperatureProfile == 0:
                tmp.append(3. * k_B * pf.InitialTemperature * np.sum(tmp[0:2]) / 2.)
            elif pf.TemperatureProfile == 1:
                tmp.append(3. * k_B * self.cosm.Tgas(pf.InitialRedshift) / np.sum(tmp[0:2]) / 2.)    
                
            tmp = np.array(tmp)
            self.guesses = lambda z: tmp
                
    def integrate(self, f, ynow, xnow, xf, znow, zf, Dfun, hpre, *args):
        """
        This routine does all the work.
        
            f: function(y, x, *args) to be solved.
            x0, y0: Initial conditions on x and y.  y0 may be a tuple if equations are coupled.
            xf: Endpoint of integration in independent variable x.
            
            *args = (nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, Heat, zeta, eta, psi)
                        
        """
                
        if hpre is None: 
            h = self.hmax
        else: 
            h = hpre
            
        x0 = xnow    
                                  
        i = 1
        ct = 0
        itr = np.zeros(4)
        while xnow < xf: 
            xnext = xnow + h
                                                                                                                                        
            # Ensure we end exactly at xf.        
            if xnext > xf: 
                h = xf - xnow
                xnext = xf 
                                                                                                            
            # Solve away
            znext = self.cosm.TimeToRedshiftConverter(0, xnext - x0, znow)
            ynext, tmp = self.solve(f, ynow, xnow, h, Dfun, znext, args)
                                                                                                                                    
            # Check for NaNs, reduce timestep or raise exception if h == hmin
            if self.pf.CheckForGoofiness:
                everything_ok = self.SolutionCheck(ynext, znow, znext, args)
                                                                                   
                if not everything_ok[0] and h > self.hmin:
                    h = max(self.hmin, h / 2.)
                    continue
                elif not everything_ok[0] and h == self.hmin:
                    raise ValueError('NAN encountered on minimum ODE step. Exiting.') 
                                  
                # Check to see if number densities are all physically reasonable
                # (i.e. positive, species fractions <= 1)        
                if not np.all(everything_ok[1:]):
                                        
                    ynext, ok = self.ApplyFloor(ynext, znow, znext, args)
                                                                                                                
                    if not np.all(ok):
                        if h > self.hmin:
                            h = max(self.hmin, h / 2.)
                            continue
                        elif h == self.hmin:
                            raise ValueError("xHII or xHeII or xHeIII < 0 or > 1, and we're on the minimum ODE step. Exiting.")
                            
            # Adaptive time-stepping
            if self.stepper:       
                                      
                tol_met, ok = self.stepper(f, ynow, xnow, ynext, xnext, h, Dfun, znow, znext, args)
                                
                if not np.all(tol_met):  
                    if h == self.hmin: 
                        raise ValueError('ERROR: Tolerance not met on minimum ODE step.  Exiting.')
                                                
                    # Make step smaller
                    h = max(self.hmin, h / 2.)
                    continue
                                       
            # If we've gotten this far, increase h 
            h = min(self.hmax, 2. * h)        
                                                                               
            xnow = xnext        
            ynow = ynext            
            i += 1
            itr += tmp
            ct = 0 
                
        return xnow, ynow, h, i, itr
           
    def ImplicitEuler(self, f, yi, xi, h, Dfun, znext, args):
        """
        Integrate ODE using backward (implicit) Euler method.  Must apply
        minimization technique separately for each yi, hence the odd array
        manipulation and loop.
        """                
        
        yip1 = copy.copy(yi)
        itr = [0] * len(yi)
        for i, element in enumerate(yi):

            # If isothermal or Hydrogen only, do not change temperature or helium values
            if (self.pf.MultiSpecies == 0 and (i == 1 or i == 2)) or (self.pf.Isothermal and i == 3):
                yip1[i] = yi[i]
            else:
                newargs = list(args)
                newargs.append(i)
                
                def ynext(y):
                    if i == 0: 
                        return y - h * f([y, yi[1], yi[2], yi[3]], xi + h, newargs)[i] - yi[i]
                    if i == 1: 
                        return y - h * f([yi[0], y, yi[2], yi[3]], xi + h, newargs)[i] - yi[i]
                    if i == 2: 
                        return y - h * f([yi[0], yi[1], y, yi[3]], xi + h, newargs)[i] - yi[i]
                    if i == 3: 
                        return y - h * f([yi[0], yi[1], yi[2], y], xi + h, newargs)[i] - yi[i]
                
                # Guesses = 0 or for example a guess for n_HI > n_H will mess things up                
                if yi[i] <= 0: 
                    guess = 0.4999 * self.guesses(znext)[i]
                elif (yi[i] > self.guesses(znext)[i] and i < 3):
                    guess = 0.4999 * self.guesses(znext)[i]    
                elif i < 3: 
                    guess = 0.4999 * yi[i]
                else:
                    guess = yi[i]

                yip1[i], itr[i] = self.rootfinder(ynext, guess, i)
                                                                                                                              
        rtn = yi + h * f(yip1, xi + h, args)
        if self.pf.MultiSpecies == 0:
            rtn[1] = yip1[1]
            rtn[2] = yip1[2]
        if self.pf.Isothermal:
            rtn[3] = yip1[3]
                            
        return rtn, itr
        
    def SolutionCheck(self, ynext, znow, znext, args):
        """
        Return four-element array representing things that could be wrong with
        our solutions. 
        
        everything_ok = [all_finite, all_positive, nHII <= nH, (nHeII + nHeIII) <= nHe]
        
            Remember, 
            args = (nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, Heat, zeta, eta, psi, xi)
        """    
                
        if self.pf.CosmologicalExpansion:
            nH = self.cosm.nH0 * (1. + znext)**3
            nHe = self.cosm.nHe0 * (1. + znext)**3
        else:            
            nH = args[2]
            nHe = args[3]
            
        nHII = ynext[0]
        nHeII = ynext[1] 
        nHeIII = ynext[2] 
        nHe_ions = nHeII + nHeIII
        
        finite = np.isfinite(ynext)            
        positive = np.greater_equal(ynext, 0.)            
        feasible_H = np.less_equal(nHII, nH)
        feasible_He = np.less_equal(np.sum(nHe_ions), nHe)
        
        everything_ok = [1, 1, 1, 1]          
        if not np.all(finite):
            everything_ok[0] = 0
        if not np.all(positive):
            everything_ok[1] = 0 
        if not np.all(feasible_H):
            everything_ok[2] = 0 
        if not np.all(feasible_He):
            everything_ok[3] = 0 
                 
        return everything_ok                      
                                                        
    def ApplyFloor(self, ynext, znow, znext, args):
        """
        Apply floors in ionization (and potentially, but not implemented) internal energy.
        
        ok = [0 < n_HII <= n_H, 0 < n_HeII <= n_He, 0 < n_HeIII <= n_He, (n_HeII + n_HeIII) <= n_He]
        
        """   
        
        if self.pf.CosmologicalExpansion:
            nH = self.cosm.nH0 * (1. + znext)**3
            nHe = self.cosm.nHe0 * (1. + znext)**3
        else:            
            nH = args[2]
            nHe = args[3]

        nHII = ynext[0] 
        nHeII = ynext[1]
        nHeIII = ynext[2] 
        xHII = nHII / nH
        xHeII = nHeII / nHe
        xHeIII = nHeIII / nHe 
        nHe_ions = nHeII + nHeIII
        
        ok = [1, 1, 1, 1]
                        
        # Hydrogen first - overionization     
        if nHII > nH:
            if np.allclose(xHII, 1, rtol = 0, atol = 1):
                ynext[0] = nH * (1. - self.xmin)
            else:
                ok[0] = 0
        
        # Hitting MinimumSpeciesFraction
        elif np.allclose(xHII, self.xmin, rtol = 0, atol = self.xmin):
            ynext[0] = nH * self.xmin
        
        # Negative    
        elif nHII < 0:
            if np.allclose(xHII, 0, rtol = 0, atol = self.xmin):
                ynext[0] = nH * self.xmin
            else:
                ok[0] = 0
           
        # Helium if necessary
        if self.pf.MultiSpecies:
            
            if nHeII < 0:
                if np.allclose(xHeII, 0, rtol = 0, atol = self.xmin):
                    ynext[1] = nHe * self.xmin
                else:
                    ok[1] = 0
            
            if nHeIII < 0:
                if np.allclose(xHeIII, 0, rtol = 0, atol = self.xmin):
                    ynext[2] = nHe * self.xmin
                else:
                    ok[2] = 0 
                                
            if nHe_ions > nHe:
                if np.allclose(nHe_ions / nHe, 1., rtol = 0, atol = 1):
                    norm = nHe_ions / nHe / (1. - self.xmin)
                    ynext[1] /= norm
                    ynext[2] /= norm                         
                else:
                    ok[3] = 0     
                    
        return ynext, ok
        
    def StepDoubling(self, f, yi, xi, yip1, xip1, h, Dfun, znow, znext, args):    
        """
        Calculate y_n+1 in two ways - first via a single step spanning 2h, and second
        using two steps spanning h each.  The difference gives an estimate of the 
        truncation error, which we can use to adapt our step size in self.integrate.
        """
                
        ynp2_os, tmp = self.solve(f, yi, xi, 2. * h, Dfun, znext, args) # y_n+2 using one step        
        ynp2_ts, tmp = self.solve(f, yip1, xip1, h, Dfun, znext, args)  # y_n+2 using two steps
            
        ynp2_os, ok = self.ApplyFloor(ynp2_os, znow, znext, args)  
                
        err_abs = np.abs(ynp2_ts - ynp2_os)        
        err_tol = 1e-8 + 1e-5 * ynp2_ts

        return np.less_equal(err_abs, err_tol), ok
        
    def Newton(self, f, y_guess, j):
        """
        Find the roots of the function f using the Newton-Raphson method.
        
        Let's remind ourselves what f and y_guess are:
            y_guess is [nHII, nHeII, nHeIII, T]
        
        """    

        ynow = y_guess
        
        if j < 3:
            this_atol = 1e-8
            this_rtol = 0.1 * self.xmin
        else:
            this_atol = 1e-24
            this_rtol = 1e-5
        
        i = 0
        err = 1
        err_tol = 0
        while err > err_tol:
            y1 = ynow
            y2 = max(ynow - 1e-3 * ynow, 0)
            fy1 = f(y1)
            fy2 = f(y2)
            fp = (fy1 - fy2) / (y1 - y2)
                                                                                                                                            
            # Calculate new estimate of the root - fy1 = f(ynow)
            dy = fy1 / fp
            ypre = ynow
            ynow -= dy
                                                                                                                         
            # Calculate deviation between this estimate and last            
            err = abs(ypre - ynow)
            err_tol = this_atol + this_rtol * ynow
                        
            # If we've reached the maximum number of iterations, break
            if i >= self.maxiter: 
                print "Maximum number of iterations (%i) reached." % self.maxiter
                break 
            
            i += 1                       
        
        return ynow, i

        
        
        
