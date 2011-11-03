"""
Integrate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-09-07.

Description: Integration routines.
     
"""

import numpy as np

# Simpson's rule (adaptive)
def simpsons_rule(f, xmin, xmax):
    return (xmax - xmin) * (f(xmin) + 4. * f((xmin + xmax) / 2.) + f(xmax)) / 6. 
    
def simpson_recursion(f, xmin, xmax, total, tol = 1e-8):
    midpt = (xmin + xmax) / 2.
    left = simpsons_rule(f, xmin, midpt)
    right = simpsons_rule(f, midpt, xmax)
    
    if abs(left + right - total) <= 15. * tol:
        return left + right + (left + right - total) / 15.
    
    return simpson_recursion(f, xmin, midpt, left, tol / 2.) + simpson_recursion(f, midpt, xmax, right, tol / 2.)
    
def simpson(f, xmin, xmax, epsrel = 1e-8):
    """
    Integrate function f from xmin to xmax using Simpson's rule adaptively.
    """
    return (simpson_recursion(f, xmin, xmax, simpsons_rule(f, xmin, xmax), tol = epsrel), None)

