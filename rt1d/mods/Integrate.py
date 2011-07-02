"""
Integrate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-08-25.

Description: Compute the value of input function (integrand) via numerous methods.
     
"""

def Romberg(function, a, b, eps = 1e-8):
    """
    Approximate the definite integral of 'function' from a to b using Romberg's method.
    eps is the desired accuracy.
    """
    
    R = [[0.5 * (b - a) * (function(a) + function(b))]]  # R[0][0]
    n = 1
    while True:
        h = float(b - a) / 2**n
        R.append([None] * (n + 1))  # Add an empty row.
        # for proper limits
        R[n][0] = 0.5*R[n-1][0] + h*sum(function(a+(2*k-1)*h) for k in xrange(1, 2**(n-1)+1))
        for m in xrange(1, n+1):
            R[n][m] = R[n][m-1] + (R[n][m-1] - R[n-1][m-1]) / (4 ** m - 1)
        if abs(R[n][n-1] - R[n][n]) < eps:
            return R[n][n]
        n += 1
        