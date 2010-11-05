"""
Radiate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-08-18.

Description: This routine essentially runs the show.  The method 'EvolvePhotons' is the
driver of rt1d, calling our solvers which call all the various physics modules.
     
"""

from RadiationSource import *
from Interpolate import *
from scipy.integrate import odeint
import copy

m_e = 9.10938188*10**-28 		# Electron mass - [m_e] = g
m_p = 1.67262158*10**-24		# Proton mass - [m_p] = g
m_n = 1.67492729*10**-24        # Neutron mass - [m_n] = g

m_H = m_p + m_e
m_HeI = 2.0 * (m_p + m_n + m_e)
m_HeII = 2.0 * (m_p + m_n) + m_e

tiny_number = 1e-50

SolverList = ["SolveHIRateEquation", "SolveHeIRateEquation", "SolveHeIIRateEquation", "SolveHeatRateEquation"]

class Radiate:
    def __init__(self, pf, itabs, n): 
        self.rs = RadiationSource(pf)
        self.pf = pf
        self.itabs = itabs
        self.InterpolationMethod = pf["InterpolationMethod"]
        self.GridDimensions = pf["GridDimensions"]
        self.LengthUnits = pf["LengthUnits"]
        self.StartRadius = pf["StartRadius"]
        self.dx = self.LengthUnits / self.GridDimensions
        self.grid = np.arange(self.GridDimensions)
        self.HIColumn = n[0]
        self.HeIColumn = n[1]
        self.HeIIColumn = n[2]
        
    def EvolvePhotons(self, data, t, dt):
        """
        This routine calls our solvers, updates the values in 'data', and 
        computes what the next timestep ought to be.
        """
        
        newdata = copy.deepcopy(data)
        
        def HIIRateEquation(n_HII_0, t, n_HI, n_e, Gamma_HI, alpha_HII):
            """
            This is the right side of the HI rate equation (equation 1 in Thomas & Zaroubi 2007).
            It is called for each cell on every timestep.
            """
            
            n_HII = n_HII_0[0]
            
            return Gamma_HI * n_HI - alpha_HII * n_e * n_HII

        # Loop over cells radially, solve rate equations, update values in data
        for cell in self.grid:
            
            if (cell * self.dx < (self.StartRadius * self.LengthUnits)): continue
                        
            # Column densities for absorbers
            ncol_HI = np.sum(data["HIDensity"][0:cell] * self.dx)
            ncol_HeI = np.sum(data["HeIDensity"][0:cell] * self.dx)
            ncol_HeII = np.sum(data["HeIIDensity"][0:cell] * self.dx)
                        
            T = data["Temperature"][cell]
            n_e = data["ElectronDensity"][cell]
            n_HI = data["HIDensity"][cell]
            n_HII = data["HIIDensity"][cell]
            Gamma_HI = self.IonizationRateCoefficientHI(data, cell, t)
            alpha_HII = self.RecombinationRateCoefficientHII(T)
                        
            newHII = odeint(HIIRateEquation, [n_HII, 0], [0, dt], \
                args = (n_HI, n_e, Gamma_HI, alpha_HII,), mxstep = 1000)[1][0]                
                
            if newHII > (n_HI + n_HII):    
                newHII = n_HI + n_HII - tiny_number
                newHI = tiny_number
            
            newHI = (n_HI + n_HII) - newHII
                                                                                                                                                       
            newdata["HIIDensity"][cell] = newHII
            newdata["HIDensity"][cell] = (n_HI + n_HII) - newHII
            newdata["ElectronDensity"][cell] = newHII + newdata["HeIIDensity"][cell] + 2.0 * newdata["HeIIIDensity"][cell]
                        
        dt = dt
        return newdata, dt
        
    def RecombinationRateCoefficientHII(self, T):
        """
        Using approximation of Zaroubi et al. 2007.
        """
        return 2.6e-13 * (T / 1.0e4)**-0.85
        
    def IonizationRateCoefficientHI(self, data, cell, t):
        """
        Returns HI ionization rate coefficient, Gamma_HI.  Currently only has the photoionization term.
        """    
        
        ncol_HI = np.sum(data["HIDensity"][0:cell] * self.dx)
        ncol_HeI = np.sum(data["HeIDensity"][0:cell] * self.dx)
        ncol_HeII = np.sum(data["HeIIDensity"][0:cell] * self.dx)
        r = self.pf["LengthUnits"] * cell / self.pf["GridDimensions"]
                                
        PhotoIonizationTerm = self.rs.BolometricLuminosity(t) * \
            Interpolate3D(self.itabs["HIPhotoIonizationRateIntegral"], [self.HIColumn, self.HeIColumn, self.HeIIColumn], [ncol_HI, ncol_HeI, ncol_HeII], self.InterpolationMethod) / \
            4.0 / np.pi / r**2
                                                                                        
        return PhotoIonizationTerm
        
        
    
        
        
        