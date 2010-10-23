""" 

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2009-09-01.

Description: Contains various constants that may be of use.

Notes: 
      -All units are cgs unless stated otherwise.

"""

from math import *

### General Physics
h = 6.626068*10**-27 			# Planck's constant - [h] = erg*s
h_bar = h/(2*pi) 				# H-bar - [h_bar] = erg*s
c = 29979245800.0 				# Speed of light - [c] = cm/s
k_B = 1.3806503*10**-16			# Boltzmann's constant - [k_B] = erg/K
G = 6.673*10**-8 				# Gravitational constant - [G] = cm^3/g/s^2
e = 1.60217646*10**-19 			# Electron charge - [e] = C
m_e = 9.10938188*10**-28 		# Electron mass - [m_e] = g
m_p = 1.67262158*10**-24		# Proton mass - [m_p] = g
sigma_T = 6.65*10**-25			# Cross section for Thomson scattering - [sigma_T] = cm^2
alpha = 1/137.035999070 		# Fine structure constant - unitless

### Conversions
km_per_pc = 3.08568*10**13
km_per_mpc = km_per_pc*10**6
km_per_gpc = km_per_mpc*10**3
cm_per_pc = km_per_pc*10**5
cm_per_kpc = cm_per_pc*10**3
cm_per_mpc = cm_per_pc*10**6
cm_per_gpc = cm_per_mpc*10**3
cm_per_km = 1.0*10**5
g_per_msun = 1.98892*10**33
s_per_yr = 365.25*24*3600
s_per_myr = s_per_yr*10**6
s_per_gyr = s_per_myr*10**3
sqdeg_per_std = (180.0**2)/(pi**2)
erg_per_j = 10.0**-7
erg_per_ev = e/erg_per_j
erg_per_kev = erg_per_ev*1000

### Hydrogen Specific
A10 = 2.85*10**-15 				# HI 21cm spontaneous emission coefficient - [A10] = Hz
E10 = 5.9*10**-6 				# Energy difference between hyperfine states - [E10] = eV
m_H = 1.674*10**-24 			# Mass of a hydrogen atom - [m_H] = g
nu_0 = 1420.4057*10**6 			# Rest frequency of HI 21cm line - [nu_0] = Hz
T_star = 0.068 					# Corresponding temperature difference between HI hyperfine states - [T_star] = K
a_0 = 5.292*10**-9 				# Bohr radius - [a_0] = cm

