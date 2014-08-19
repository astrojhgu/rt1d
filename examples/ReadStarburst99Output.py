"""

ReadStarburst99Output.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jul 20 11:14:22 2012

Description: Parse output files of starburst99 calculation, read in everything
we need for rt1d.

Notes: Supply path to starburst99 parameter file, output file with suffix 'spectrum' 
and name of output (hdf5) file.

"""

import sys, h5py
import numpy as np
from rt1d.mods.Constants import h, c, erg_per_ev

fields = ['time', 'E', 'L_E', 'L_E_stellar', 'L_E_nebular']

# Read parameter file
#f = open(sys.argv[1], 'r')
#pf = {}
#
#for line in f:
#    if line[0] in ['*', '#']:
#        continue
#    if not line.split():
#        continue
#        
#    newline = line.split()
#    if len(newline) == 1:
#        newline = line.split(',')
#    
#    if newline[0].isdigit():
#        try:
#            pf[name] = float(newline[0])
#        except:
#            pf[name] = map(float, newline)            
#        continue  
#        
#    if (newline[-1][0] == '[') and (newline[-1][-1] == ']'):
#        name = newline[-1].strip('[]')
#                        
#f.close()        

# Read data
f = open(sys.argv[1], 'r')

time = []
result = []

tmp = {}
for field in fields:
    tmp[field] = []

for i, line in enumerate(f):
    if i < 6: 
        continue
    
    t, wavelength, log_tot, log_stellar, log_nebular = line.split()
        
    if len(tmp['time']):
        if float(t) != tmp['time'][-1]:
            time.append(tmp['time'][-1])
            
            # Ascending emission energy
            for field in fields:
                tmp[field].reverse()
            
            result.append(tmp)
            
            tmp = {}
            for field in fields:
                tmp[field] = []
                    
    tmp['time'].append(float(t))
    tmp['E'].append(h * c * 1e8 / float(wavelength) / erg_per_ev)
    tmp['L_E'].append(10**float(log_tot))
    tmp['L_E_stellar'].append(10**float(log_stellar))
    tmp['L_E_nebular'].append(10**float(log_nebular))

f.close()

# Prepare output arrays
E = np.array(result[0]['E'])
L_E = np.zeros([len(time), len(E)])
L_E_stellar = np.zeros_like(L_E)
L_E_nebular = np.zeros_like(L_E)

for i, t in enumerate(time):
    L_E[i] = np.array(result[i]['L_E'])
    L_E_stellar[i] = np.array(result[i]['L_E_stellar'])
    L_E_nebular[i] = np.array(result[i]['L_E_nebular'])

# Write out data to hdf5
f = h5py.File(sys.argv[2])
f.create_dataset('time_yr', data = np.array(time))    
f.create_dataset('E', data = E)
f.create_dataset('L_E', data = L_E)
f.create_dataset('L_E_stellar', data = L_E_stellar)
f.create_dataset('L_E_nebular', data = L_E_nebular)        
f.close()
