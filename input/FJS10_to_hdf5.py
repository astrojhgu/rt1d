"""
FJS10_to_hdf5.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-09-20.

Description: Read in the Furlanetto & Stoever 2010 results, and make new tables 
that will be more suitable for Enzo and rt1d.

Notes: Run this script inside of whatever directory you download the Furlanetto 
& Stoever 2010 results in.  It will produce files called secondary_electron*.dat, 
and one file called secondary_electron_data.h5.  This last one is the most important,
and will be used when SecondaryIonization = 3.
     
"""

import h5py
import numpy as np

E_th = [13.6, 24.6, 54.4]

def readtab(fn, start = 0, stop = None, header = False):
    """
    Returns columns from ASCII file as lists. 
    """
    
    f = open(fn, 'r')
    
    data = []
    for i, line in enumerate(f):
        
        # Ignore blank
        if not line.strip(): 
            continue
        
        # Possibly read in header
        if line.split()[0][0] == '#': 
            if not header: 
                hdr = None            
            else: 
                hdr = line.split()[1:]
            
            continue
        
        # Only read between start and stop
        if (i + 1) < start: 
            continue
        if stop is not None:
            if (i + 1) > stop: 
                continue
                                    
        data.append(line.split())
        
        for j, element in enumerate(data[-1]):
            try: 
                data[-1][j] = float(element)
            except ValueError: 
                data[-1][j] = str(element)
    
    f.close()
    
    out = zip(*data)
    
    if header: 
        out.append(hdr)
    
    # Accommodate 1 column files
    if len(out) == 1: 
        return out[0]
    else: 
        return out


x = np.array([1.0e-4, 2.318e-4, 4.677e-4, 1.0e-3, 2.318e-3, 
              4.677e-3, 1.0e-2, 2.318e-2, 4.677e-2, 1.0e-1, 
              0.5, 0.9, 0.99, 0.999])

files = ['xi_0.999.dat', 'xi_0.990.dat', 'xi_0.900.dat', 'xi_0.500.dat', 
         'log_xi_-1.0.dat', 'log_xi_-1.3.dat', 'log_xi_-1.6.dat',
         'log_xi_-2.0.dat', 'log_xi_-2.3.dat', 'log_xi_-2.6.dat',
         'log_xi_-3.0.dat', 'log_xi_-3.3.dat', 'log_xi_-3.6.dat',
         'log_xi_-4.0.dat']

files.reverse()
         
energies = np.zeros(258)
heat = np.zeros([len(files), 258])
fion = np.zeros_like(heat)
fexc = np.zeros_like(heat)
fLya = np.zeros_like(heat)
fHI = np.zeros_like(heat)
fHeI = np.zeros_like(heat)
fHeII = np.zeros_like(heat)

# Read in energy and fractional heat deposition for each ionized fraction.
for i, f in enumerate(files):
    PhotonEnergy, f_ion, f_heat, f_exc, n_Lya, n_ionHI, n_ionHeI, n_ionHeII, shull_heat = \
        readtab('x_int_tables/%s' % f)
       
    # I think PhotonEnergy = ElectronEnergy...Steve?   
       
    if i == 0:          
        for j, energy in enumerate(PhotonEnergy):
            energies[j] = energy
    
    for j, h in enumerate(f_heat):
        heat[i][j] = h
        fion[i][j] = f_ion[j]
        fexc[i][j] = f_exc[j]
        fLya[i][j] = (n_Lya[j] * 10.2) / energies[j]
        fHI[i][j] = (n_ionHI[j] * E_th[0]) / energies[j]
        fHeI[i][j] = (n_ionHeI[j] * E_th[1]) / energies[j]
        fHeII[i][j] = (n_ionHeII[j] * E_th[2]) / energies[j]
 
# We also want the heating as a function of ionized fraction for each photon energy.        
heat_xi = zip(*heat)
fion_xi = zip(*fion)
fexc_xi = zip(*fexc)
fLya_xi = zip(*fLya)
fHI_xi = zip(*fHI)
fHeI_xi = zip(*fHeI)
fHeII_xi = zip(*fHeII)

# Make the newly formatted text files.
f1 = open('secondary_electron_energies.dat', "w")
f2 = open('secondary_electron_heat.dat', "w")
f3 = open('secondary_electron_fHI.dat', "w")
f4 = open('secondary_electron_fHeI.dat', "w")
f5 = open('secondary_electron_fHeII.dat', "w")
f6 = open('secondary_electron_fLya.dat', "w")
f7 = open('secondary_electron_fion.dat', "w")
f8 = open('secondary_electron_fexc.dat', "w")


print >> f1, "# Column " + str(1) + ": Photon Energy (eV)"
print >> f1, ""

for i in range(14):
    print >> f2, "# Column " + str(i) + ": xi = %e" % x[i]
    print >> f3, "# Column " + str(i) + ": xi = %e" % x[i]
    print >> f4, "# Column " + str(i) + ": xi = %e" % x[i]
    print >> f5, "# Column " + str(i) + ": xi = %e" % x[i]
    print >> f6, "# Column " + str(i) + ": xi = %e" % x[i]
    print >> f7, "# Column " + str(i) + ": xi = %e" % x[i]
    print >> f8, "# Column " + str(i) + ": xi = %e" % x[i]

print >> f2, ""
print >> f3, ""
print >> f4, ""
print >> f5, ""
print >> f6, ""
print >> f7, ""
print >> f8, ""

for i, energy in enumerate(energies):
    print >> f1, energy 
                
    print >> f2, "%e %e %e %e %e %e %e %e %e %e %e %e %e %e" % heat_xi[i]
    print >> f3, "%e %e %e %e %e %e %e %e %e %e %e %e %e %e" % fHI_xi[i]
    print >> f4, "%e %e %e %e %e %e %e %e %e %e %e %e %e %e" % fHeI_xi[i]
    print >> f5, "%e %e %e %e %e %e %e %e %e %e %e %e %e %e" % fHeII_xi[i]
    print >> f6, "%e %e %e %e %e %e %e %e %e %e %e %e %e %e" % fLya_xi[i]
    print >> f7, "%e %e %e %e %e %e %e %e %e %e %e %e %e %e" % fion_xi[i]
    print >> f8, "%e %e %e %e %e %e %e %e %e %e %e %e %e %e" % fexc_xi[i]
           
for i in np.arange(1, 9):
    exec('f%i.close()' % i)           
                
# Make HDF5 file as well (containing everything)
f = h5py.File('secondary_electron_data.h5', 'w')

f.create_dataset('electron_energy', data = energies)
f.create_dataset('ionized_fraction', data = np.array(x))
f.create_dataset('f_heat', data = heat_xi)
f.create_dataset('fion_HI', data = fHI_xi)
f.create_dataset('fion_HeI', data = fHeI_xi)
f.create_dataset('fion_HeII', data = fHeII_xi)
f.create_dataset('f_Lya', data = fLya_xi)
f.create_dataset('fion', data = fion_xi)
f.create_dataset('fexc', data = fexc_xi)


f.close()

    

