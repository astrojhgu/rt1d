"""

PrintInfo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jul 17 15:05:13 MDT 2014

Description: 

"""

import numpy as np
import types, os, textwrap
from ..physics.Constants import cm_per_kpc, m_H

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank; size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0; size = 1
 
# FORMATTING   
width = 84
pre = post = '#'*4    
twidth = width - len(pre) - len(post) - 2
#

RT1D = os.environ.get('RT1D')

e_methods = \
{
 0: 'all photo-electron energy -> heat',
 1: 'Shull & vanSteenberg (1985)',
 2: 'Ricotti, Gnedin, & Shull (2002)',
 3: 'Furlanetto & Stoever (2010)'
}
             
rate_srcs = \
{
 'fk94': 'Fukugita & Kawasaki (1994)',
 'chianti': 'Chianti'
}
             
S_methods = \
{
 1: 'Salpha = const. = 1',
 2: 'Chuzhoy, Alvarez, & Shapiro (2005)',
 3: 'Furlanetto & Pritchard (2006)'
}             

def line(s, just='l'):
    """ 
    Take a string, add a prefix and suffix (some number of # symbols).
    
    Optionally justify string, 'c' for 'center', 'l' for 'left', and 'r' for
    'right'. Defaults to left-justified.
    
    """
    if just == 'c':
        return "%s %s %s" % (pre, s.center(twidth), post)
    elif just == 'l':
        return "%s %s %s" % (pre, s.ljust(twidth), post)
    else:
        return "%s %s %s" % (pre, s.rjust(twidth), post)
        
def tabulate(data, rows, cols, cwidth=12):
    """
    Take table, row names, column names, and output nicely.
    """
    
    assert (cwidth % 2 == 0), \
        "Table elements must have an even number of characters."
        
    assert (len(pre) + len(post) + (1 + len(cols)) * cwidth) <= width, \
        "Table wider than maximum allowed width!"
    
    # Initialize empty list of correct length
    hdr = [' ' for i in range(width)]
    hdr[0:len(pre)] = list(pre)
    hdr[-len(post):] = list(post)
    
    hnames = []
    for i, col in enumerate(cols):
        tmp = col.center(cwidth)
        hnames.extend(list(tmp))
            
    start = len(pre) + cwidth + 3
    hdr[start:start + len(hnames)] = hnames
    
    # Convert from list to string        
    hdr_s = ''
    for element in hdr:
        hdr_s += element
        
    print hdr_s

    # Print out data
    for i in range(len(rows)):
    
        d = [' ' for j in range(width)]
        
        d[0:len(pre)] = list(pre)
        d[-len(post):] = list(post)
        
        d[len(pre)+1:len(pre)+1+len(rows[i])] = list(rows[i])
        d[len(pre)+1+cwidth] = ':'

        # Loop over columns
        numbers = ''
        for j in range(len(cols)):
            if type(data[i][j]) is str:
                numbers += data[i][j].center(cwidth)
                continue
            elif type(data[i][j]) is bool:
                numbers += str(int(data[i][j])).center(cwidth)
                continue 
            numbers += ('%.4e' % data[i][j]).center(cwidth)
        numbers += ' '

        c = len(pre) + 1 + cwidth + 2
        d[c:c+len(numbers)] = list(numbers)
        
        d_s = ''
        for element in d:
            d_s += element
    
        print d_s
        
def print_warning(s, headerd='WARNING'):
    dedented_s = textwrap.dedent(s).strip()
    snew = textwrap.fill(dedented_s, width=twidth)
    snew_by_line = snew.split('\n')
    
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width
    
    for l in snew_by_line:
        print line(l)
    
    print "#"*width        

def print_sim(sim):

    if rank > 0:
        return

    warnings = []

    header = 'Initializer: Radiative Transfer Simulation'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width
    
    print line('-'*twidth)       
    print line('Book-Keeping')     
    print line('-'*twidth)
    
    if sim.pf['dtDataDump'] is not None:
        print line("dtDataDump  : every %i Myr" % sim.pf['dtDataDump'])
    else:
        print line("dtDataDump  : no regularly-spaced time dumps")
    
    if sim.pf['dzDataDump'] is not None:
        print line("dzDataDump  : every dz=%.2g" % sim.pf['dzDataDump'])
    else:
        print line("dzDataDump  : no regularly-spaced redshift dumps")    
       
    print line("initial dt  : %.2g Myr" % sim.pf['initial_timestep'])        
           
    rdt = ""
    for element in sim.pf['restricted_timestep']:
        rdt += '%s, ' % element
    rdt = rdt.strip().rstrip(',')       
    print line("restrict dt : %s" % rdt)
    print line("max change  : %.4g%% per time-step" % \
        (sim.pf['epsilon_dt'] * 100))

    print line('-'*twidth)       
    print line('Grid')     
    print line('-'*twidth)
    
    print line("cells       : %i" % sim.pf['grid_cells'], just='l')
    print line("logarithmic : %i" % sim.pf['logarithmic_grid'], just='l')
    print line("r0          : %.3g (code units)" % sim.pf['start_radius'], 
        just='l')
    print line("size        : %.3g (kpc)" \
        % (sim.pf['length_units'] / cm_per_kpc), just='l')
    print line("density     : %.2e (g cm**-3 / m_H)" % (sim.pf['density_units'] / m_H))
    
    print line('-'*twidth)       
    print line('Chemical Network')     
    print line('-'*twidth)
    
    Z = ''
    A = ''
    for i, element in enumerate(sim.pf['Z']):
        if element == 1:
            Z += 'H'
            A += '%.2g' % (sim.pf['abundances'][i])
        elif element == 2:
            Z += ', He'
            A += ', %.2g' % (sim.pf['abundances'][i])
            
    print line("elements    : %s" % Z, just='l')
    print line("abundances  : %s" % A, just='l')
    print line("rates       : %s" % rate_srcs[sim.pf['rate_source']], 
        just='l')
    
    print line('-'*twidth)       
    print line('Physics')     
    print line('-'*twidth)
    
    print line("radiation   : %i" % sim.pf['radiative_transfer'])
    print line("isothermal  : %i" % sim.pf['isothermal'], just='l')
    print line("expansion   : %i" % sim.pf['expansion'], just='l')
    if sim.pf['radiative_transfer']:
        print line("phot. cons. : %i" % sim.pf['photon_conserving'])
        print line("planar      : %s" % sim.pf['plane_parallel'], 
            just='l')        
    print line("electrons   : %s" % e_methods[sim.pf['secondary_ionization']], 
        just='l')
            
    # Should really loop over sources here        
    
    if sim.pf['radiative_transfer']:
    
        print line('-'*twidth)       
        print line('Source')     
        print line('-'*twidth)        
        
        print line("type        : %s" % sim.pf['source_type'])
        if sim.pf['source_type'] == 'star':
            print line("T_surf      : %.2e K" % sim.pf['source_temperature'])
            print line("Qdot        : %.2e photons / sec" % sim.pf['source_qdot'])
        
        print line('-'*twidth)       
        print line('Spectrum')     
        print line('-'*twidth)
        print line('not yet implemented')


        #if sim.pf['spectrum_E'] is not None:
        #    tabulate()
        

    print "#"*width
    print ""

