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

e_methods = {0: 'all photo-electron energy -> heat',
             1: 'Shull & vanSteenberg (1985)',
             2: 'Ricotti, Gnedin, & Shull (2002)',
             3: 'Furlanetto & Stoever (2010)'}
             
rate_srcs = {'fk94': 'Fukugita & Kawasaki (1994)',
             'chianti': 'Chianti'}

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
        
def print_warning(s):
    dedented_s = textwrap.dedent(s).strip()
    snew = textwrap.fill(dedented_s, width=twidth)
    snew_by_line = snew.split('\n')
    
    header = 'WARNING'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width
    
    for l in snew_by_line:
        print line(l)
    
    print "#"*width        

    def print_sim(sim):
        """
        Print information about 21-cm simulation to screen.

        Parameters
        ----------
        sim : instance of Simulation class

        """

        if rank > 0 or not sim.pf['verbose']:
            return

        warnings = []

        header = 'Initializer: 21-cm Simulation'
        print "\n" + "#"*width
        print "%s %s %s" % (pre, header.center(twidth), post)
        print "#"*width

        print line('-'*twidth)
        print line('Book-Keeping')
        print line('-'*twidth)

        print line("z_initial   : %.1i" % sim.pf['initial_redshift'])
        print line("first-light : z=%.1i" % sim.pf['first_light_redshift'])
        if sim.pf['stop'] is not None:
            print line("z_final     : @ turning point %s " % sim.pf['stop'])
        else:
            if sim.pf['stop_xavg'] is not None:    
                print line("z_final     : when x_i > %.6g OR" % sim.pf['stop_xavg'])

            print line("z_final     : %.2g" % sim.pf['final_redshift'])

        if sim.pf['dtDataDump'] is not None:
            print line("dtDataDump  : every %i Myr" % sim.pf['dtDataDump'])
        else:
            print line("dtDataDump  : no regularly-spaced time dumps")

        if sim.pf['dzDataDump'] is not None:
            print line("dzDataDump  : every dz=%.2g" % sim.pf['dzDataDump'], just='l')
        else:
            print line("dzDataDump  : no regularly-spaced redshift dumps", just='l')    

        if sim.pf['max_dt'] is not None:  
            print line("max_dt      : %.2g Myr" % sim.pf['max_dt'], just='l')
        else:
            print line("max_dt      : no maximum time-step", just='l')

        if sim.pf['max_dz'] is not None:  
            print line("max_dz      : %.2g" % sim.pf['max_dz'], just='l')
        else:
            print line("max_dz      : no maximum redshift-step", just='l') 

        print line("initial dt  : %.2g Myr" % sim.pf['initial_timestep'], just='l')        

        rdt = ""
        for element in sim.pf['restricted_timestep']:
            rdt += '%s, ' % element
        rdt = rdt.strip().rstrip(',')       
        print line("restrict dt : %s" % rdt, just='l')
        print line("max change  : %.4g%% per time-step" % \
            (sim.pf['epsilon_dt'] * 100), just='l')

        ##
        # ICs
        ##
        if GLORB and hasattr(sim, 'inits_path'):

            print line('-'*twidth)
            print line('Initial Conditions')
            print line('-'*twidth)

            fn = sim.inits_path[sim.inits_path.rfind('/')+1:]
            path = sim.inits_path[:sim.inits_path.rfind('/')+1]

            print line("file        : %s" % fn, just='l')

            if GLORB in path:
                path = path.replace(GLORB, '')
                print line("path        : $GLORB%s" % path, just='l')
            else:
                print line("path        : %s" % path, just='l')

            if sim.pf['initial_redshift'] > sim.pf['first_light_redshift']:
                print line("FYI         : Can set initial_redshift=first_light_redshift for speed-up.", 
                    just='l')

        ##
        # PHYSICS
        ##        

        print line('-'*twidth)
        print line('Physics')
        print line('-'*twidth)

        print line("radiation   : %i" % sim.pf['radiative_transfer'], just='l')
        print line("electrons   : %s" % e_methods[sim.pf['secondary_ionization']], 
            just='l')
        if type(sim.pf['clumping_factor']) is types.FunctionType:
            print line("clumping    : parameterized", just='l')
        else:  
            print line("clumping    : C = const. = %i" % sim.pf['clumping_factor'], just='l')

        if type(sim.pf['feedback']) in [int, bool]:
            print line("feedback    : %i" % sim.pf['feedback'], just='l')
        else:
            print line("feedback    : %i" % sum(sim.pf['feedback']), just='l')

        print line("approx He   : %i" % sim.pf['approx_helium'], just='l')
        print line("approx Sa   : %s" % S_methods[sim.pf['approx_Salpha']], 
            just='l')

        print "#"*width

        if not GLORB:
            warnings.append(hmf_no_tab)
        elif not os.path.exists('%s/input/hmf' % GLORB):
            warnings.append(hmf_no_tab)

        for warning in warnings:
            print_warning(warning)       

def print_sim(sim):

    if rank > 0:
        return

    warnings = []

    header = 'Initializer: Radiative Transfer Simulation'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

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
            A += '%.2g' % (sim.pf['abundance'][i])
        elif element == 2:
            Z += ', He'
            A += ', %.2g' % (sim.pf['abundance'][i])
            
    print line("elements    : %s" % Z, just='l')
    print line("abundance   : %s" % A, just='l')
    print line("rates       : %s" % rate_srcs[sim.pf['rate_source']], 
        just='l')
    
    print line('-'*twidth)       
    print line('Physics')     
    print line('-'*twidth)
    
    print line("radiation   : %i" % sim.pf['radiative_transfer'])
    print line("isothermal  : %i" % sim.pf['isothermal'], just='l')
    if sim.pf['radiative_transfer']:
        print line("phot. cons. : %i" % sim.pf['photon_conserving'])
        print line("planar      : %s" % sim.pf['plane_parallel'], 
            just='l')        
    print line("electrons   : %s" % e_methods[sim.pf['secondary_ionization']], 
        just='l')
            
    # Should really loop over sources here        
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


    #if sim.pf['spectrum_E'] is not None:
    #    tabulate()
        

    print "#"*width
    print ""

