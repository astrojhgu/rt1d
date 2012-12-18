"""

RunAllTests.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 29 11:07:49 2012

Description: Run all test problems, and analyze results.

Note: This script will copy everything it needs from $RT1D/doc/tests, and
store the output in your CWD.

"""

import os, re

RT1D = os.environ.get('RT1D')
os.system('cp -r %s/doc/tests/* .' % RT1D)

testdir = os.getcwd()

if not os.path.exists('frames'):
    os.mkdir('frames')

for fn in os.listdir(os.getcwd()):
    if not os.path.isdir(fn):
        continue
        
    os.chdir(fn)
    os.system('cp %s/bin/RT1D.py .' % RT1D)

    # Run all simulations in this dir (given by .dat parameter files)
    for pf in os.listdir(os.getcwd()):
        if not re.search('.dat', pf):
            continue
        
        os.system('python RT1D.py %s' % pf)
        
    # Run analysis
    for py in os.listdir(os.getcwd()):
        if not re.search('.py', py):
            continue
        if py == 'RT1D.py':
            continue    
                        
        os.system('python %s' % py)  
        os.system('cp *.png ../frames')      
    
    os.chdir(testdir)
    
        





