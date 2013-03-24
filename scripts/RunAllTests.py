"""

RunAllTests.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 29 11:07:49 2012

Description: Run all test problems, and analyze results.

Note: This script will copy everything it needs from $RT1D/tests, and
store the output in your CWD.

"""

import os, re

RT1D = os.environ.get('RT1D')
os.system('cp -r %s/tests/* .' % RT1D)

testdir = os.getcwd()

if not os.path.exists('frames'):
    os.mkdir('frames')

for fn in os.listdir(os.getcwd()):
    if not re.search('.py', fn):
        continue    

    os.system('python %s' % fn)
    os.system('cp *.png frames')      
        
