"""
FieldList.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-18.

Description: Container for all rt1d fields.  These should be accompanied by methods in the InitializeGrid
class, with naming convention 'Initialize{0}'.format(FieldName), and appropriate methods in the solving
routines (which have not yet been written).
     
"""

FieldList = \
    ["Density", "Temperature", "HIDensity", "HIIDensity", "HeIDensity", \
     "HeIIDensity", "HeIIIDensity"]