"""

Extras.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jun  3 14:02:33 2012

Description: 

"""

class dotdictify(dict):
    """
    Class taken from:
    
    http://stackoverflow.com/questions/3031219/
    python-recursively-access-dict-via-attributes-as-well-as-index-access
    
    Allows 'dot' access to dictionary keys.

    """
    
    marker = object()
    def __init__(self, value = None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError, 'expected dict'

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, dotdictify):
            value = dotdictify(value)
        dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        found = self.get(key, dotdictify.marker)
        if found is dotdictify.marker:
            found = dotdictify()
            dict.__setitem__(self, key, found)
        return found

    __setattr__ = __setitem__
    __getattr__ = __getitem__        

