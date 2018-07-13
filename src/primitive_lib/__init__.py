"""
A module for importing, validating, enumerating, running etc. primitives.
"""

import d3m.index
import d3m.metadata.base as metadata

def list_primitives():
    """
    Returns a list of all primitives, as PrimitiveClass[].
    """
    primcs = []
    prims = d3m.index.search()
        
    for pc in prims:
        try:
            primitive_obj = d3m.index.get_primitive(pc)
        except:
            continue

        if hasattr(primitive_obj, 'metadata'):
            try:
                primitive = PrimitiveClass(primitive_obj.metadata, primitive_obj)
            except:
                continue
            primcs.append(primitive)

    return primcs
 
class PrimitiveClass(object):
    """
    A loader that mainly just contains primitive metadata.
    Eventually it should be able to do more useful things too.
    """
    def __init__(self, metadata, classname):
        self.id = metadata.query()['id']
        self.name = metadata.query()['name']
        self.version = metadata.query()['version']
        self.python_path = metadata.query()['python_path'] 
        self.classname = classname
        self.family = metadata.query()['primitive_family']
        self.digest = metadata.query()['digest']

        self.arguments=[]
        args = metadata.query()['primitive_code']['arguments']
        for name,value in args.items():
            if value['kind'] == "PIPELINE":
                self.arguments.append(name)

        self.produce_methods=[]
        args = metadata.query()['primitive_code']['instance_methods']
        for name,value in args.items():
            if value['kind'] == "PRODUCE":
                self.produce_methods.append(name)
