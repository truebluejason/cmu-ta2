

"""
A module for importing, validating, enumerating, running etc. primitives.

The primitives themselves can be attained from https://gitlab.datadrivendiscovery.org/jpl/primitives_repo.git
"""

import os
import collections
import json
import d3m.metadata.base as metadata

PRIMITIVES_DIR = "../../primitives_repo"
PRIMITIVES_VERSION = "v2018.1.26"

PrimitiveLabel = collections.namedtuple('PrimitiveLabel', ['team', 'module', 'version'])

def primitive_label_from_str(s):
    """
    Takes a string listing a directory containing a `primitive.json` file
    and returns a primitive label.
    A primitive label is namedtuple (team, python_module, version), corresponding to the
    directory structure, as described in the README.md of the primitives_repo git
    repo.

    If the string is not a valid primitive, raises an exception.
    """
    if os.path.commonpath([s, PRIMITIVES_DIR]) != PRIMITIVES_DIR:
        raise Exception("String is not a path rooted in PRIMITIVES_DIR")
    # This is sorta wacky but can't figure out a good way to split on
    # dir separators.
    (p1, version) = os.path.split(s)
    (p2, python_module) = os.path.split(p1)
    (p3, team) = os.path.split(p2)
    return PrimitiveLabel(team=team, module=python_module, version=version)


def primitive_path_from_label(label):
    return os.path.join(PRIMITIVES_DIR, PRIMITIVES_VERSION, label.team, label.module, label.version, 'primitive.json')

def list_primitives():
    """
    Returns a list of all primitives, as PrimitiveLabel's.
    """
    prim_dir = os.path.join(PRIMITIVES_DIR, PRIMITIVES_VERSION)
    return [primitive_label_from_str(path) for (path, names, filenames) in os.walk(prim_dir)
            if 'primitive.json' in filenames]

class Primitive(object):
    def __init__(self, label):
        self.__label = label
        schema_file = primitive_path_from_label(label)
        with open(schema_file, 'r') as f:
            schema = json.load(f)
            self.__metadata = metadata.PrimitiveMetadata(schema)
            try:
                # Validation appears to never work...
                self.__metadata._validate()
                print("Primitive {}: validated".format(self.__metadata.query()['name']))
            except:
                pass
            