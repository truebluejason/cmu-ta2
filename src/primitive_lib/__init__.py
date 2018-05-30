"""
A module for importing, validating, enumerating, running etc. primitives.

The primitives themselves can be attained from https://gitlab.datadrivendiscovery.org/jpl/primitives_repo.git
"""

import os
import collections
import json
import d3m.metadata.base as metadata

PRIMITIVES_DIR = "/home/sray/DARPA-D3M-newrepo/cmu-ta2_original/primitives_repo"
# PRIMITIVES_VERSION = "v2018.1.26"
PRIMITIVES_VERSION = "v2018.4.18"

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
    # Not worth caching anything; I checked.
    prim_dir = os.path.join(PRIMITIVES_DIR, PRIMITIVES_VERSION)
    print("Prim dir:", prim_dir)
    # print(os.path.abspath(prim_dir))
    prims = [primitive_label_from_str(path) for (path, names, filenames) in os.walk(prim_dir)
            if 'primitive.json' in filenames]
    return prims

class Primitive(object):
    """
    A loader that mainly just contains primitive metadata.
    Eventually it should be able to do more useful things too.
    """
    def __init__(self, label):
        self._label = label
        schema_file = primitive_path_from_label(label)
        with open(schema_file, 'r') as f:
            schema = json.load(f)
            self._metadata = metadata.PrimitiveMetadata(schema)
            # try:
            #     # Validation appears to never work, so don't bother trying.
            #     self._metadata._validate()
            #     print("Primitive {}: validated".format(self._metadata.query()['name']))
            # except:
            #     pass
            
def install_primitive(prim_metadata):
    """
    Very ghetto way of downloading and installing primitives, but...

    Takes a d3m.metadata.base.PrimitiveMetadata object or something

    if there's a nicer method in the d3m package I can't find it.
    d3m.index is useless 'cause none of the primitives are on `pip`.
    And it has no way to install things anyway.
    ...though once things ARE installed, d3m.index.search() Actually Magically Works.

    This will ask for your gitlab password on the console as necessary,
    which isn't really ideal, but oh well.  Only has to be run once per install.

    TODO: make it not re-install packages that already exist
    """
    import subprocess
    m = prim_metadata.query()
    for inst in m['installation']:
        if inst['type'] == 'PIP':
            print("Can install package", m['name'], "from", inst['package_uri'])
            subprocess.run(['pip', 'install', inst['package_uri']], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            break
        print("Can't install package", m['name'])
