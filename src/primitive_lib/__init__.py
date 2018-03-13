

"""
A module for importing, validating, enumerating, running etc. primitives.

The primitives themselves can be attained from https://gitlab.datadrivendiscovery.org/jpl/primitives_repo.git
"""

import os
import collections

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


def list_primitives():
    """
    Returns a list of all primitives, as strings.
    """
    prim_dir = os.path.join(PRIMITIVES_DIR, PRIMITIVES_VERSION)
    return [primitive_label_from_str(path) for (path, names, filenames) in os.walk(prim_dir)
            if 'primitive.json' in filenames]