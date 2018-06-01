"""
A module for importing, validating, enumerating, running etc. primitives.
"""

import d3m.index
import d3m.metadata.base as metadata
import problem

def list_primitives():
    """
    Returns a list of all primitives, as PrimitiveLabel's.
    """
    prims = d3m.index.search()
    primcs = [PrimitiveClass(pc.metadata, pc) for p,pc in prims.items()]
    
    #(X, y) = problem.load_dataset("file:///home/sray/DARPA_D3M/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json")
    #for p in primcs:
        #if p.name == 'sklearn.ensemble.weight_boosting.AdaBoostClassifier':
            #continue
        #if p.name == 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier':
            #continue

        #if p._metadata.query()['primitive_family'] == "CLASSIFICATION":
            #print(p.python_path)
            #pipe = problem.SolutionDescription(p._metadata, p.classname, None)
            #score = pipe.score_solution(X, y, -1)
            #print(score)
    return primcs

class PrimitiveClass(object):
    """
    A loader that mainly just contains primitive metadata.
    Eventually it should be able to do more useful things too.
    """
    def __init__(self, metadata, classname):
        self._metadata = metadata
        self.id = metadata.query()['id']
        self.name = metadata.query()['name']
        self.version = metadata.query()['version']
        self.python_path = metadata.query()['python_path'] 
        self.classname = classname
