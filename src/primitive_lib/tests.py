import unittest
from . import *

class TestEnumeration(unittest.TestCase):
    def test_label(self):
        "Tests whether we can turn a primitive into a valid label"
        prim_path = os.path.join(PRIMITIVES_DIR, PRIMITIVES_VERSION, "test_team", "test_module", "0.0.0")
        label = primitive_label_from_str(prim_path)
        target_label = PrimitiveLabel(team="test_team", module="test_module", version="0.0.0")
        self.assertEqual(label, target_label)

    def test_list(self):
        prims = list_primitives()
        # for p in prims:
        #     prim = Primitive(p)
            # print(prim)
            # install_primitive(prim._metadata)
        target_label = PrimitiveLabel(team='JPL', module='d3m.primitives.sklearn_wrap.SKRandomForestRegressor', version='0.1.0')
        self.assertIn(target_label, prims)

        target_label = PrimitiveLabel(team='CMU', module='d3m.primitives.cmu.autonlab.find_projections.Search', version='2.0')
        self.assertIn(target_label, prims)

    def test_list_all(self):
        "Test whether we can create (or at least load metadata for) all the primitives in the index."
        prims = list_primitives()
        for label in prims:
            Primitive(label)
        


if __name__ == '__main__':
    unittest.main()