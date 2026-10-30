import unittest
from .evaluate_training import assert_split_separation,TEST_SEED
from .training_study import CONFIG

class EvaluationChecks(unittest.TestCase):
    def test_amended_seed_ranges_disjoint(self):
        assert_split_separation()
        self.assertNotEqual(TEST_SEED,CONFIG['test_seed'])

if __name__=='__main__':unittest.main()
