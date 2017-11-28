import unittest

from org.campagnelab.dl.pytorch.cifar10.CrossValidatedProblem import CrossValidatedProblem
from org.campagnelab.dl.pytorch.cifar10.STL10Problem import STL10Problem


class TestCrossValidatedProblem(unittest.TestCase):
    def testTrainingSize(self):
        batch_size=50
        problem = STL10Problem(mini_batch_size=batch_size)
        self.assertEqual(5000, len(problem._trainset))
        with  open("./data/stl10_binary/fold_indices.txt") as folds:
            fold_definitions =folds.readlines()
            fold =fold_definitions[0]
            splitted = fold.split(sep=" ")
            splitted.remove("\n")
            self.assertEqual(1000, len(splitted))
            train_indices = [int(index) for index in splitted]
            reduced_problem = CrossValidatedProblem(problem, train_indices)
            self.assertEqual(1000, len(reduced_problem.train_loader())*batch_size)
            self.assertEqual(4000, len(reduced_problem.test_loader())*batch_size)