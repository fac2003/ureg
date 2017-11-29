import unittest

import torch
from torch.autograd import Variable
from torch.nn import BCELoss, Sequential

from org.campagnelab.dl import pytorch
from org.campagnelab.dl.pytorch.ureg.MyFeatureExtractor import MyFeatureExtractor
from org.campagnelab.dl.pytorch.ureg.tests.SimpleModel import SimpleModel


class TestMyFeatureExtractor(unittest.TestCase):
    def setUp(self):
        num_main_model_input_features = 100
        num_features = 40
        self.num_main_model_input_features = num_main_model_input_features
        self.num_features = num_features

        self.model = Sequential(
            torch.nn.Linear(num_main_model_input_features, num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, 2),
            torch.nn.Softmax()
        )

    def test_extract_inner_activations(self):
        features = Variable(torch.ones(1, self.num_main_model_input_features))
        mfe = MyFeatureExtractor(self.model, None)
        mfe.register()
        mfe.clear_outputs()
        unsupervised_outputs = mfe.collect_outputs(features)
        mfe.cleanup()
        mfe.clear_outputs()
        self.assertEqual(6, len(unsupervised_outputs))

    def test_extract_inner_activations_subset(self):
        features = Variable(torch.ones(1, self.num_main_model_input_features))
        mfe = MyFeatureExtractor(self.model, None)
        mfe.register()
        mfe.clear_outputs()
        unsupervised_outputs = mfe.collect_outputs(features,
                                                   [True,
                                                    True,
                                                    False,
                                                    False,
                                                    False,
                                                    True])
        mfe.cleanup()
        mfe.clear_outputs()
        self.assertEqual(3, len(unsupervised_outputs))
