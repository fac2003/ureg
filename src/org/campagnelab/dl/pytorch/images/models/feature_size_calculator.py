import torch
from torch.autograd import Variable
from torch.nn import Module


class EstimateFeatureSize(Module):
    """
    Module that helps estimate the number of features up to a point in a module being
    constructed. """
    def estimate_output_size_with_model(self, input_shape, model, use_cuda=False):
        """Calculate the convolution output size using a partially constructed model. """
        bs = 1
        input = Variable(torch.rand(bs, *input_shape))
        if use_cuda:
            input=input.cuda()
        output_feat = model(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def estimate_output_size_with_dual_model(self, input_shape, model):
        """Calculate the convolution output size using a partially constructed model. """
        bs = 1
        input = Variable(torch.rand(bs, *input_shape))

        output_feat,_, _ = model(input,input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def estimate_output_size(self,input_shape, forward_features_function):
        """Calculate the convolution output size using a forward function that outputs the features
        (e.g.,  pre-classifier to figure out the number of features that go into the classifier). """
        bs = 1
        input = Variable(torch.rand(bs, *input_shape))

        output = forward_features_function(input)

        n_size = output.data.view(bs, -1).size(1)
        return n_size

    def estimate_output_size_with_dual_function(self, input_shape, forward_features_function):
        """Calculate the convolution output size using a forward function of a dual model
        (e.g.,  pre-classifier to figure out the number of features that go into the classifier). """
        bs = 1
        input = Variable(torch.rand(bs, *input_shape))

        outputs, outputu, loss = forward_features_function(input, input)

        n_size = outputs.data.view(bs, -1).size(1)
        return n_size
