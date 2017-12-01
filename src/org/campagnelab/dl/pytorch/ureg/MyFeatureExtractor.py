import torch
from torch import nn
import threading


# Code adapted from https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/13


class MyFeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers=None):
        """

        :param submodule:
        :param extracted_layers:
        :param max_output_size: maximum size of an output to collect.
        """
        super(MyFeatureExtractor, self).__init__()
        self.submodule = submodule
        self.outputs = []
        self.collect_output=None
        self.lock = threading.Lock()
        self.output_index=0
        if extracted_layers is None:
            self.match_all = True
            self.extracted_layers = []
        else:
            self.match_all = False
            self.extracted_layers = extracted_layers
        self.remove_handles = []
        self.use_cuda = torch.cuda.is_available()

    def register(self, container=None):
        if container is None:
            container = self.submodule

            self._register_internal(container)

    def _register_internal(self, container):
        if sum(1 for _ in container._modules.items()) == 0:
            #print("Registering module " + str(container))
            # this module is a leaf, register its activations:
            with self.lock:
                self.remove_handles += [
                    container.register_forward_hook(lambda m, i, o: self._accumulate_output(m, o))]
        else:
            for name, module in container._modules.items():
                # if self.match_all or name in self.extracted_layers:
                # else:
                #print("Registering children of module " + name)
                # register the submodules of this module:

                self._register_internal(module)

    def _accumulate_output(self, module, output):
        with self.lock:
            if self.collect_output is None or self.collect_output[self.output_index]:

                mini_batch_size = output.size()[0]

                flattened = output.view(mini_batch_size, -1).clone()
                if self.use_cuda:
                    flattened = flattened.cuda()
                # print("output.size={}".format(flattened.size()))
                self.outputs += [flattened]
                # print("obtaining outputs {} from module: # outputs {}".format(module, [len(o) for o in self.outputs]))

            self.output_index += 1

    def collect_outputs(self,x , collect_output=None):
        self.collect_output=collect_output
        # print("x.cuda? {}".format(x.is_cuda))
        # do a forward to collect the outputs:
        self.submodule(x)
        # print("returning #outputs: {}".format(len(self.outputs)))
        # return the outputs
        return self.outputs

    def clear_outputs(self):

        with self.lock:
            self.output_index=0
            self.outputs = []

    def cleanup(self):
        with self.lock:
            # remove the registered hook:
            for handle in self.remove_handles:
                handle.remove()
            self.output_index = 0