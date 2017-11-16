import torch
from torch import nn
import threading


# Code adapted from https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/13


class MyFeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers=None):
        super(MyFeatureExtractor, self).__init__()
        self.submodule = submodule
        self.outputs = []
        self.lock = threading.Lock()
        if extracted_layers is None:
            self.match_all = True
            self.extracted_layers = []
        else:
            self.match_all = False
            self.extracted_layers = extracted_layers
        self.remove_handles = []


    def register(self):
        with self.lock:
            for name, module in self.submodule._modules.items():
                if self.match_all or name in self.extracted_layers:
                    # print("registering hook on {}".format(name))
                    self.remove_handles += [module.register_forward_hook(lambda m, i, o: self._accumulate_output(m, o))]

    def _accumulate_output(self, module, output):
        with self.lock:
            mini_batch_size = output.size()[0]

            flattened = output.view(mini_batch_size, -1)
            #print("output.size={}".format(flattened.size()))
            self.outputs += [flattened]
            #print("obtaining outputs {} from module: # outputs {}".format(module, [len(o) for o in self.outputs]))

    def collect_outputs(self, x, seed):

        #print("x.cuda? {}".format(x.is_cuda))
        # do a forward to collect the outputs:
        self.submodule(x)
        #print("returning #outputs: {}".format(len(self.outputs)))
        # return the outputs
        return self.outputs


    def clear_outputs(self):
        with self.lock:
            self.outputs = []


    def cleanup(self):
        with self.lock:
            # remove the registered hook:
            for handle in self.remove_handles:
                handle.remove()
