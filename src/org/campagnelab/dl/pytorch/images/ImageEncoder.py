import torch
from torch import nn

from org.campagnelab.dl.pytorch.images.ImageGenerator import weights_init


class ImageEncoder(nn.Module):
    def __init__(self, model, input_shape, number_encoded_features,
                 ngpu=1, use_cuda=False):
        super(ImageEncoder, self).__init__()
        self.ngpu = ngpu
        self.main = model
        self.number_encoded_features=number_encoded_features
        num_out=model.estimate_output_size_with_model(input_shape, model=model, use_cuda=use_cuda)
        self.projection = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            # reduce the number of state features progressively to nz
            nn.Linear(num_out, int(num_out/4)),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(int(num_out/4), int(num_out/8)),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(int(num_out/8), number_encoded_features)
        )
        self.main.apply(weights_init)
        if use_cuda:
            self.projection=self.projection.cuda()

    def parameters(self):
        for param in self.main.parameters():
            yield param
        for param in self.projection.parameters():
            yield param

    def train(self):
        self.main.train()
        self.projection.train()

    def eval(self):
        self.main.eval()
        self.projection.eval()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.view(output.size()[0], -1)
        output = self.projection(output)
        output = output.view(output.size()[0], self.number_encoded_features, 1, 1)
        return output