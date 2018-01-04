import torch
from torch import nn

from org.campagnelab.dl.pytorch.images.ImageGenerator import weights_init
from org.campagnelab.dl.pytorch.images.utils import init_params


class ImageEncoder(nn.Module):
    def __init__(self, model, input_shape, number_encoder_features=100,
                 number_representation_features=100,
                 ngpu=1, use_cuda=False):
        super(ImageEncoder, self).__init__()
        self.ngpu = ngpu
        nc=3
        ndf=number_encoder_features
        self.nz=number_representation_features
        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.number_encoded_features=number_encoder_features

        self.projection = nn.Sequential(
                # state size. (ndf*8) x 4 x 4
                # reduce the number of state features progressively to nz
                nn.Linear(ndf * 8*4*4, ndf*8*4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(ndf*8*4, self.nz*8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.nz * 8, self.nz)
            )
        init_params(self.main)
        init_params(self.projection)

        if use_cuda:
            self.projection=self.projection.cuda()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.view(output.size()[0], -1)
        output = self.projection(output)
        output = output.view(output.size()[0], self.nz, 1, 1)
        return output