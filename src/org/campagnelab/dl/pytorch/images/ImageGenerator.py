import torch
from torch import nn
from torch.nn import Upsample, AvgPool2d, Module


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ImageGenerator(nn.Module):
    def __init__(self, number_encoded_features, number_of_generator_features,

                 output_shape,
                 number_of_channels=3, ngpu=1, use_cuda=False):
        super(ImageGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is output of discriminator, going into a convolution to reconstitute image2
            nn.ConvTranspose2d(number_encoded_features, number_of_generator_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(number_of_generator_features * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(number_of_generator_features * 8, number_of_generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(number_of_generator_features * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(number_of_generator_features * 4, number_of_generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(number_of_generator_features * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(number_of_generator_features * 2, number_of_generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(number_of_generator_features),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(number_of_generator_features, number_of_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        height=output_shape[1]
        if height>64 :
            self.resizer=Upsample(scale_factor=height/64.0)
        if height<64:
            self.resizer=AvgPool2d(int(64/height))
        if use_cuda:
            self.main.cuda()
            self.resizer.cuda()


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        if self.resizer is not None:
            output=self.resizer(output)
        return output
