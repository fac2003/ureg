from __future__ import print_function
import argparse
import os
import random

import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='images | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'images':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif opt.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                                transform=transforms.ToTensor())
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3


    def get_random_slope():
        pi = math.atan(1) * 4
        angle = random.uniform(pi, -pi)
        slope = math.tan(angle)
        return slope

    def half_images(inputs, slope):
        def above_line(xp, yp, slope, b):
            # remember that y coordinate increase downward in images
            yl = slope * xp + b
            return yp < yl

        def on_line(xp, yp, slope, b):
            # remember that y coordinate increase downward in images
            yl = slope * xp + b
            return abs(yp - yl) < 2

        uinputs = Variable(inputs, requires_grad=True)
        uinputs2 = Variable(inputs, requires_grad=False)

        mask_up = torch.ByteTensor(uinputs.size()[2:])  # index 2 drops minibatch size and channel dimension.
        mask_down = torch.ByteTensor(uinputs.size()[2:])  # index 2 drops minibatch size and channel dimension.

        channels = uinputs.size()[1]
        width = uinputs.size()[2]
        height = uinputs.size()[3]
        # print("mask down-------------")
        # fill in the first channel:
        for x in range(0, width):
            for y in range(0, height):
                above_the_line = above_line(x - width / 2, y, slope=slope, b=height / 2.0)
                on_the_line = on_line(x - width / 2, y, slope=slope, b=height / 2.0)
                mask_up[x, y] = 1 if above_the_line and not on_the_line else 0
                mask_down[x, y] = 0 if above_the_line  else \
                    (1 if not on_the_line else 0)
                # print("." if mask_down[x, y] else " ",end="")
                #    print("." if mask_up[x, y] else " ",end="")
        # print("|")

        # print("-----------mask down")
        mask1 = torch.ByteTensor(uinputs.size()[1:])
        mask2 = torch.ByteTensor(uinputs.size()[1:])

        if opt.cuda:
             mask_up = mask_up.cuda()
             mask_down = mask_down.cuda()
             mask1 = mask1.cuda()
             mask2 = mask2.cuda()
        for c in range(0, channels):
            mask1[c] = mask_up
            mask2[c] = mask_down
        mask1 = Variable(mask1, requires_grad=False)
        mask2 = Variable(mask2, requires_grad=False)

        return uinputs.masked_fill(mask1, 0), uinputs2.masked_fill(mask2, 0)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    class _netG(nn.Module):
        def __init__(self, ngpu):
            super(_netG, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is output of discriminator, going into a convolution to reconstitute image2
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output


    netG = _netG(ngpu)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)


    class _netD(nn.Module):
        def __init__(self, ngpu):
            super(_netD, self).__init__()
            self.ngpu = ngpu
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
            self.projection=nn.Sequential(
                # state size. (ndf*8) x 4 x 4
                # reduce the number of state features progressively to nz
                nn.Linear(ndf * 8*4*4, ndf*8*4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5, inplace=True),
                nn.Linear(ndf*8*4, nz*8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5, inplace=True),
                nn.Linear(nz * 8, nz)
            )

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            output=output.view(output.size()[0],-1)
            output=self.projection(output)
            output=output.view(output.size()[0],nz,1,1)
            return output


    netD = _netD(ngpu)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.MSELoss()

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    all_params=[]
    all_params+=list(netD.parameters())
    all_params+= list(netG.parameters())
    optimizer = optim.Adam(all_params, lr=opt.lr, betas=(opt.beta1, 0.999))
    #optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    netG.train()
    netD.train()

    for epoch in range(opt.niter):
        average_loss=0
        count=0
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            netG.zero_grad()
            optimizer.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            image1, image2=half_images(input, slope=random.random())
            # train the discriminator/generator pair on the first half of the image:
            encoded=netD(image1)

            output = netG(encoded)
            reconstituted_image=output+image1
            full_image=Variable(input, volatile=True)
            loss = criterion(reconstituted_image, full_image)
            loss.backward()
            optimizer.step()


            average_loss+=loss.data[0]
            count+=1


            if i % 1000 == 0:
                print('[%d/%d][%d/%d] Loss: %.4f '
                      % (epoch, opt.niter, i, len(dataloader),
                         average_loss / count))
                average_loss=0
                count=0
                image1, image2 = half_images(real_cpu, slope=random.random())

                vutils.save_image(image1.data,
                        '%s/real_image1.png' % opt.outf,
                        normalize=True)

                vutils.save_image(image2.data,
                                  '%s/real_image2.png' % opt.outf,
                                  normalize=True)
                netD.eval()
                netG.eval()
                # train the discriminator/generator pair on the first half of the image:
                encoded=netD(image1)
                img2 = netG(encoded)

                vutils.save_image(img2.data+image1.data,
                        '%s/fake_samples_split_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                netD.train()
                netG.train()

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
