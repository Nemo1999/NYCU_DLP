import torch
import torch.nn as nn
import torch.optim as optim

from get_device import device


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



nz = 100
ngf = 64
ndf = 64

# Generator Code

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc=24):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nchannel=3) x 64 x 64
        )

    def forward(self, z, c):
        # z should have shape=(b_size, nz, 1, 1)
        # c should have shape=(b_size, nc, 1, 1)
        input = torch.cat([z, c], 1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc=24):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.embbed_cond = nn.Linear(nc, nc, bias=False)
        self.main = nn.Sequential(
            # input is (nchannel) x 64 x 64
            # the extra dimension is for condition
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, nc, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.final = nn.Linear(nc, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, img, c):
        c = self.embbed_cond(c).view(-1, 24)
        # input = torch.cat([img, c], 1)
        h = self.main(img).view(-1,24)
        out = self.final(h).view(-1,)
        out += torch.sum(h * c, dim=1, keepdim=False)
        return self.sigmoid(out)
