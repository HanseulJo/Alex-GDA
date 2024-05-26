from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resample):
        super(ResidualBlock, self).__init__()
        if resample=='down':
            self.shortcut = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
            self.main = nn.Sequential(
                nn.InstanceNorm2d(in_channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(in_channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            )
        elif resample=='up':
            self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
            self.main = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            )
        elif resample==None:
            self.shortcut = nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            self.main = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            )
        else:
            raise Exception('invalid resample value')

        if in_channels==out_channels and resample==None:
            self.shortcut = nn.Identity() # Identity skip-connection

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class GoodGenerator(nn.Module):
    def __init__(self, cfg):
        super(GoodGenerator, self).__init__()
        self.cfg = cfg
        self.main = nn.Sequential(
            nn.ConvTranspose2d(cfg.nz, cfg.ngf * 8, cfg.image_size//16, 1, 0, bias=False),
            ResidualBlock(cfg.ngf * 8, cfg.ngf * 8, 'up'),
            ResidualBlock(cfg.ngf * 8, cfg.ngf * 4, 'up'),
            ResidualBlock(cfg.ngf * 4, cfg.ngf * 2, 'up'),
            ResidualBlock(cfg.ngf * 2, cfg.ngf * 1, 'up'),
            nn.BatchNorm2d(cfg.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(cfg.ngf, cfg.nc, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class GoodDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(GoodDiscriminator, self).__init__()
        self.cfg = cfg
        self.main = nn.Sequential(
            nn.Conv2d(cfg.nc, cfg.ndf, 3, 1, 1),
            ResidualBlock(cfg.ndf * 1, cfg.ndf * 2, 'down'),
            ResidualBlock(cfg.ndf * 2, cfg.ndf * 4, 'down'),
            ResidualBlock(cfg.ndf * 4, cfg.ndf * 8, 'down'),
            ResidualBlock(cfg.ndf * 8, cfg.ndf * 8, 'down'),
            nn.Conv2d(cfg.ndf * 8, 1, cfg.image_size//16, 1, 0)
        )

    def forward(self, x):
        return self.main(x)
