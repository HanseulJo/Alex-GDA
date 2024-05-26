from torch import nn
try:
    from utils import weights_init
except ImportError:
    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg

        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(cfg.nc, cfg.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(cfg.ndf, cfg.ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(cfg.ndf * 2),  # no BN for WGAN-GP
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(cfg.ndf * 2, cfg.ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(cfg.ndf * 4),  # no BN for WGAN-GP
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(cfg.ndf * 4, cfg.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(cfg.ndf * 8),  # no BN for WGAN-GP
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(cfg.ndf * 8, 1, cfg.image_size//16, 1, 0, bias=False),
            # nn.Sigmoid()  # no Sigmoid for WGAN
        )

    def forward(self, input):
        return self.main(input)
    

if __name__ == "__main__":
    import torch
    from dotmap import DotMap

    gpus = [2]
    ngpu = len(gpus)
    device = torch.device(f"cuda:{gpus[0]}" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    cfg = DotMap(
        ngpu = ngpu,
        nc = 3,
        nz = 100,
        ndf = 64
    )

    # Create the Discriminator
    netD = Discriminator(cfg).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
        
    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)