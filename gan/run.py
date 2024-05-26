"""
    refs:
    - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    - https://github.com/mseitzer/pytorch-fid
    - https://www.kaggle.com/code/ibtesama/gan-in-pytorch-with-fid/notebook#Fretchet-Inception-Distance
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import argparse
from copy import deepcopy
from dotmap import DotMap
from hydra import initialize, compose
import json
import math
from omegaconf import OmegaConf

import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
from IPython.display import HTML
from tqdm import tqdm

from src.model import GoodGenerator, GoodDiscriminator, weights_init, InceptionV3
from src.evaluation import calculate_frechet, calculate_gradient_penalty


def run(args):
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    log_path = os.path.join('logs', args.log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)

    # Save config
    dict_cfg = OmegaConf.to_container(cfg)
    with open(os.path.join(log_path, 'cfg.json'), 'w') as f:
        json.dump(dict_cfg, f, indent=2)

    # Set random seed
    seed = cfg.seed
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results

    #normalizing input between -1 and 1
    transform=transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(0, 1),
    ])
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    if config_name in ['cifar10', 'cifar10_large']:
        dataset = dset.CIFAR10(root=cfg.dataroot, train=True, download=True, transform=transform)
    elif config_name == 'mnist':
        dataset = dset.MNIST(root=cfg.dataroot, train=True, download=True, transform=transform)
    elif config_name == 'celeba':
        dataset = dset.CelebA(root=cfg.dataroot, split='train', download=True, transform=transform)
    elif config_name == 'lsun':
        category = log_path.split('/')[1]
        print("LSUN category: ", category)
        if category == 'lsun_bedroom':
            dataset = dset.LSUN(root=cfg.dataroot, classes=['bedroom_train'], transform=transform)
        elif category == 'lsun_church_outdoor':
            dataset = dset.LSUN(root=cfg.dataroot, classes=['church_outdoor_train'], transform=transform)
    else:
        raise ValueError(f'wrong config name: {config_name}')

    # Create the dataloader
    batch_size = cfg.batch_size
    batch_size_eval = cfg.batch_size_eval
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=cfg.workers)
    # dataloader_eval = torch.utils.data.DataLoader(dataset, batch_size=batch_size_eval,
    #                                         shuffle=True, num_workers=cfg.workers)

    # Decide which device we want to run on
    gpus = cfg.gpus
    ngpu = len(gpus)
    device = torch.device(f"cuda:{gpus[0]}" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig("training_images.pdf")

    # Create the generator
    netG = GoodGenerator(cfg).to(device)

    # Create the Discriminator
    netD = GoodDiscriminator(cfg).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, gpus)
        netD = nn.DataParallel(netD, gpus)
        
    # Apply the ``weights_init`` function to randomly initialize all weights
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Inception model for calculating FID score
    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    # inception_model = InceptionV3([block_idx]).to(device)

    # Initialize the ``BCELoss`` function
    # criterion = nn.BCELoss()  # DCGAN

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, cfg.nz, 1, 1, device=device)
    # fixed_noises_for_fid = [torch.randn(batch_size_eval, cfg.nz, 1, 1, device=device) for _ in range(math.ceil(50000/batch_size_eval))]

    # Establish convention for real and fake labels during training 
    # Apply label smoothing
    real_label = 0.9
    fake_label = 0.1 

    # Setup Adam optimizers for both G and D
    optimizer_kwargs = OmegaConf.to_container(cfg.optimizer)
    optimizerG_type = optimizer_kwargs['G'].pop('type')
    optimizerD_type = optimizer_kwargs['D'].pop('type')
    optimizerG = getattr(optim, optimizerG_type, optim.Adam)(netG.parameters(), **optimizer_kwargs['G'])
    optimizerD = getattr(optim, optimizerD_type, optim.Adam)(netD.parameters(), **optimizer_kwargs['D'])

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    frechet_distances = []
    iters_G = 0

    print("Starting Training Loop...")
    num_epochs = cfg.num_epochs
    lambda_gp = cfg.get('lambda_gp', 10)
    algorithm = cfg.algorithm
    netG_ = deepcopy(netG)
    # For each epoch
    for epoch in range(1, num_epochs+1):

        # For each batch in the dataloader
        pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader))
        for i, data in pbar:
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD_ = deepcopy(netD) if algorithm=='Alex-GDA' else netD # Alex-GDA
            netD.zero_grad()
            # Format batch
            real_image = data[0].to(device)
            b_size = real_image.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            real_output = netD(real_image).view(-1)
            # Calculate loss on all-real batch
            ## errD_real = criterion(real_output, label)
            errD_real = torch.mean(real_output)
            # Calculate gradients for D in backward pass
            ## errD_real.backward()
            D_x = real_output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, cfg.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake_image = netG_(noise)  # Alex-GDA
            # fake_image = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            fake_output = netD(fake_image.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            ## errD_fake = criterion(fake_output, label)  # DCGAN
            errD_fake = torch.mean(fake_output)  # WGAN
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            # errD_fake.backward()
            D_G_z1 = fake_output.mean().item()

            
            # Compute error of D as sum over the fake and the real batches
            ## errD = errD_real + errD_fake
            gradient_penalty = calculate_gradient_penalty(netD, real_image.data, fake_image.data, device)
            errD = -errD_real + errD_fake + lambda_gp * gradient_penalty
            errD.backward()
            if algorithm != 'Sim-GDA':
                optimizerD.step()
            if algorithm == 'Alex-GDA':
                for online, target in zip(netD.parameters(), netD_.parameters()):
                    target.data = cfg.gamma * online.data + (1 - cfg.gamma) * target.data  # Alex-GDA
            # Save Losses for plotting later
            D_losses.append(errD.item())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG_ = deepcopy(netG) if algorithm=='Alex-GDA' else netG # Alex-GDA
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            fake_image = netG(noise)
            fake_output = netD_(fake_image).view(-1)  # Alex-GDA
            # fake_output = netD(fake_image).view(-1)
            # Calculate G's loss based on this output
            ## errG = criterion(fake_output, label)  # DCGAN
            errG = -torch.mean(fake_output)  # WGAN
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = fake_output.mean().item()
            # Update G
            if algorithm == 'Sim-GDA':
                optimizerD.step()
            optimizerG.step()
            if algorithm=='Alex-GDA':
                for online, target in zip(netG.parameters(), netG_.parameters()):
                    target.data = cfg.delta * online.data + (1 - cfg.delta) * target.data  # Alex-GDA
            # Save Losses for plotting later
            G_losses.append(errG.item())

            iters_G += 1
            
            # Output training stats
            # if i in [1, len(dataloader)]:
            if i % 100 == 1 or i == len(dataloader):
                pbar.set_description(
                    f"[{epoch}/{num_epochs}] "
                    f"Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}, "
                    f"D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

            # if (epoch%(num_epochs//10)==0) or ((epoch == num_epochs) and (i == len(dataloader))):
            #     with torch.no_grad():
            #         fake_display = netG(fixed_noise).detach().cpu()
            # Check how the generator is doing by saving G's output on fixed_noise
            # if ((epoch == num_epochs) and (i == len(dataloader))):
            #     img_list.append(vutils.make_grid(fake_display, padding=2, normalize=True))
            #     # Calculate Fréchet distance
            #     with torch.no_grad():
            #         fake_images = [(netG(z).detach().cpu(),) for z in fixed_noises_for_fid]
            #     frechet_distance = calculate_frechet(dataloader_eval,fake_images,inception_model,batch_size_eval,device)
            #     frechet_distances.append(frechet_distance)

            #     print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Frechet Inception Distance Score {frechet_distance}")

        if (epoch%(num_epochs//10)==0) or ((epoch == num_epochs) and (i == len(dataloader))):
            with torch.no_grad():
                fake_display = netG(fixed_noise).detach().cpu()
            plt.figure(figsize=(10, 10))
            plt.axis("off")
            pictures=vutils.make_grid(fake_display,nrow=8,padding=2, normalize=True)
            plt.imshow(np.transpose(pictures,(1,2,0)))
            plt.savefig(os.path.join(log_path, f"fake_epoch{epoch}.pdf"))
            plt.close()

    # Visualization (1) Loss & FID score

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(log_path, "loss.pdf"))
    plt.close()

    # plt.figure(figsize=(10,5))
    # plt.title("Frechet Inception Distance During Training")
    # plt.plot(frechet_distances, label=f"last:{frechet_distances[-1]}")
    # plt.xlabel("epochs")
    # plt.ylabel("FID score")
    # plt.legend()
    # plt.savefig(os.path.join(log_path, "fid_score.pdf"))
    # plt.close()

    # Visualization (2) Images

    # fig = plt.figure(figsize=(8,8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # html_obj = HTML(ani.to_jshtml())

    # html = html_obj.data
    # with open(os.path.join(log_path, 'training.html'), 'w') as f:
    #     f.write(html)

    # Visualization (3) Generated Imgaes
    
    # # Grab a batch of real images from the dataloader
    # real_batch = next(iter(dataloader))

    # # Plot the real images
    # plt.figure(figsize=(15,15))
    # plt.subplot(1,2,1)
    # plt.axis("off")
    # plt.title("Real Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    
    # # Plot the fake images from the last epoch
    # plt.subplot(1,2,2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    # plt.savefig(os.path.join(log_path, "real_vs_fake.pdf"))
    # plt.close()

    # Save training results
    with open(os.path.join(log_path, 'losses_D.json'), 'w') as f:
        json.dump(D_losses, f, indent=2)
    with open(os.path.join(log_path, 'losses_G.json'), 'w') as f:
        json.dump(G_losses, f, indent=2)
    # with open(os.path.join(log_path, 'FID.json'), 'w') as f:
    #     json.dump(frechet_distances, f, indent=2)
    torch.save(netG.state_dict(), os.path.join(log_path, f"netG.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str, default='./configs')
    parser.add_argument('--config_name', type=str, default='cifar10') 
    parser.add_argument('--log_path', type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--overrides',  type=str, default=[],   nargs='*')

    args = parser.parse_args()
    args = vars(args)
    print(args)

    run(args)