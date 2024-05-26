import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import argparse
from dotmap import DotMap
from hydra import initialize, compose
from glob import glob
import random
import json
import math
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from src.model import GoodGenerator, GoodDiscriminator, weights_init, InceptionV3
from src.evaluation.fid_score import calculate_activation_statistics, calculate_frechet_distance

def evaluate(args):
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    category = args.category

    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)

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

    if config_name in ['cifar10', 'cifar10_large']:
        dataset = dset.CIFAR10(root=cfg.dataroot, train=True, download=True, transform=transform)
    elif config_name == 'mnist':
        dataset = dset.MNIST(root=cfg.dataroot, train=True, download=True, transform=transform)
    elif config_name == 'celeba':
        dataset = dset.CelebA(root=cfg.dataroot, split='train', download=True, transform=transform)
    elif config_name == 'lsun':
        print("LSUN category: ", category)
        if category == 'lsun_bedroom':
            dataset = dset.LSUN(root=cfg.dataroot, classes=['bedroom_train'], transform=transform)
        elif category == 'lsun_church_outdoor':
            dataset = dset.LSUN(root=cfg.dataroot, classes=['church_outdoor_train'], transform=transform)
    else:
        raise ValueError(f'wrong config name: {config_name}')

    # Create the dataloader
    batch_size_eval = 2000
    dataloader_eval = torch.utils.data.DataLoader(dataset, batch_size=batch_size_eval, shuffle=False, num_workers=cfg.workers)

    gpus = cfg.gpus
    ngpu = len(gpus)
    device = torch.device(f"cuda:{gpus[0]}" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netG = GoodGenerator(cfg).to(device)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)
    fixed_noises_for_fid = [torch.randn(batch_size_eval, cfg.nz, 1, 1, device=device) for _ in range(math.ceil(50000/batch_size_eval))]

    # real stat
    mu_1, std_1 = calculate_activation_statistics(dataloader_eval,inception_model,batch_size_eval,device=device)

    paths = sorted(glob(f'logs/{config_name if config_name!="lsun" else category}/*/*/*/'))
    print(paths)

    for log_path in paths:
        if os.path.exists(os.path.join(log_path, 'FID.json')): continue
        if not os.path.exists(os.path.join(log_path, 'netG.pt')): continue
        print(log_path)
        
        netG.load_state_dict(torch.load(os.path.join(log_path, 'netG.pt')))
        with torch.no_grad():
            fake_images = [(netG(z).detach().cpu(),) for z in fixed_noises_for_fid]
        mu_2, std_2 = calculate_activation_statistics(fake_images,inception_model,batch_size_eval,device=device)

        fid = [calculate_frechet_distance(mu_1, std_1, mu_2, std_2)]

        with open(os.path.join(log_path, 'FID.json'), 'w') as f:
            json.dump(fid, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str, default='./configs')
    parser.add_argument('--config_name', type=str, default='cifar10') 
    parser.add_argument('--category', type=str, default='lsun_bedroom')
    parser.add_argument('--overrides',  type=str, default=['gpus=[0]'],   nargs='*')

    args = parser.parse_args()
    args = vars(args)
    print(args)

    evaluate(args)