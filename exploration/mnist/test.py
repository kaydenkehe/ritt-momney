import torch
import yaml
from vae import VAE

device = 'cpu'

# load mnist vae
config_path = 'VAE-Pytorch/config/vae_kl.yaml'
with open (config_path, 'r') as file:
        config = yaml.safe_load(file)

vae = VAE(config['model_params'])
vae.to(device)
pth_path = 'models/vae.pth'
vae.load_state_dict(torch.load(pth_path, map_location=device))

# test on mnist image

