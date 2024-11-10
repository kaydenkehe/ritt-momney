# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Latent Space Disentanglement
# Find the most relevant axes of the latent space for CXR-Age predictions


# %%
# silence warnings
import warnings
warnings.filterwarnings("ignore")

# allow cxrage import
import sys
sys.path.append('../assets/scripts')

from cxrage import *
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cheff import CheffAEModel
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision.transforms as transforms


# %%
device = 'cuda'

# load LDM
sdm_path = '../../assets/models/cheff_diff_uncond.pt'
ae_path = '../../assets/models/cheff_autoencoder.pt'
cheff_ae = CheffAEModel(model_path=ae_path, device=device)

# load cxr-age
cxr_age = load_model(path='../../assets/')
cxr_age.model.to(device)

# load, encode sample images
sample_imgs = [Image.open(f'../../assets/cxrs/nih/ae_nih{i}.png') for i in range(1, 6)]
with torch.no_grad():
    lreps = [cheff_ae.encode(to_tensor(img).unsqueeze(0).to(device)) for img in sample_imgs]


