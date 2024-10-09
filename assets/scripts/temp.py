# This is from a previous project - just here for reference

import sys
sys.path.append('tests')  # allow helper import
# fastai - 2.7.12
# torch - 2.0.1
# torchvision - 0.15.2


from helper import *
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from pytorch_msssim import ssim

type = 'high'

# reform optimized image into something usable
def make_the_image_normal_again(image):
    image = image.cpu().squeeze().detach().numpy()
    image = np.moveaxis(image, 0, -1)  # (C, H, W) -> (H, W, C)
    image = (image - image.min()) / (image.max() - image.min())  # normalize to [0, 1]
    image = (image * 255).astype(np.uint8)  # scale to [0, 255]
    image = Image.fromarray(image)  # convert to PIL image

    return image

learn = load_model()  # load model
original_image = process('samples/rep/00000135_000.png')  # process image

# optimize input image
image = nn.Parameter(original_image.clone())  # convert image to nn.Parameter
optimizer = torch.optim.AdamW([image], lr=0.2)
iterations = 300
similarity_weight = 200  # weight for the similarity loss

for i in range(iterations):
    optimizer.zero_grad()
    output = learn.model(image)

    # Calculate SSIM loss
    ssim_loss = 1 - ssim(image, original_image, data_range=1.0, size_average=True)

    if type == 'high':
        dir_loss = -output  # objective is to maximize output
    elif type == 'low':
        dir_loss = output  # objective is to minimize output

    # Combine losses
    alpha = 0.0025  # Weight for balancing classification and similarity losses
    total_loss = alpha * dir_loss + (1 - alpha) * ssim_loss

    total_loss.backward()
    optimizer.step()

    if i % 10 == 0:
        image_saveable = make_the_image_normal_again(image)
        image_saveable.save(f'tests/img_opt/images/{type}_similar/{i}_{round(output.item(), 3)}.png')
