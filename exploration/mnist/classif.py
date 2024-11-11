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

# %%
import torch
import yaml
from vae import VAE
from mnist import Network
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn


# %%
device = 'cpu'

# load mnist vae
config_path = 'VAE-Pytorch/config/vae_kl.yaml'
with open (config_path, 'r') as file:
        config = yaml.safe_load(file)

vae = VAE(config['model_params'])
vae.to(device)
pth_path = 'VAE-Pytorch/vae_kl/best_vae_kl_ckpt.pth'
vae.load_state_dict(torch.load(pth_path, map_location=device))

# load mnist classifier
classifier = Network()
classifier.to(device)
pth_path = 'hand-written-digit-classification/MNIST_model.pth'
classifier.load_state_dict(torch.load(pth_path, map_location=device))
classifier.eval()


# %%
path = 'VAE-Pytorch/data/train/images/4/20712.png'
img = Image.open(path).convert('L')

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

img = transform(img).unsqueeze(0).to(device)
out = vae(img)
mu = out['mean']
std = torch.exp(0.5 * out['log_variance'])
z = mu + std * torch.randn_like(std)


# %%
def loss_fn(sm_output):
    return -sm_output[0, 9]


# %%
img_temp = Image.open(path).convert('L')
img_temp = transform(img_temp).unsqueeze(0).to(device)
# predict on original image
pred = torch.exp(classifier(img_temp))
print(pred)
# predict on generated image
img_orig = vae.generate(z)
pred = torch.exp(classifier(img_orig))
print(pred)

# plot original and generated images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_temp.squeeze().detach().cpu().numpy(), cmap='gray')
ax[1].imshow(img_orig.squeeze().detach().cpu().numpy(), cmap='gray')


# %%
# optimize over latent space rep (z) for class 9
epochs = 75
lr = 0.025
imgs = []

mu_par = nn.Parameter(mu.clone())
mu_opt = torch.optim.AdamW([mu_par], lr=lr)

for epoch in range(epochs):
    mu_opt.zero_grad()
    img = vae.generate(mu_par)
    pred = classifier(img)
    loss = loss_fn(pred)
    loss.backward()
    mu_opt.step()

    imgs.append(img.detach().cpu())
    print(f'{epoch} {loss.item()}')


# %%
print(loss.item())


# %%
# output every image in streams/1
for i, img in enumerate(imgs):
    img = img.squeeze().numpy()
    img = (img + 1) / 2 * 255
    img = Image.fromarray((img).astype('uint8'))
    img.save(f'streams/1/{i}.png')


# %%
# plot images
fig, axes = plt.subplots(1, len(imgs), figsize=(20, 2))
for i, ax in enumerate(axes):
    ax.imshow(imgs[i].squeeze().numpy(), cmap='gray')
    ax.set_title(f'{torch.exp(classifier(imgs[i])[0, 9]).item():.2f}')
plt.savefig('runitup.png')


# %%
# plot animation of images superimposed with prediction probability of class 9
# remove axes, tight padded layout
import matplotlib.animation as animation
from matplotlib import rc

# remove axes
fig, ax = plt.subplots(figsize=(6, 6))

ims = []
for i in range(len(imgs)):
    im = ax.imshow(imgs[i].squeeze().numpy(), cmap='gray')
    text = ax.text(0.2, 1.4, f'{torch.exp(classifier(imgs[i])[0, 9]).item():.2f}', color='red', fontsize=20)
    ax.axis('off')
    plt.tight_layout(pad=1.5)
    ims.append([im, text])

ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True, repeat_delay=1000)
rc('animation', html='jshtml')
ani.save('runitup.gif', writer='imagemagick')


# %%

