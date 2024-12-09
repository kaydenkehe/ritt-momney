# silence warnings
import warnings
warnings.filterwarnings("ignore")

# allow helper import
import sys
sys.path.append('../../assets/scripts')

from cheff import CheffAEModel
from cxrage import load_model, age_fn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor



device = 'cuda'

# load LDM
sdm_path = '../../assets/models/cheff_diff_uncond.pt'
ae_path = '../../assets/models/cheff_autoencoder.pt'
cheff_ae = CheffAEModel(model_path=ae_path, device=device)

# load cxr-age
cxr_age = load_model(path='../../assets/')
cxr_age.model.to(device)

# load, encode sample images
sample_imgs = [Image.open(f'../../assets/cxrs/nih/ae_nih{i}.png')
               for i in range(1, 6)]
with torch.no_grad():
    lreps = [cheff_ae.encode(to_tensor(img).unsqueeze(0).to(device))
             for img in sample_imgs]



# incentivize high outputs and low difference entropy
def loss_fn(a, pred, curr_img, prev_img):
    diff = (curr_img - prev_img.detach())[0]
    diff = diff.mean(dim=0)
    diff[0] += 1e-6 # prevent divide by 0
    diff = (diff - diff.min()) / (diff.max() - diff.min())

    # calculate total variation loss in diff
    tv = torch.mean(torch.abs(diff[1:] - diff[:-1])) + \
            torch.mean(torch.abs(diff[:, 1:] - diff[:, :-1]))

    return a * pred + (1 - a) * tv



a = 0.2
lr = 0.005 
epochs = 20

# image processing for cxr-age
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

imgs = []
ages = []

# iterate over latent space reps
# can't batch :( not enough ram. shut up.
for i, lrep in enumerate(lreps):
    if i > 0:
        print('')
    print(f'Image {i + 1}/{len(lreps)}:')

    lrep_par = nn.Parameter(lrep.clone())
    opt = torch.optim.AdamW([lrep_par], lr=lr)

    imgs_batch = []
    ages_batch = []

    # training loop
    for epoch in range(epochs):
        opt.zero_grad()

        # decode latent space rep
        img_orig = cheff_ae.decode(lrep_par)

        # pass img through cxrage model
        img = preprocess(img_orig)
        pred = cxr_age.model(img)
        loss = loss_fn(a, pred, img_orig,
                       imgs_batch[-1] if len(imgs_batch) > 0 else img_orig)

        # optimize latent rep
        loss.backward()
        opt.step()

        # store image, age prediction
        imgs_batch.append(img_orig)
        ages_batch.append(age_fn(pred).item())

        print(f'Epoch {epoch + 1}/{epochs} | Age: {ages_batch[-1]:.2f}')

    imgs.append(imgs_batch)
    ages.append(ages_batch)



# visualizations
for i, (img, age) in enumerate(zip(imgs, ages)):
    # plot each img and age
    fig, ax = plt.subplots(1, len(img), figsize=(20, 20)) 

    for j, (img_j, age_j) in enumerate(zip(img, age)):
        img_d = img_j.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        img_d = (img_d - img_d.min()) / (img_d.max() - img_d.min())
        
        ax[j].imshow(img_d)
        ax[j].axis('off')
        ax[j].set_title(f'Age: {age_j:.2f}', fontsize=12)

    # Remove extra whitespace
    plt.tight_layout(pad=0.5)

    # Save the figure
    plt.savefig(f'progs/prog_{i + 1}.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()


    # create animation
    fig, ax = plt.subplots(figsize=(2.56, 2.56))  # Assuming 256x256 images

    ims = []
    for img_j, age_j in zip(img, age):
        img_d = img_j.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        img_d = (img_d - img_d.min()) / (img_d.max() - img_d.min())
        
        im = ax.imshow(img_d, animated=True)
        text = ax.text(0.21, 0.94, f'Age: {age_j:.2f}', ha='center',
                       va='center', color='red', fontsize=12,
                       transform=ax.transAxes)
        
        ax.axis('off')
        plt.tight_layout(pad=0.4)
        ims.append([im, text])

    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000,
                                    blit=True)
    ani.save(f'vids/vid_{i + 1}.mp4', writer='ffmpeg', fps=1, dpi=400)
    plt.close()


    # show difference
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))

    # original
    img_od = img[0].detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    img_od = (img_od - img_od.min()) / (img_od.max() - img_od.min())
    ax[0].imshow(img_od)
    ax[0].set_title(f'Original    Age: {age[0]:.2f}', fontsize=18)
    ax[0].axis('off')

    # final
    img_fd = img[-1].detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    img_fd = (img_fd - img_fd.min()) / (img_fd.max() - img_fd.min())
    ax[1].imshow(img_fd)
    ax[1].set_title(f'Final    Age: {age[-1]:.2f}', fontsize=18)
    ax[1].axis('off')

    # difference
    diff = (img[-1] - img[0]).detach().cpu().numpy()
    diff = diff.squeeze().transpose(1, 2, 0)
    diff = diff.mean(-1)
    diff_r = np.maximum(diff, 0) # red, positive
    diff_b = np.maximum(-diff, 0) # blue, negative
    diff = np.stack([diff_r, np.zeros_like(diff), diff_b], axis=-1)
    diff = (diff - diff.min()) / (diff.max() - diff.min())
    diff = diff * 1.5 # amplify for better visualization
    diff = np.clip(diff, 0, 1)
    ax[2].imshow(diff)
    ax[2].set_title('Difference (Red+, Blue-)', fontsize=18)
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'diffs/diff_{i + 1}.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
