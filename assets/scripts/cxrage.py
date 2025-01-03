import fastai
from fastai.vision.all import *
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms

def load_model(path='assets/'):
    # construct base model
    learn = vision_learner(
        DataLoaders.from_dsets([], [], bs=1), # dummy dataloader 
        fastai.vision.models.resnet34,
        n_out = 1,
        loss_func = MSELossFlat(), # dummy loss
    )

    # modifications made for cxr-age
    learn.model[1] = nn.Sequential(
        *learn.model[1][:-5],
        nn.Linear(1024, 512, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, 16, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(16),
        nn.Linear(16, 1, bias=True)
    )

    # load params
    # next two functions assume model is in <path>/models/PLCO_Fine_Tuned_120419.pth
    # idk why
    learn.path = Path(path)
    learn.load('PLCO_Fine_Tuned_120419')
    learn.eval()

    return learn

def process(img, path=True):
    # we receive a path to an image
    if path:
        img = Image.open(img).convert('RGB')
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet normalization
    ])
    img = transform(img).unsqueeze(0)

    return img

# biological age is a quadratic function of model output
def age_fn(output):
    output = output * 8.03342449139388 + 63.8723890235948
    output = output * 6.75523 - 0.03771 * output * output - 213.77257

    return output