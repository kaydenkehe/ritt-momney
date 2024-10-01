# this will probably need quite a bit of re-tooling for our purposes

import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastai.vision.all import *
import fastai

model_path = 'assets/models/PLCO_Fine_Tuned_120419.pth'

def change_model_path(path):
    global model_path
    model_path = path

def load_model():
    # construct base
    learn = vision_learner(
        DataLoaders.from_dsets([], [], bs=1), # dummy dataloader 
        fastai.vision.models.resnet34,
        n_out = 1,
        loss_func = MSELossFlat(), # dummy loss
    )

    # modify
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
    learn.path = Path('../assets/')
    learn.load('PLCO_Fine_Tuned_120419')
    learn.eval()

    learn.model = learn.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    return learn

def process(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet normalization
    ])
    image = transform(image).unsqueeze(0)
    image = image.to('cuda' if torch.cuda.is_available() else 'cpu')

    return image

def out_to_age(output):
    output = output * 8.03342449139388 + 63.8723890235948
    output = output * 6.75523 - 0.03771 * output * output - 213.77257

    return output