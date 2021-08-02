import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import  models
import copy



def initialize_resnet(hp,num_classes=5):

    model = None
    
    # load resnet models with pretrained or not~
    if hp['model'] == 'ResNet18':
        model = models.resnet18(pretrained=hp['pretrain'])
    else :
        assert hp['model'] == 'ResNet50', 'Invalid model name'
        model = models.resnet50(pretrained=hp['pretrain'])

    # turn off parameter updates if we are using feature extraction mode
    if hp['feature_extracting']:
        for p in model.parameters():
            p.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model



