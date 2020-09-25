# Importing the libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
from torchvision import datasets
import torch.nn as nn
import torchvision.models as models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import splitfolders
import torchvision.transforms as transforms

# Importing the dataset
input_folder = 'PlantVillage'
output = 'split'

# Splitting the datset into train and test directories
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.75,.25))

# Setting the train and test directories
train_dir = r"C:\Users\Eshika\Downloads\plant dataset\split\train"
test_dir = r"C:\Users\Eshika\Downloads\plant dataset\split\val"

# Initializing the transforms
data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]) ]),
                   'test': transforms.Compose([transforms.Resize(size=(224,224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
                  }
