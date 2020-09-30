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

# Loading the train and test images into train_data and test_data respectively
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

# Defining VGG16 model
vgg16 = models.vgg16(pretrained=True)

classes = ['Pepper__bell___Bacterial_spot', 
           'Pepper__bell___healthy', 
           'Potato___Early_blight', 
           'Potato___healthy', 
           'Potato___Late_blight',
           'Tomato__Target_Spot',
           'Tomato__Tomato_mosaic_virus',
           'Tomato__Tomato_YellowLeaf__Curl_Virus',
           'Tomato_Bacterial_spot',
           'Tomato_Early_blight',
           'Tomato_healthy',
           'Tomato_Late_blight',
           'Tomato_Leaf_Mold',
           'Tomato_Septoria_leaf_spot',
           'Tomato_Spider_mites_Two_spotted_spider_mite']

# define dataloader parameters
batch_size = 20
num_workers=0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)
