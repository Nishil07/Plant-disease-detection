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

# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False


n_inputs = vgg16.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
vgg16.classifier[6] = last_layer

# check to see that your last layer produces the expected number of outputs
'''print(vgg16.classifier[6].out_features)'''


# specify loss function (categorical cross-entropy) 
# specify optimizer (stochastic gradient descent) and learning rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 21

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    
    for batch_i, (data, target) in enumerate(train_loader):
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = vgg16(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss 
        train_loss += loss.item()
        
        if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
            print('Epoch %d, Batch %d loss: %.16f' %
                  (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0

def test(loaders, model, criterion):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders):
        # move to GPU
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

test(train_loader, vgg16, criterion)

joblib.dump(vgg16,'model.pkl')
            
