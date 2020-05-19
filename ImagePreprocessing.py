from __future__ import print_function, division
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib.ticker as mtick
from scipy.integrate import cumtrapz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io, transform, color
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler, WeightedRandomSampler
import time




### Add integer label to the dataframe
def add_label(df1,df2):

    '''
    Attach a new column named Label to the dataframe:
    integer labels of the images according to their classes (0 - 782);

    Args:
        df (pandas.Dataframe): the original df without int labels
    Return:
        df (pandas.Dataframe): the new df with int labels
    '''
    classes = pd.Series(df1['scientific_name'].value_counts().keys(), index=[_ for _ in range(len(df1['scientific_name'].value_counts().keys()))])
    labels1 = []
    labels2 = []
    for i in range(len(df1)):
        class_name = df1['scientific_name'][i]
        #print(class_name,classes,classes[classes == class_name])
        labels1.append(classes[classes == class_name].index[0])
    for i in range(len(df2)):
        class_name = df2['scientific_name'][i]
        #print(class_name,classes,classes[classes == class_name])
        labels2.append(classes[classes == class_name].index[0])
    df1['Label'] = labels1
    df2['Label'] = labels2
    data_address = "data/"
    df1.to_csv(os.path.join(data_address,'training_data_with_labels.csv'))
    df2.to_csv(os.path.join(data_address,'val_data_with_labels.csv'))
    return df


### Preprocessing
class preprocessing():
    def __init__(self, size=(200,200)):
        self.size = size # Resize
        self.transform = transforms.Compose([transforms.ToTensor()])   # Add more steps to achieve augmentation

    def new_image(self,image):
        image = transform.resize(image,self.size)
        image = self.transform(image)
        return image



### Create Custom Dataset
class SnakeDataSet(Dataset):
    def __init__(self, filename, image_path, preprocess=preprocessing()):
        '''
        Args:
            filename (str): filename of the csv that stores the pd.dataframe with the int labels
            image_path (str): the path of the folder where images are stored
            preprocess (bool): Whether to preprocess on image
        '''
        self.data = pd.read_csv(filename)    # The
        self.id = self.data['hashed_id']       # The hashed ids of the images
        self.labels = self.data['Label']     # The int labels of the images
        self.path = image_path               # The directory the images are
        self.preprocess = preprocess           # The transform performed on the images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #(idx)
        #print(self.id[idx])
        image = io.imread(os.path.join(self.path,(self.id[idx]+'.jpg')))
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        label = self.labels[idx]
        if self.preprocess:
            image = self.preprocess.new_image(image)
        sample = (image, label)
        return sample



### CNN class
class BasicNet(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(6,6), stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), stride=1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4232, 2000)
        self.fc2 = nn.Linear(2000, 783)
        self.bn = nn.BatchNorm2d(8)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout1(x)
        output = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output
    
### Function to train the Net
def train(model, device, train_loader, fraction, optimizer, epoch, log_interval = 100):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    train_length = int(len(train_loader.dataset) / fraction)
    model.train()   # Set the model to training mode
    ts = time.time()  # Starting time
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data.float())                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step

        if batch_idx % log_interval == 0:

            # Report each 100 batch
            f=open('running.txt','a')
            message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), train_length,
                100. * batch_idx *len(data) / train_length, loss.item())
            f.write(message)
            print(message)
            te = time.time()                             # ending time of 100 batchs
            message = 'Time Elapsed: %.i s.\n' % (int(te - ts))
            f.write(message)
            print(message)
            f.close()
            ts = time.time()                             # restart Timer

### Test function
def test(model, device, test_loader, fraction):

    correct = 0
    test_loss = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data0, target0 in test_loader:
            data, target = data0.to(device), target0.to(device)
            output = model(data.float())  
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_length = int(len(test_loader.dataset) / fraction)
    test_loss /= test_length

    # Report each epoch
    f = open('running.txt', 'a')
    message = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_length,
        100. * correct / test_length)
    f.write(message)
    f.close()
    print(message)
        
    return test_loss

