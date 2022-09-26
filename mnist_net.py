import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8,kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(in_features = 16*4*4, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 64)
        self.fc3 = nn.Linear(in_features = 64, out_features = 10)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x)) 
        #print (x.shape)      # First convolution followed by
        x = self.pool(x)  
        #print(x.shape)              # a relu activation and a max pooling#
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)       
        x = self.fc3(x)
        #print(x.shape)
        return x

    def get_features(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)
            return x



if __name__=='__main__':
    x = torch.rand(16,1,28,28)
    net = MNISTNet()
    y = net(x)
    assert y.shape == (16,10)
