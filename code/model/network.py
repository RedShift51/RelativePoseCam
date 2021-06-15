import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

class featureExtractor(nn.Module):
    def __init__(self):
        super(featureExtractor, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True).cuda()
        self.resnet34 = nn.Sequential(*(list(self.resnet34.children())[:-2]))

    def forward(self, x):
        x1 = self.resnet34(x)

        return x1#torch.cat((x1, x2), axis=1)

class dataMatcher(nn.Module):
    def __init__(self):
        super(dataMatcher, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, 3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(512, 256, 3, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 128, 3, stride=2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 6, 1)

        self.act_out = nn.Tanh()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y) 
        y = self.conv3(y)
        y = self.relu3(y)
        y = F.max_pool2d(y, y.size()[2:])
        y = self.conv4(y)
        y = y.view(-1, 6)

        return 2*np.pi*self.act_out(y[:, :3]), y[:, 3:]


class wholeNN(nn.Module):
    def __init__(self):
        super(wholeNN, self).__init__()
        self.featExtr = featureExtractor()
        self.head = dataMatcher()

    def forward(self, x):
        y = self.featExtr(x)
        y = self.head(y)

        return y


