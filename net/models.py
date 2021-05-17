import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule, MaskedLinear

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(120, 84)
        self.fc2 = linear(84, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3
        x = self.conv3(x)
        x = F.relu(x)

        # Fully-connected
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

import torch
import torch.nn as nn 
from .prune import PruningModule, MaskedLinear

vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG(PruningModule):
    def __init__(self, mask=False, in_channels=3, num_classes=10):
        super(VGG, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.in_channels = in_channels
        self.conv = self._make_layer(vgg16)
        self.fc = nn.Sequential(
			      linear(512*1*1, 4096),
			      nn.ReLU(),
			      nn.Dropout(p=0.5),
			      linear(4096, 4096),
			      nn.ReLU(),
			      nn.Dropout(p=0.5),
			      linear(4096, num_classes)
			  )

    def forward(self, x):
        x = self.conv(x)
        x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                    out_channels = x
                    layers += [
						              nn.Conv2d(
							            in_channels=in_channels,
							            out_channels=out_channels,
							            kernel_size=(3, 3),
							            stride=(1, 1),
							            padding=(1, 1),
					              	),
					      	        nn.BatchNorm2d(x),
						              nn.ReLU(),
					          ]
                    in_channels = x
            elif x == "M":
                    layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers) 
