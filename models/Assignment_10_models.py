import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.1

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Convolution Block 1 - Input Block
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # receptive field = 

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = 1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # receptive field = 

        self.r1 = ResidualBlock(128,128)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = 1, bias = False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3,3), padding = 1, bias = False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        self.r2 = ResidualBlock(512,512)

        self.max_pool = nn.MaxPool2d(kernel_size=4)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        ) 
        
        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.prep_layer(x)
        out = self.layer1(x)
        out = self.r1(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.r2(out)
        out = self.dropout(out)
        out = self.max_pool(out)
        out = self.fc(out)
        out = out.view(-1, 10)
        return F.log_softmax(out, dim=-1)