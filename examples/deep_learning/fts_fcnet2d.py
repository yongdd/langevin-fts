import torch

class FtsNet2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 5
        padding = (kernel_size-1)//2
        in_channels = 1
        mid_channels = 128
        out_channels = 1
        
        self.conv1 = torch.nn.Conv2d(in_channels,  mid_channels, kernel_size, padding=padding, padding_mode='circular')
        self.conv2 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
        self.conv3 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
        self.conv4 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
        self.conv5 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
        self.conv6 = torch.nn.Conv2d(mid_channels, out_channels, 1)
        
        #self.bn1 = torch.nn.BatchNorm2d(mid_channels)
        #self.bn2 = torch.nn.BatchNorm2d(mid_channels)

    def forward(self, x):
        #x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        #x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = self.conv6(x)
        return x