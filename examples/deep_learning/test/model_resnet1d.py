import torch

class Bottleneck(torch.nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.block = torch.nn.Sequential(
            #torch.nn.BatchNorm2d(channels), #
            torch.nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, padding_mode='circular'), 
            torch.nn.ReLU(), 
            torch.nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, padding_mode='circular')
        )

    def forward(self, x):
        out = self.block(x)
        x = torch.nn.functional.relu(x + out)
        return x

class FtsResNet1d(torch.nn.Module):
    def __init__(self):
        super().__init__()

        kernel_size = 5
        padding = (kernel_size-1)//2
    
        in_channels = 1
        mid_channels = 128
        out_channels = 1
        
        self.conv1 = torch.nn.Conv1d(in_channels,  mid_channels, kernel_size, padding=padding, padding_mode='circular')
        self.conv2 = self.make_layers(mid_channels, kernel_size, padding, 20)
        self.conv3 = torch.nn.Conv1d(mid_channels, out_channels, 1)

    def make_layers(self, channels, kernel_size, padding, repeat):
        layers = []
        for i in range(0, repeat):
            layers.append(Bottleneck(channels, kernel_size, padding))
        return torch.nn.Sequential(*layers)
        
    def forward(self, x): 
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return x
