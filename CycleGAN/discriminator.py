import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride = 2):
        super(CNNBlock,self).__init__() 
        #super() ensures that the CNNBlock class is properly initialized as a subclass of nn.Module
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,1,bias=True,padding_mode='reflect'), 
            #4 is kernel_size and 1 is padding
            #reflect means the input is padded using the reflection of the input boundary.
            nn.InstanceNorm2d(out_channels),
            #normalizes each channel of each sample in a batch independently
            nn.LeakyReLU(0.2,inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self,in_channels = 3,features = [64,128,256,512]):
        #Features will be the features of each layer
        super(Discriminator,self).__init__()
        self.initial = nn.Sequential(
            #We use here the first feature
            nn.Conv2d(in_channels,out_channels=features[0],kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2,inplace=True)
        )
        layers = []
        #We will store in layers the main layers of the discriminator
        in_channels = features[0]
        for feature in features[1:]:
            #We use the remaining features for each layer
            layers.append(CNNBlock(in_channels,out_channels=feature,stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode='reflect'))
        self.model = nn.Sequential(*layers)
        #*layers unpacks the list into individual arguments
    
    def forward(self,x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()