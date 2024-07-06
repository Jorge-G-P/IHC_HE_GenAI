import torch
import torch.nn as nn

class ConvInstanceNormLeakyReLUBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        Class object initialization for Convolution-InstanceNorm-LeakyReLU layer

        We use leaky ReLUs with a slope of 0.2.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        """
        Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with 
        k filters and stride 2. Discriminator architecture is: C64-C128-C256-C512. 
        
        After the last layer, we apply a convolution to produce a 1-dimensional 
        output. 
        
        We use leaky ReLUs with a slope of 0.2.
        """
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                ConvInstanceNormLeakyReLUBlock(
                    in_channels, 
                    feature, 
                    stride=1 if feature == features[-1] else 2,
                )
            )
            in_channels = feature

        # After the last layer, we apply a convolution to produce a 1-dimensional output 
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        # feed the model output into a sigmoid function to make a 1/0 label
        return torch.sigmoid(self.model(x))


    def get_features(self):     # Used for fine-tuning on small dataset after training with bigger dataset
        return nn.Sequential(
            self.initial_layer,
            *self.model[:-1]
        )

    @staticmethod
    def clone_layer(layer, last_layer=True):

        cloned_layer = nn.Sequential(
                            type(layer)(
                                    in_channels=layer.in_channels,
                                    out_channels=layer.out_channels,
                                    kernel_size=layer.kernel_size,
                                    stride=layer.stride,
                                    padding=layer.padding,
                                    padding_mode=layer.padding_mode
                            ),
                            nn.Sigmoid() if last_layer else nn.Identity(),
                        )
        
        return cloned_layer
    
def test():  
    
    """ Just used to test some features, not applied to training of the model """

    x = torch.randn((5, 3, 512, 512))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)
    
    new_last_layer = model.clone_layer(model.model[-1], last_layer=False)
    print(new_last_layer)

    
if __name__ == "__main__":
    test()