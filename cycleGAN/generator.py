import torch
import torch.nn as nn
        
        
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_downsampling: bool = True,
        add_activation: bool = True,
        **kwargs
    ):
        '''Creates a convolutional block, with option to be applied on encoder or decoder part, depending if downsampling is used or not
            
            Flags:
                is_downsampling (bool)     -- whether downsampling (True) or upsampling (False)
                add_activation (bool)      -- use activation or not
                **kwargs                   -- allows us to provide any additional keyword arguments that are accepted by the nn.Conv2d or nn.ConvTranspose2d (padding, stride, etc)
        '''
        super().__init__()
        if is_downsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(
        self, 
        channels: int
    ):

        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, add_activation=True, kernel_size=3, padding=1),
            ConvBlock(channels, channels, add_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self, 
        img_channels: int, 
        num_features: int = 64, 
        num_residuals: int = 9
    ):

        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

        self.downsampling_layers = nn.ModuleList(
            [
                ConvBlock(
                    num_features, 
                    num_features * 2,
                    is_downsampling=True, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.residual_layers = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.upsampling_layers = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last_layer = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = self.residual_layers(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        return torch.tanh(self.last_layer(x))


    def get_features(self, all=False):     # Used for fine-tuning on small dataset after training with bigger dataset
        return nn.Sequential(
                            self.initial_layer,
                            *self.downsampling_layers,
                            self.residual_layers,
                            *self.upsampling_layers
                        ) if not all else nn.Sequential(
                                                        self.initial_layer,
                                                        *self.downsampling_layers,
                                                        self.residual_layers,
                                                        *self.upsampling_layers,
                                                        self.last_layer,
                                                        nn.Tanh(),
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
                            nn.Tanh() if last_layer else nn.Identity(),
                        )
        
        return cloned_layer

def test():

    """ Just used to test some features, not applied to training of the model """

    img_channels = 3
    img_size = 512
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, num_features=64, num_residuals=9)
    print(gen(x).shape)

    new_last_layer = gen.clone_layer(gen.last_layer, last_layer=True)
    print(new_last_layer)
    print(type(new_last_layer))

    # Compare the weights
    for param1, param2 in zip(gen.last_layer.parameters(), new_last_layer.parameters()):
        if torch.equal(param1.data, param2.data):
            print("Weights are the same")
        else:
            print("Weights are different")

if __name__ == "__main__":
    test()