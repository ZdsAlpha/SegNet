import torch
import torch.nn as nn
import torchvision.models as models
F = nn.functional

class UNet(nn.Module):
    def __init__(self,in_channels,out_channels,depth=4,init_filters=64,kernel_size=3,padding=1):
        super(UNet,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.init_filters = init_filters
        self.kernel_size = kernel_size
        self.padding = padding
        encoders = []
        decoders = []
        for i in range(depth):
            layers = self.calculateConvLayers(i+1)
            if i == 0:
                encoder = self.createEncoderLayer(in_channels,self.calculateFilters(i),kernel_size,padding,layers)
                decoder = self.createDecoderLayer(self.calculateFilters(i),out_channels,kernel_size,padding,layers+1)
            else:
                encoder = self.createEncoderLayer(self.calculateFilters(i-1),self.calculateFilters(i),kernel_size,padding,layers)
                decoder = self.createDecoderLayer(self.calculateFilters(i),self.calculateFilters(i-1),kernel_size,padding,layers)
            encoders = encoders + [encoder]
            decoders = [decoder] + decoders
        center = nn.Sequential(*[
            self.createEncoderLayer(self.calculateFilters(depth-1),self.calculateFilters(depth),kernel_size,padding,1),
            self.createDecoderLayer(self.calculateFilters(depth-1),self.calculateFilters(depth),kernel_size,padding,1)
        ])
        self.encoders = nn.Sequential(*encoders)
        self.center = center
        self.decoders = nn.Sequential(*decoders)

    def calculateFilters(self,depth):
        return self.init_filters * (2 ** depth)

    def calculateConvLayers(self,depth):
        return 2

    def createEncoderLayer(self,input_filters,output_filters,kernel_size=3,padding=1,convLayers=2):
        layer = []
        for i in range(convLayers):
            if i == 0:
                layer.extend([
                    nn.Conv2d(input_filters,output_filters,kernel_size,padding=padding),
                    nn.BatchNorm2d(output_filters),
                    nn.LeakyReLU()
                ])
            elif i == convLayers-1:
                layer.extend([
                    nn.Conv2d(output_filters,output_filters,kernel_size,padding=padding),
                    nn.BatchNorm2d(output_filters),
                    nn.LeakyReLU()
                ])
            else:
                layer.extend([
                    nn.Conv2d(output_filters,output_filters,kernel_size,padding=padding),
                    nn.BatchNorm2d(output_filters),
                    nn.LeakyReLU()
                ])
        return nn.Sequential(*layer)

    def createDecoderLayer(self,input_filters,output_filters,kernel_size=3,padding=1,convLayers=2):
        layer = []
        for i in range(convLayers):
            if i == 0:
                layer.extend([
                    nn.ConvTranspose2d(2*input_filters,input_filters,kernel_size,padding=padding),
                    nn.BatchNorm2d(input_filters),
                    nn.LeakyReLU()
                ])
            elif i == convLayers-1:
                layer.extend([
                    nn.ConvTranspose2d(input_filters,output_filters,kernel_size,padding=padding),
                    nn.BatchNorm2d(output_filters),
                    nn.Softmax(dim=1)
                ])
            else:
                layer.extend([
                    nn.ConvTranspose2d(input_filters,input_filters,kernel_size,padding=padding),
                    nn.BatchNorm2d(input_filters),
                    nn.LeakyReLU()
                ])
        return nn.Sequential(*layer)

    def encode(self,x):
        tensors = []
        indices = []
        sizes = []
        for encoder in self.encoders:
            x = encoder(x)
            sizes.append(x.size())
            tensors.append(x)
            x,ind = F.max_pool2d(x,2,2,return_indices=True)
            indices.append(ind)
        return (x,tensors,indices,sizes)

    def decode(self,x):
        x,tensors,indices,sizes = x
        for decoder in self.decoders:
            tensor = tensors.pop(len(indices)-1)
            size = sizes.pop(len(indices)-1)
            ind = indices.pop(len(indices)-1)
            x = F.max_unpool2d(x,ind,2,2,output_size=size)
            x = torch.cat([tensor,x],dim=1)
            x = decoder(x)
        return x

    def forward(self,x):
        x,tensors,indices,sizes = self.encode(x)
        x = self.center(x)
        return self.decode((x,tensors,indices,sizes))

    def initialize(self):
        for part in [self.encoders,self.center,self.decoders]:
            for layer in part:
                for i in range(len(layer)):
                    if i%3==0:
                        nn.init.xavier_normal_(layer[i].weight)
