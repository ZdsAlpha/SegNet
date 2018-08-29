import torch
import torch.nn as nn
import torchvision.models as models
F = nn.functional

class SegNet3D(nn.Module):
    def __init__(self,in_channels,out_channels,frames=4,depth=5,init_filters=32,kernel_size=3,padding=1):
        super(SegNet3D,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frames = frames
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
                decoder = self.createDecoderLayer(self.calculateFilters(i),out_channels,kernel_size,padding,layers)
            elif i == depth - 1:
                encoder = self.createEncoderLayer(self.calculateFilters(i-1),self.calculateFilters(i-1),kernel_size,padding,layers)
                decoder = self.createDecoderLayer(self.calculateFilters(i-1),self.calculateFilters(i-1),kernel_size,padding,layers)
            else:
                encoder = self.createEncoderLayer(self.calculateFilters(i-1),self.calculateFilters(i),kernel_size,padding,layers)
                decoder = self.createDecoderLayer(self.calculateFilters(i),self.calculateFilters(i-1),kernel_size,padding,layers)
            encoders = encoders + [encoder]
            decoders = [decoder] + decoders
        self.encoders = nn.Sequential(*encoders)
        self.decoders = nn.Sequential(*decoders)

    def calculateFilters(self,depth):
        return self.init_filters * (2 ** depth)

    def calculateConvLayers(self,depth):
        _convLayers=2
        _i = 0
        for i in range(depth):
            if _i == _convLayers:
                _i = 0
                _convLayers+=1
            _i += 1
        return _convLayers

    def createEncoderLayer(self,input_filters,output_filters,kernel_size=3,padding=1,convLayers=2):
        layer = []
        for i in range(convLayers):
            if i == 0:
                layer.extend([
                    nn.Conv3d(input_filters,output_filters,kernel_size,padding=padding),
                    nn.BatchNorm3d(output_filters),
                    nn.LeakyReLU()
                ])
            elif i == convLayers-1:
                layer.extend([
                    nn.Conv3d(output_filters,output_filters,kernel_size,padding=padding),
                    nn.BatchNorm3d(output_filters),
                    nn.LeakyReLU()
                ])
            else:
                layer.extend([
                    nn.Conv3d(output_filters,output_filters,kernel_size,padding=padding),
                    nn.BatchNorm3d(output_filters),
                    nn.LeakyReLU()
                ])
        return nn.Sequential(*layer)

    def createDecoderLayer(self,input_filters,output_filters,kernel_size=3,padding=1,convLayers=2):
        layer = []
        for i in range(convLayers):
            if i == 0:
                layer.extend([
                    nn.ConvTranspose3d(input_filters,input_filters,kernel_size,padding=padding),
                    nn.BatchNorm3d(input_filters),
                    nn.LeakyReLU()
                ])
            elif i == convLayers-1:
                layer.extend([
                    nn.ConvTranspose3d(input_filters,output_filters,kernel_size,padding=padding),
                    nn.BatchNorm3d(output_filters),
                    nn.Softmax(dim=1)
                ])
            else:
                layer.extend([
                    nn.ConvTranspose3d(input_filters,input_filters,kernel_size,padding=padding),
                    nn.BatchNorm3d(input_filters),
                    nn.LeakyReLU()
                ])
        return nn.Sequential(*layer)

    def encode(self,x):
        indices = []
        sizes = []
        for encoder in self.encoders:
            sizes.append(x.size())
            x,ind = F.max_pool3d(encoder(x),(1,2,2),(1,2,2),return_indices=True)
            indices.append(ind)
        return (x,indices,sizes)

    def decode(self,x):
        x,indices,sizes = x
        for decoder in self.decoders:
            size = sizes.pop(len(indices)-1)
            ind = indices.pop(len(indices)-1)
            x = decoder(F.max_unpool3d(x,ind,(1,2,2),(1,2,2),output_size=size))
        return x

    def forward(self,x):
        return self.decode(self.encode(x))

    def initialize(self):
        for part in [self.encoders,self.decoders]:
            for layer in part:
                for i in range(len(layer)):
                    if i%3==0:
                        nn.init.xavier_normal_(layer[i].weight)
