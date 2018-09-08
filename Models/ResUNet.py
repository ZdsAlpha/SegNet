import torch
import torch.nn as nn
import torchvision.models as models
F = nn.functional

class ResUNet(nn.Module):
    def __init__(self,in_channels,out_channels,depth=3,filters=8,kernel_size=3,padding=1,resLayers=2,convPerRes=2):
        super(ResUNet,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.resLayers = resLayers
        self.convPerRes = convPerRes

        encoders = []
        decoders = []
        for i in range(depth):
            if i == 0:
                encoder = self.createEncoderLayer(in_channels,self.calculateFilters(i),kernel_size,padding,resLayers,convPerRes)
                decoder = self.createDecoderLayer(self.calculateFilters(i),out_channels,kernel_size,padding,resLayers,convPerRes)
            else:
                encoder = self.createEncoderLayer(self.calculateFilters(i-1),self.calculateFilters(i),kernel_size,padding,resLayers,convPerRes)
                decoder = self.createDecoderLayer(self.calculateFilters(i),self.calculateFilters(i-1),kernel_size,padding,resLayers,convPerRes)
            encoders = encoders + [encoder]
            decoders = [decoder] + decoders
        center = nn.Sequential(*[
            self.createConvLayer(self.calculateFilters(depth-1),self.calculateFilters(depth-1),kernel_size,padding),
            self.createTransposeConvLayer(self.calculateFilters(depth-1),self.calculateFilters(depth-1),kernel_size,padding)
        ])
        final = self.createTransposeConvLayer(out_channels,out_channels,1,0)
        self.encoders = nn.Sequential(*encoders)
        self.center = center
        self.decoders = nn.Sequential(*decoders)
        self.final = final 

    def calculateFilters(self,depth):
        return self.filters * (2 ** depth)
    
    def createEncoderLayer(self,in_filters,out_filters,kernel_size=3,padding=1,resLayers=1,convPerRes=2):
        layer = nn.Sequential()
        layer.add_module("in",self.createConvLayer(in_filters,out_filters,kernel_size,padding))
        res = []
        for i in range(resLayers):
            resLayer = []
            for j in range(convPerRes):
                resLayer.append(self.createConvLayer(out_filters,out_filters,kernel_size,padding))
            res.append(nn.Sequential(*resLayer))
        layer.add_module("res", nn.Sequential(*res))
        return layer

    def createDecoderLayer(self,in_filters,out_filters,kernel_size,padding=1,resLayers=1,convPerRes=2):
        layer = nn.Sequential()
        layer.add_module("in",self.createTransposeConvLayer(2*in_filters,in_filters,kernel_size,padding))
        res = []
        for i in range(resLayers):
            resLayer = []
            for j in range(convPerRes):
                resLayer.append(self.createTransposeConvLayer(in_filters,in_filters,kernel_size,padding))
            res.append(nn.Sequential(*resLayer))
        layer.add_module("res",nn.Sequential(*res))
        layer.add_module("out",self.createTransposeConvLayer(in_filters,out_filters,kernel_size,padding))
        return layer
    
    def createConvLayer(self,in_filters,out_filters,kernel_size=3,padding=1,bn=True):
        convLayer = []
        convLayer.append(nn.Conv2d(in_filters,out_filters,kernel_size,padding=padding))
        if bn:
            convLayer.append(nn.BatchNorm2d(out_filters))
        return nn.Sequential(*convLayer)
    
    def createTransposeConvLayer(self,in_filters,out_filters,kernel_size=3,padding=1,bn=True):
        tconvLayer = []
        tconvLayer.append(nn.ConvTranspose2d(in_filters,out_filters,kernel_size,padding=padding))
        if bn:
            tconvLayer.append(nn.BatchNorm2d(out_filters))
        return nn.Sequential(*tconvLayer)
    
    def encode(self,x):
        tensors = []
        indices = []
        sizes = []
        for encoder in self.encoders:
            x = F.leaky_relu(encoder[0](x))
            last = None
            for res in encoder[1]:
                last = x
                for conv in range(len(res)):
                    if conv == len(res) - 1:
                        x = res[conv](x)
                        x = x + last
                        x = F.leaky_relu(x)
                    else:
                        x = F.leaky_relu(res[conv](x))
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
            x = F.leaky_relu(decoder[0](x))
            last = None
            for res in decoder[1]:
                last = x
                for conv in range(len(res)):
                    if conv == len(res) - 1:
                        x = res[conv](x)
                        x = x + last
                        x = F.leaky_relu(x)
                    else:
                        x = F.leaky_relu(res[conv](x))
            x = F.leaky_relu(decoder[2](x))
        return x
    
    def forward(self,x):
        x,tensors,indices,sizes = self.encode(x)
        x = self.center(x)
        x = self.decode((x,tensors,indices,sizes))
        return self.final(x)

    def initialize(self):
        for encoder in self.encoders:
            nn.init.xavier_normal_(encoder[0][0].weight)
            for res in encoder[1]:
                for conv in res:
                    nn.init.xavier_normal_(conv[0].weight)
        for layer in self.center:
            nn.init.xavier_normal_(layer[0].weight)
        for decoder in self.decoders:
            nn.init.xavier_normal_(decoder[0][0].weight)
            for res in decoder[1]:
                for conv in res:
                    nn.init.xavier_normal_(conv[0].weight)
            nn.init.xavier_normal_(decoder[2][0].weight)
        nn.init.xavier_normal_(self.final[0].weight)
