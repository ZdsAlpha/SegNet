import torch
import torch.nn as nn
import torchvision.models as models
F = nn.functional

class UNet(nn.Module):
    def __init__(self,in_channels,out_channels,depth=4,init_filters=64,kernel_size=3,padding=1,convPerLayer=2,dropout=0):
        super(UNet,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.init_filters = init_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.convPerLayer = convPerLayer
        self.dropout = dropout
        encoders = []
        decoders = []
        for i in range(depth):
            if i == 0:
                encoder = self.createEncoderLayer(in_channels,self.calculateFilters(i),kernel_size,padding,convPerLayer)
                decoder = self.createDecoderLayer(self.calculateFilters(i),out_channels,kernel_size,padding,convPerLayer)
            else:
                encoder = self.createEncoderLayer(self.calculateFilters(i-1),self.calculateFilters(i),kernel_size,padding,convPerLayer)
                decoder = self.createDecoderLayer(self.calculateFilters(i),self.calculateFilters(i-1),kernel_size,padding,convPerLayer)
            encoders = encoders + [encoder]
            decoders = [decoder] + decoders
        encoders = nn.Sequential(*encoders)
        decoders = nn.Sequential(*decoders)
        center = nn.Sequential(*[
            self.createConvLayer(self.calculateFilters(depth-1),self.calculateFilters(depth-1),kernel_size,padding),
            self.createTransposeConvLayer(self.calculateFilters(depth-1),self.calculateFilters(depth-1),kernel_size,padding)
        ])
        final = self.createConvLayer(out_channels,out_channels,1,0)
        self.encoders = encoders
        self.center = center
        self.decoders = decoders
        self.final = final

    def calculateFilters(self,depth):
        return self.init_filters * (2 ** depth)
    
    def createEncoderLayer(self,in_filters,out_filters,kernel_size=3,padding=1,convs=2):
        layer = []
        for i in range(convs):
            if i == 0:
                layer.append(self.createConvLayer(in_filters,out_filters,kernel_size,padding))
            else:
                layer.append(self.createConvLayer(out_filters,out_filters,kernel_size,padding))
        return nn.Sequential(*layer)

    def createDecoderLayer(self,in_filters,out_filters,kernel_size=3,padding=1,convs=2):
        layer = []
        for i in range(convs):
            _in = in_filters
            _out = in_filters
            if i == 0:
                _in = 2 * in_filters
            if i == convs - 1:
                _out = out_filters
            layer.append(self.createTransposeConvLayer(_in,_out,kernel_size,padding))
        return nn.Sequential(*layer)

    def createConvLayer(self,in_filters,out_filters,kernel_size=3,padding=1,bn=True):
        convLayer = []
        convLayer.append(nn.Conv2d(in_filters,out_filters,kernel_size,padding=padding))
        if bn:
            convLayer.append(nn.BatchNorm2d(out_filters))
        return nn.Sequential(*convLayer)
    
    def createTransposeConvLayer(self,in_filters,out_filters,kernel_size=3,padding=1,bn=True):
        convLayer = []
        convLayer.append(nn.ConvTranspose2d(in_filters,out_filters,kernel_size,padding=padding))
        if bn:
            convLayer.append(nn.BatchNorm2d(out_filters))
        return nn.Sequential(*convLayer)

    def encode(self,x,activation=nn.LeakyReLU()):
        tensors = []
        indices = []
        sizes = []
        for encoder in self.encoders:
            for layer in encoder:
                x = activation(layer(x))
            sizes.append(x.size())
            tensors.append(x)
            x,ind = F.max_pool2d(x,2,2,return_indices=True)
            indices.append(ind)
        return (x,tensors,indices,sizes)
    
    def decode(self,x,activation=nn.LeakyReLU()):
        x,tensors,indices,sizes = x
        for decoder_id in range(len(self.decoders)):
            decoder = self.decoders[decoder_id]
            if self.training and self.dropout != 0 and decoder_id == len(self.decoders) - 1:
                x = F.dropout2d(x,self.dropout,self.training)
            tensor = tensors.pop(len(indices)-1)
            size = sizes.pop(len(indices)-1)
            ind = indices.pop(len(indices)-1)
            x = F.max_unpool2d(x,ind,2,2,output_size=size)
            x = torch.cat([tensor,x],dim=1)
            for layer in decoder:
                x = activation(layer(x))
        return x
    
    def forward(self,x,activation=nn.LeakyReLU(),final_activation=nn.Softmax2d()):
        x,tensors,indices,sizes = self.encode(x,activation=activation)
        for layer in self.center:
            x = activation(layer(x))
        x = self.decode((x,tensors,indices,sizes),activation=activation)
        return final_activation(self.final(x))
        
    def initialize(self,gain=1,std=0.02):
        for encoder in self.encoders:
            for layer in encoder:
                self.initialize_layer(layer,gain,std)
        for layer in self.center:
            self.initialize_layer(layer,gain,std)
        for decoder in self.decoders:
            for layer in decoder:
                self.initialize_layer(layer,gain,std)
        self.initialize_layer(self.final,gain,std)

    def initialize_layer(self,layer,gain=1,std=0.02):
        nn.init.xavier_normal_(layer[0].weight,gain)
        nn.init.normal_(layer[1].weight.data,1,0.02)
