import torch
import numpy as np

def getRawClasses(file):
    classes = []
    data = open(file, 'r').read()
    lines = lines = data.split('\n')
    for line in lines:
        if line == "":
            continue
        line = line.replace("		","\t")
        rgb,name = line.split('\t')
        r,g,b = rgb.split(' ')
        r,g,b = int(r),int(g),int(b)
        classes.append((name,(r,g,b)))
    return classes

def getClasses(file):
    classes = getRawClasses(file)
    for i in range(len(classes)):
        classes[i] = (classes[i][0], np.flip(np.asarray(classes[i][1]),0))
    return classes

def image_to_matrix(image_tensor,classes,device=0):
    with torch.no_grad():
        weights = [0] * classes
        image_tensor = image_tensor.to(device)
        width,height,channels = image_tensor.shape
        channels = torch.unbind(image_tensor,dim=2)
        output_tensor = torch.zeros(width,height,dtype=torch.int64,device=device)
        for class_id in range(len(classes)):
            c = classes[class_id]
            name,color_values = c
            indices = torch.ones(width,height,dtype=torch.uint8,device=device)
            for channel in range(len(channels)):
                comparison = torch.eq(channels[channel],float(color_values[channel]))
                indices = torch.mul(indices,comparison)
            weights[class_id] = torch.sum(indices).cpu().numpy().item()
            values = torch.mul(indices,class_id)
            output_tensor = output_tensor + values.to(torch.int64)
        return output_tensor,np.asarray(weights)

def images_to_matrices(images_tensor,classes,device=0):
    with torch.no_grad():
        weights = [0] * len(classes)
        images_tensor = images_tensor.to(device)
        batch_size,width,height,channels = images_tensor.shape
        channels = torch.unbind(images_tensor,dim=3)
        output_tensor = torch.zeros(batch_size,width,height,dtype=torch.int64,device=device)
        for class_id in range(len(classes)):
            c = classes[class_id]
            name,color_values = c
            indices = torch.ones(batch_size,width,height,dtype=torch.uint8,device=device)
            for channel in range(len(channels)):
                comparison = torch.eq(channels[channel],float(color_values[channel]))
                indices = torch.mul(indices,comparison)
            weights[class_id] = torch.sum(indices).cpu().numpy().item()
            values = torch.mul(indices,class_id)
            output_tensor = output_tensor + values.to(torch.int64)
        return output_tensor,np.asarray(weights) / float(batch_size)


def matrix_to_image(matrix_tensor,classes,device=0):
    with torch.no_grad():
        matrix_tensor = matrix_tensor.to(device)
        width,height = matrix_tensor.shape
        channels = [None] * len(classes[0][1])
        for channel in range(len(channels)):
            channel_tensor = torch.zeros(width,height,dtype=torch.uint8,device=device)
            for class_id in range(len(classes)):
                c = classes[class_id]
                name,color_values = c
                indices = torch.eq(matrix_tensor,float(class_id))
                values = torch.mul(indices,int(color_values[channel]))
                channel_tensor = channel_tensor + values
            channels[channel] = channel_tensor
        return torch.stack(channels,dim=2)

def matrices_to_images(matrices_tensor,classes,device=0):
    with torch.no_grad():
        matrices_tensor = matrices_tensor.to(device)
        batch_szie,width,height = matrices_tensor.shape
        channels = [None] * len(classes[0][1])
        for channel in range(len(channels)):
            channel_tensor = torch.zeros(batch_szie,width,height,dtype=torch.uint8,device=device)
            for class_id in range(len(classes)):
                c = classes[class_id]
                name,color_values = c
                indices = torch.eq(matrices_tensor,float(class_id))
                values = torch.mul(indices,int(color_values[channel]))
                channel_tensor = channel_tensor + values
            channels[channel] = channel_tensor
        return torch.stack(channels,dim=3)