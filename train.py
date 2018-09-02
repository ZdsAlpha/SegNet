import torch
import numpy as np
from functions import loadingBar

def train(model,datalaoder,criterion,optimizer,epochs=1,device=None,callback=None,images_type=torch.FloatTensor,labels_type=torch.LongTensor):
    loss_sum=0
    for epoch in range(epochs):
        count = 0
        total = len(datalaoder)
        loadingBar(count,total)
        total_loss = 0
        model.train()
        for images,labels in datalaoder:
            if not torch.is_tensor(images):
                images = images_type(images)
            else:
                images = images.type(images_type)
            if not torch.is_tensor(labels):
                labels = labels_type(labels)
            else:
                labels = labels.type(labels_type)
            if device is not None:
                images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
            loadingBar(count,total)
            del images,labels,output
        print()
        if callback is not None:
            callback(epoch+1,total_loss)
        loss_sum += total_loss
    return loss_sum / epochs