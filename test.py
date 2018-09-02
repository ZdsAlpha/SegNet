import torch
from functions import loadingBar

def test(model,dataloader,criterion,device=None,callback=None,images_type=torch.FloatTensor,labels_type=torch.LongTensor):
    count = 0
    total = len(dataloader)
    loadingBar(count,total)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for images,labels in dataloader:
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
            output = model(images)
            loss = criterion(output,labels)
            total_loss += loss.item()
            count += 1
            loadingBar(count,total)
            if callback is not None:
                callback(output,loss)
        print()
    return total_loss

def eval(model,dataloader,device=None,callback=None,images_type=torch.FloatTensor,labels_type=torch.LongTensor):
    count = 0
    total = len(dataloader)
    loadingBar(count,total)
    model.eval()
    with torch.no_grad():
        for images in dataloader:
            if not torch.is_tensor(images):
                images = images_type(images)
            else:
                images = images.type(images_type)
            if device is not None:
                images = images.to(device)
            output = model(images)
            count += 1
            loadingBar(count,total)
            if callback is not None:
                callback(output)
        print()