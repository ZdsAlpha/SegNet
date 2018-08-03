import torch

def test(model,dataloader,criterion,device=None,callback=None):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for images,labels in dataloader:
            if device is not None:
                images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss = criterion(output,labels)
            total_loss += loss.item()
            if callback is not None:
                callback(output,loss)

