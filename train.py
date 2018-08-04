import torch

def train(model,datalaoder,criterion,optimizer,epochs=1,device=None,callback=None):
    loss_sum=0
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for images,labels in datalaoder:
            if device is not None:
                images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            del images,labels,output
        if callback is not None:
            callback(epoch,total_loss)
    return loss_sum / epochs