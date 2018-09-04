import torch
from functions import loadingBar

def test(model,dataloader,criterion,device=None,onBatch=None,features_type=torch.FloatTensor,labels_type=torch.LongTensor):
    batch_id = 0
    total = len(dataloader)
    loadingBar(batch_id,total)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for features,labels in dataloader:
            if not torch.is_tensor(features):
                features = features_type(features)
            else:
                features = features.type(features_type)
            if not torch.is_tensor(labels):
                labels = labels_type(labels)
            else:
                labels = labels.type(labels_type)
            if device is not None:
                features,labels = features.to(device),labels.to(device)
            output = model(features)
            loss = criterion(output,labels)
            current_loss = loss.item()
            total_loss += current_loss
            batch_id += 1
            loadingBar(batch_id,total)
            if onBatch is not None:
                onBatch(batch_id,features,labels,output,current_loss)
        print()
    return total_loss

def eval(model,dataloader,device=None,onBatch=None,features_type=torch.FloatTensor,labels_type=torch.LongTensor):
    batch_id = 0
    total = len(dataloader)
    loadingBar(batch_id,total)
    model.eval()
    with torch.no_grad():
        for features in dataloader:
            if not torch.is_tensor(features):
                features = features_type(features)
            else:
                features = features.type(features_type)
            if device is not None:
                features = features.to(device)
            output = model(features)
            batch_id += 1
            loadingBar(batch_id,total)
            if onBatch is not None:
                onBatch(batch_id,features,output)
        print()