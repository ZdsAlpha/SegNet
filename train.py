import torch
import numpy as np
from functions import loadingBar

def train(model,datalaoder,criterion,optimizer,epochs=1,device=None,onBatch=None,onEpoch=None,features_type=torch.FloatTensor,labels_type=torch.LongTensor):
    loss_sum=0
    for epoch in range(epochs):
        batch_id = 0
        total = len(datalaoder)
        loadingBar(batch_id,total)
        total_loss = 0
        model.train()
        for features,labels in datalaoder:
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
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            batch_id += 1
            if onBatch is not None:
                onBatch(batch_id,features,labels,output,current_loss)
            loadingBar(batch_id,total)
            del features,labels,output
        print()
        if onEpoch is not None:
            onEpoch(epoch+1,total_loss)
        loss_sum += total_loss
    return loss_sum / epochs