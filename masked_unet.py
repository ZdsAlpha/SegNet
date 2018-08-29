import torch

def train(model,images,in_masks,out_masks,epochs=10,weights=None,device=0,lr=0.001,onTrainEpoch=None):
    assert len(images) == len(in_masks) == len(out_masks)
    if weights is None:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        weights = torch.FloatTensor(weights).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr)
    
    return train(model,train_loader,criterion,optimizer,args.epochs,args.device,onTrainEpoch)