import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pathlib
import os
import pickle
from tqdm import tqdm
import pdb
import torchvision.transforms as trn
def sort_sum(scores):
    
    I = scores.argsort(axis=1)[:,::-1]
    
    ordered = np.sort(scores,axis=1)[:,::-1]
    
    cumsum = np.cumsum(ordered,axis=1) 
    return I, ordered, cumsum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def validate(val_loader, model, print_bool):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            if output.shape[1]<5:
                large_k=1
            else:
                large_k =5
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, large_k))
            cvg, sz = coverage_size(S, target)

            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    if print_bool:
        print('') #Endline

    return top1.avg, top5.avg, coverage.avg, size.avg 

def coverage_size(S,targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered += 1
        size = size + S[i].shape[0]
    return float(covered)/targets.shape[0], size/targets.shape[0]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def data2tensor(data):
    imgs = torch.cat([x[0].unsqueeze(0) for x in data], dim=0).cuda()
    targets = torch.cat([torch.Tensor([int(x[1])]) for x in data], dim=0).long()
    return imgs, targets

def split2ImageFolder(path, transform, n1, n2):
    dataset = torchvision.datasets.ImageFolder(path, transform)
    data1, data2 = torch.utils.data.random_split(dataset, [n1, len(dataset)-n1])
    data2, _ = torch.utils.data.random_split(data2, [n2, len(dataset)-n1-n2])
    return data1, data2

def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(dataset, [n1, dataset.tensors[0].shape[0]-n1])
    data2, _ = torch.utils.data.random_split(temp, [n2, dataset.tensors[0].shape[0]-n1-n2])
    # data2, _ = torch.utils.data.random_split(temp, [n2, dataset.tensors[0].shape[0]-n1-n2],torch.Generator())
    return data1, data2,data1.indices,np.array(temp.indices)[np.array(data2.indices)].tolist()

# Computes logits and targets from a model and loader
def get_logits_targets_CLIP(model,datasetname, dataset,num_classes):
    print(f'Computing logits for model (only happens once).')
    
    from models import clip
    import json
    if datasetname == "imagenet":
        usr_dir = os.path.expanduser('~')
        data_dir = os.path.join(usr_dir,"data","imagenet")
        with open(os.path.join(data_dir,'human_readable_labels.json')) as f:
            readable_labels = json.load(f)
    else:
        readable_labels = dataset.classes
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in readable_labels]).cuda()
    model, preprocess = model[0],model[1]
    # Calculate features
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for  index in range(len(dataset)):
            tmp_x, tmp_label = dataset[index]  
            tmp_x = preprocess(tmp_x).unsqueeze(0)
            image_features = model.encode_image(tmp_x.cuda()) 
            image_features /= image_features.norm(dim=-1, keepdim=True)
            tmp_logits = (100.0 * image_features @ text_features.T)
            tmp_logits = tmp_logits.detach().cpu()
            logits_list.append(tmp_logits)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return dataset_logits

# Computes logits and targets from a model and loader
def get_logits_targets(model, loader,num_classes,modelname=""):
    print(f'Computing logits for model (only happens once).')
    
    
    
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for  examples in tqdm(loader):
            tmp_x, tmp_label = examples[0], examples[1]            
            tmp_logits = model(tmp_x.cuda()).detach().cpu()
            logits_list.append(tmp_logits)
            labels_list.append(tmp_label)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
    
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    
    
    
    return dataset_logits


from models.connetor import build_common_model
from datasets.connector import build_dataset


def check_transform(model_name,dataset_name):
    if model_name == "ViT" or model_name == "Inception":
        if dataset_name == "cifar10":
            mean = (0.492, 0.482, 0.446)
            std = (0.247, 0.244, 0.262)
            transform = trn.Compose([transforms.Resize(224),
                                    trn.ToTensor(), 
                                  trn.Normalize(mean, std)])
            return transform
        elif dataset_name =="cifar100":
            CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            if model_name == "ViT":
                transform = trn.Compose([
                                        transforms.Resize(224),
                                        trn.ToTensor(), 
                                    trn.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
            elif model_name == "Inception":
                transform = trn.Compose([
                                        trn.ToTensor(), 
                                    trn.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
            return transform
        else:
            return None
    else:
        return None

def get_logits_dataset(modelname, datasetname):
    
    cache = os.path.join(os.getcwd(),"data",datasetname,'pkl')
    if not os.path.exists(cache):
        os.mkdir(cache)
    fname = cache +'/' + modelname + '.pkl' 
    
    # If the file exists, load and return it.
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle)
    if modelname == "CLIP":
        transform = lambda x:x
        dataset,num_classes = build_dataset(datasetname,"test",transform)
    else:
        transform = check_transform(modelname,datasetname)
        dataset,num_classes = build_dataset(datasetname,"test",transform)
    model = build_common_model(modelname,datasetname)

    
    if modelname == "CLIP":
        # Get the logits and targets
        dataset_logits = get_logits_targets_CLIP(model, datasetname,dataset, num_classes)
    else:
        # Get the logits and targets
        loader = torch.utils.data.DataLoader(dataset, batch_size = 320, shuffle=False, pin_memory=True)
        dataset_logits = get_logits_targets(model, loader,num_classes)

    # Save the dataset 
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as handle:
        pickle.dump(dataset_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    

    return dataset_logits

