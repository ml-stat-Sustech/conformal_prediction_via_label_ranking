import numpy as np 
from sklearn.model_selection import train_test_split


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


def marginalCoverageSize(S,targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered += 1
        size = size + S[i].shape[0]
    return float(covered)/targets.shape[0], size/targets.shape[0]


def conditionalCoverageSize(S,targets):
    """ account size of the right prediction sets   """
    num = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            size = size + S[i].shape[0]
            num +=1
    return size/num





    
        
    
    
    
    
