
from .utils import sort_sum,validate
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import pandas as pd
import time
from tqdm import tqdm
import pdb
from scipy.stats import kstest
import joblib

import matplotlib.pyplot as plt

    
    


def conformal_calibration_logits(cmodel, calib_loader,cal_softmax=True,randomized=True,allow_zero_sets=True):
    with torch.no_grad():
        E = np.array([])
        targets_list= []
        logits_list = []
        for logits, targets in calib_loader:
            logits = logits.detach().cpu().numpy()
            targets_list.append(targets.detach().cpu().numpy())
            logits_list.append(logits)
            if cal_softmax==True:
                scores = softmax(logits, axis=1)
            else:
                scores = logits

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum,penalties=cmodel.penalties,randomized=randomized,allow_zero_sets=allow_zero_sets)))
            
        Qhat = np.quantile(E,1-cmodel.alpha,method='higher')
        

    return Qhat,E

def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 


### CORE CONFORMAL INFERENCE FUNCTIONS

# Generalized conditional quantile function.
def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):

    # ordered += 1e-7
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1]) # 1-1000
    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i,sizes_base[i]-1] * \
                    (tau-(cumsum[i,sizes_base[i]-1]-ordered[i,sizes_base[i]-1])-penalties_cumsum[0,sizes_base[i]-1]) # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1] # always predict max size if alpha==0. (Avoids numerical error.)

    if not allow_zero_sets:
        sizes[sizes == 0] = 1 # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i,0:sizes[i]],]

    return S

# Get the 'p-value'
def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets): # For one example
    idx = np.where(I==target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]
    
    U = np.random.random()
    if idx == (0,0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0] 
    else:
        if idx[1][0]==cumsum.shape[1]:
            return U * ordered[idx] + cumsum[(idx[0],idx[1]-1)] + (penalty[0:(idx[1][0])]).sum()
        else:
            return U * ordered[idx] + cumsum[(idx[0],idx[1]-1)] + (penalty[0:(idx[1][0]+1)]).sum()

# Gets the histogram of Taus. 
def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        if penalties.shape[0]==1:
            E[i] = get_tau(scores[i:i+1,:],targets[i].item(),I[i:i+1,:],ordered[i:i+1,:],cumsum[i:i+1,:],penalties[0,:],randomized=randomized, allow_zero_sets=allow_zero_sets)
        else:
            E[i] = get_tau(scores[i:i+1,:],targets[i].item(),I[i:i+1,:],ordered[i:i+1,:],cumsum[i:i+1,:],penalties[i,:],randomized=randomized, allow_zero_sets=allow_zero_sets)
    return E







    
    
class SAPS(nn.Module):

    def __init__(self, calib_loader, alpha, rank_pen=None,kreg=None,lamda=None, randomized=True, allow_zero_sets=True,batch_size=32,pct_paramtune=0.33, rank_pen_criterion='size'):
        super(SAPS, self).__init__()
        self.alpha = alpha
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        

        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        
        if rank_pen == None:
            if rank_pen_criterion == "size":
                rank_pen,paramtune_logits, calib_logits = self.pick_parameters_rank( calib_loader.dataset, alpha, self.randomized , self.allow_zero_sets, pct_paramtune, batch_size, "size")
            else:
                rank_pen,paramtune_logits, calib_logits = self.pick_parameters_rank_adaptiveness( calib_loader.dataset, alpha, self.randomized , self.allow_zero_sets, pct_paramtune, batch_size, "adaptiveness")
            
            
            calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
            
        
        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        
        self.rank_pen = rank_pen 
        
        self.Qhat,self.E = self.conformal_calibration_logits(calib_loader,self.randomized,self.allow_zero_sets)
        


    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        
        with torch.no_grad():
            scores = logits.detach().cpu().numpy()
            


            I, _, _ = sort_sum(scores)
            ordered= []
            cumsum= []
            for i in range(scores.shape[0]):
                scores[i] = self.transform(scores[i])
                score_orded = scores[i,I[i]]
                score_cusum = np.cumsum(score_orded)
                ordered.append(score_orded.reshape(1,-1))
                cumsum.append(score_cusum.reshape(1,-1))
            ordered = np.concatenate(ordered)
            cumsum = np.concatenate(cumsum)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties[:,:scores.shape[1]], randomized=self.randomized, allow_zero_sets=self.allow_zero_sets)
            
        return logits, S
    
    
    def conformal_calibration_logits(self, calib_loader,randomized,allow_zero_sets):
        with torch.no_grad():
            E = np.array([])
            targets_list=[]
            logits_list = []
            for logits, targets in calib_loader:
                logits = logits.detach().cpu().numpy()
                scores = logits.copy()
                logits_list.append(logits)
                targets = targets.detach().cpu().numpy()
                targets_list.append(targets)

                I, _, _ = sort_sum(logits)
                ordered= []
                cumsum= []
                for i in range(scores.shape[0]):
                    scores[i] = self.transform(scores[i])
                    score_orded = scores[i,I[i]]
                    score_cusum = np.cumsum(score_orded)
                    ordered.append(score_orded.reshape(1,-1))
                    cumsum.append(score_cusum.reshape(1,-1))
                ordered = np.concatenate(ordered)
                cumsum = np.concatenate(cumsum)

                E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum,penalties=self.penalties,randomized=randomized,allow_zero_sets=allow_zero_sets)))
            Qhat = np.quantile(E,1-self.alpha,method='higher')

        return Qhat,E
    
    


    
    def transform(self,score):#

        # k=1
        score = softmax(score)
        arg_max = np.argmax(score)
        max_calue = np.max(score)
        score[np.arange(score.shape[0]) != arg_max] =  self.rank_pen
        return score

    def pick_parameters_rank( self,calib_logits, alpha, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion):
        num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
        paramtune_logits, calib_logits = tdata.random_split(calib_logits, [num_paramtune, len(calib_logits)-num_paramtune])
        calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
        paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        
        lamda = self.pick_lamda_size_rank(paramtune_loader, alpha,  randomized, allow_zero_sets)
        return lamda,paramtune_logits, calib_logits
    
    
    
    def pick_lamda_size_rank(self,paramtune_loader, alpha, randomized, allow_zero_sets):
        # Calculate lamda_star
        best_size = iter(paramtune_loader).__next__()[0][1].shape[0] # number of classes 
        # Use the paramtune data to pick lamda.  Does not violate exchangeability.
        lamda_star = 0.5
        for temp_rank_pen in np.insert(np.arange(0.05,0.6,0.05), 0, 0.02): 
        
            conformal_model = SAPS( paramtune_loader, alpha=alpha,randomized=randomized,allow_zero_sets=allow_zero_sets,rank_pen=temp_rank_pen,kreg=0,lamda=0)
            
            top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
            if sz_avg < best_size:
                best_size = sz_avg
                lamda_star = temp_rank_pen

        return lamda_star
    
    




    
    


    
    
