
from .utils import sort_sum,validate
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.utils.data as tdata
import pandas as pd





class RAPS(nn.Module):
    def __init__(self, calib_loader, alpha,kreg=None,lamda=None, randomized=True, allow_zero_sets=True,pct_paramtune = 0.1, batch_size=32, lamda_criterion='size'):
        super(RAPS, self).__init__()
        self.alpha = alpha
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        if (kreg == None or lamda == None):
            kreg, lamda, calib_logits = pick_parameters( calib_loader.dataset, alpha, kreg, lamda, self.randomized, self.allow_zero_sets, pct_paramtune, batch_size, lamda_criterion)
            
        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))

        self.penalties[:, kreg:] += lamda
        self.Qhat,self.E = conformal_calibration_logits(self, calib_loader,randomized=self.randomized,allow_zero_sets=self.allow_zero_sets)
        self.kreg = kreg
        self.lamda= lamda
        

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy, axis=1)

            I, ordered, cumsum = sort_sum(scores)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties[:,:logits_numpy.shape[1]], randomized=randomized, allow_zero_sets=allow_zero_sets)

        return logits, S



class APS(nn.Module):
    def __init__(self, calib_loader, alpha, randomized=True, allow_zero_sets=True):
        super(APS, self).__init__()
        self.alpha = alpha
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets


        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        
        self.Qhat,self.E = conformal_calibration_logits(self, calib_loader,randomized=self.randomized,allow_zero_sets=self.allow_zero_sets)
        
        

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy, axis=1)


            I, ordered, cumsum = sort_sum(scores)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties[:,:logits_numpy.shape[1]], randomized=randomized, allow_zero_sets=allow_zero_sets)
            

        return logits, S
    


def conformal_calibration_logits(cmodel, calib_loader,randomized=True,allow_zero_sets=True):
    with torch.no_grad():
        E = np.array([])
        targets_list= []
        logits_list = []
        for logits, targets in calib_loader:
            logits = logits.detach().cpu().numpy()
            targets_list.append(targets.detach().cpu().numpy())
            logits_list.append(logits)
            scores = softmax(logits, axis=1)
            

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum,penalties=cmodel.penalties,randomized=randomized,allow_zero_sets=allow_zero_sets)))
            
        Qhat = np.quantile(E,1-cmodel.alpha,method='higher')
    

    return Qhat,E




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

### AUTOMATIC PARAMETER TUNING FUNCTIONS
def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in paramtune_logits])
    kstar = np.quantile(gt_locs_kstar, 1-alpha, method='higher') + 1
    return kstar 

def pick_lamda_size(paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    # Calculate lamda_star
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0] # number of classes 
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    lamda_star = 0.5
    for temp_lam in [0.001, 0.01, 0.1, 0.15,0.2, 0.25,0.3,0.35,0.4,0.45,0.5]:
        conformal_model = RAPS( paramtune_loader, alpha=alpha,kreg=kreg,lamda=temp_lam)
        
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
    
    return lamda_star

def pick_lamda_adaptiveness( paramtune_loader, alpha, kreg, randomized, allow_zero_sets, strata=[[0,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]):
    # Calculate lamda_star
    strata = []
    for i in range(1,1001):
        strata.append([i,i])
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]: # predefined grid, change if more precision desired.
        conformal_model = RAPS( paramtune_loader, alpha=alpha,kreg=kreg,lamda=temp_lam)
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation 
            lamda_star = temp_lam
    return lamda_star

def pick_parameters( calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    paramtune_logits, calib_logits = tdata.random_split(calib_logits, [num_paramtune, len(calib_logits)-num_paramtune])
    calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
    paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

    if kreg == None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(paramtune_loader, alpha, kreg,  randomized, allow_zero_sets)
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
    return kreg, lamda, calib_logits

def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=['size', 'correct'])
    for logit, target in loader_paramtune:
        # compute output
        output, S = cmodel(logit) # This is a 'dummy model' which takes logits, for efficiency.
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy()) 
        correct = np.zeros_like(size)
        for j in range(correct.shape[0]):
            correct[j] = int( target[j] in list(S[j]) )
        batch_df = pd.DataFrame({'size': size, 'correct': correct})
        df = pd.concat([df,batch_df], ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[ (df['size'] >= stratum[0]) & (df['size'] <= stratum[1]) ]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean()-(1-alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation # the violation


from .utils import accuracy


    
    

    
class SAPS(nn.Module):
    """
    Sorted Adaptive Prediction Sets 
    
    """
    def __init__(self, calib_loader, alpha,rank_pen=None, randomized=True, allow_zero_sets=True,batch_size=32,pct_paramtune=0.33):
        super(SAPS, self).__init__()
        self.alpha = alpha
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        

        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        
        if rank_pen == None:
            rank_pen, paramtune_logits, calib_logits = self.pick_parameters_rank( calib_loader.dataset, alpha, self.randomized , self.allow_zero_sets, pct_paramtune, batch_size)
        
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

        score = softmax(score)
        arg_max = np.argmax(score)
        max_calue = np.max(score)
        score[np.arange(score.shape[0]) != arg_max] =  self.rank_pen
        return score

    def pick_parameters_rank( self,calib_logits, alpha, randomized, allow_zero_sets, pct_paramtune, batch_size):
        num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
        paramtune_logits, calib_logits = tdata.random_split(calib_logits, [num_paramtune, len(calib_logits)-num_paramtune])
        calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
        paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        
        lamda = self.pick_lamda_size_rank(paramtune_loader, alpha,  randomized, allow_zero_sets)
        return lamda,paramtune_logits, calib_logits
    
    def pick_parameters_rank_adaptiveness( self,calib_logits, alpha, randomized, allow_zero_sets, pct_paramtune, batch_size):
        num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
        paramtune_logits, calib_logits = tdata.random_split(calib_logits, [num_paramtune, len(calib_logits)-num_paramtune])
        calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
        paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        
        lamda = self.pick_lamda_adaptiveness_rank(paramtune_loader, alpha,  randomized, allow_zero_sets)
        return lamda,paramtune_logits, calib_logits
    
    
    def pick_lamda_size_rank(self,paramtune_loader, alpha, randomized, allow_zero_sets):
        # Calculate lamda_star
        best_size = iter(paramtune_loader).__next__()[0][1].shape[0] # number of classes 
        # Use the paramtune data to pick lamda.  Does not violate exchangeability.
        lamda_star = 0.6
        for temp_rank_pen in np.arange(0.02,0.6,0.03): 
        
            conformal_model = SAPS( paramtune_loader, alpha=alpha,randomized=randomized,allow_zero_sets=allow_zero_sets,rank_pen=temp_rank_pen)
            top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
            if sz_avg < best_size:
                best_size = sz_avg
                lamda_star = temp_rank_pen

        return lamda_star
    


