import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
from models.utils import build_common_model
from lib.utils import *
from lib.post_process import*

import torchcp
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import SAPS,APS
from torchcp.classification.utils.metrics import Metrics

metrics = Metrics()
class experiment:
    def __init__(self,model_name,alpha,predictor,dataset_name,post_hoc,num_trials) -> None:
        self.model_name = model_name
        self.alpha=  alpha
        self.predictor = predictor
        self.dataset_name = dataset_name
        if self.dataset_name =="imagenet":
            self.num_calsses = 1000
        else:
            raise NotImplementedError
        self.post_hoc =  post_hoc
        self.model = build_common_model(self.model_name,dataset_name)

        ### Data Loading
        self.dataset = get_logits_dataset(self.model_name,self.dataset_name)
        
        # trials
        self.num_trials = num_trials
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def run(self, n_data_conf, n_data_val, pct_paramtune, bsz):
        ### Perform experiment
        top1s = np.zeros((self.num_trials ,))
        top5s = np.zeros((self.num_trials ,))
        coverages = np.zeros((self.num_trials ,))
        sizes = np.zeros((self.num_trials ,))

        for i in tqdm(range(self.num_trials )):
            self.seed  = i
            self._fix_randomness(self.seed)
            top1_avg, top5_avg, cvg_avg, sz_avg  = self.trial( n_data_conf, n_data_val, pct_paramtune, bsz)
            top1s[i] = top1_avg
            top5s[i] = top5_avg
            coverages[i] = cvg_avg
            sizes[i] = sz_avg
            print(f'\n\tTop1: {np.median(top1s[0:i+1]):.3f}, Top5: {np.median(top5s[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}\033[F', end='')
        print('')
    



    def trial(self, n_data_conf, n_data_val, pct_paramtune, bsz):
        alpha = self.alpha
        cal_dataset, val_dataset = split2(self.dataset, n_data_conf, n_data_val) 
        # Calibrate the temperature via temperature scaling
        if self.post_hoc == "TS":
            transformation = OptimalTeamperatureScaling(1.3)
        else:
            raise NotImplementedError
        
       # Prepare the loaders
        cal_loader = torch.utils.data.DataLoader(cal_dataset, batch_size = bsz, shuffle=False, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = bsz, shuffle=False, pin_memory=True)
        
        # optimzing the temperature
        transformation =  self.get_optimal_parameters(transformation, cal_loader)            
        cal_logits, cal_labels = postHocLogits(transformation,cal_loader,self.device,self.num_calsses )
        val_logits, val_lables = postHocLogits(transformation,val_loader,self.device,self.num_calsses )
        
        if self.predictor == "SAPS":

            ################
            # Choose the best ranking weight
            ################
            pc_indices = int(cal_logits.size(0)*pct_paramtune)
            indices = torch.randperm(cal_logits.size(0))
            tuning_logits = cal_logits[indices[:pc_indices]]
            tuning_labels = cal_labels[indices[:pc_indices]]
            cal_logits = cal_logits[indices[pc_indices:]]
            cal_labels = cal_labels[indices[pc_indices:]]
            
            ranking_weight_star = 0
            best_set_size = self.num_calsses
            for temp_ranking_weight in np.insert(np.arange(0.05,0.65,0.05), 0, 0.02): 
                predictor = SplitPredictor(SAPS(temp_ranking_weight))
                predictor.calculate_threshold(tuning_logits,tuning_labels, alpha)       
                prediction_sets = predictor.predict_with_logits(tuning_logits)
                average_size = metrics('average_size')(prediction_sets, tuning_labels)
                if average_size < best_set_size:
                    ranking_weight_star = temp_ranking_weight
                    best_set_size = average_size
            predictor = SplitPredictor(SAPS(ranking_weight_star))
        elif self.predictor == "APS":
            predictor = SplitPredictor(APS())
        else:
            raise NotImplementedError
            
            
        predictor.calculate_threshold(cal_logits,cal_labels, alpha)
        prediction_sets = predictor.predict_with_logits(val_logits)
        coverage_rate = metrics('coverage_rate')(prediction_sets, val_lables)
        average_size = metrics('average_size')(prediction_sets, val_lables)
        prec1, prec5 = accuracy(val_logits, val_lables, topk=(1, 5))

        return prec1,prec5,coverage_rate,average_size
    

    def  get_optimal_parameters(self,transformation, calib_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        
        device = self.device
        transformation.to(device)
        max_iters=10
        lr=0.01
        epsilon=0.01
        nll_criterion = nn.CrossEntropyLoss().cuda()

        T = transformation.temperature

        optimizer = optim.SGD([transformation.temperature], lr=lr)
        for iter in range(max_iters):
            T_old = T.item()
            # print(T_old)
            for x, targets in calib_loader:
                optimizer.zero_grad()
                x = x.cuda()
                x.requires_grad = True
                out = x/transformation.temperature
                loss = nll_criterion(out, targets.long().cuda())
                
                loss.backward()
                optimizer.step()
            T = transformation.temperature
            if abs(T_old - T.item()) < epsilon:
                break

        return transformation

    def _fix_randomness(self,seed=0):
        ### Fix randomness 
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates conformal predictors',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset')
    parser.add_argument('--model', type=str, default='ResNeXt101', help='model')
    parser.add_argument('--predictor', type=str, default='SAPS', help='the predictor of CP.')
    parser.add_argument('--alpha', type=float, default=0.1, help='the error rate.')
    parser.add_argument('--trials', type=int, default=5, help='number of trials')
    parser.add_argument('--post_hoc', type=str, default="TS", help='the confidence calibration method.')
    
    args = parser.parse_args()
    dataset_name = args.dataset
    model = args.model
    num_trials = args.trials        
    alpha = args.alpha
    post_hoc = args.post_hoc
    predictor = args.predictor
    
    if dataset_name==  "imagenet":
        n_data_conf = 30000
        n_data_val = 20000
    else:
        raise NotImplementedError
    
    pct_paramtune = 0.2
    bsz = 128
    cudnn.benchmark = True
    print(f'Model: {model} | Desired coverage: {1-alpha} | Predictor: {predictor}| Calibration: {post_hoc}')
    this_experiment =  experiment(model,alpha,predictor,dataset_name,post_hoc,num_trials)
    this_experiment.run( n_data_conf, n_data_val, pct_paramtune, bsz) 

                
                
    