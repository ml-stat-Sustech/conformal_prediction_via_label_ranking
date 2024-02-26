import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
from models.utils import build_common_model
from lib.utils import *
from lib.post_process import*
from lib.saps import *


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
        self.logits = get_logits_dataset(self.model_name,self.dataset_name)
        
        # trials
        self.num_trials = num_trials
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def run(self, n_data_conf, n_data_val, pct_paramtune, bsz):
        ### Perform experiment
        top1s = np.zeros((self.num_trials ,))
        top5s = np.zeros((self.num_trials ,))
        coverages = np.zeros((self.num_trials ,))
        sizes = np.zeros((self.num_trials ,))
        escvs = np.zeros((self.num_trials ,))

        for i in tqdm(range(self.num_trials )):
            self.seed  = i
            self._fix_randomness(self.seed)
            top1_avg, top5_avg, cvg_avg, sz_avg,escv  = self.trial( n_data_conf, n_data_val, pct_paramtune, bsz)
            top1s[i] = top1_avg
            top5s[i] = top5_avg
            coverages[i] = cvg_avg
            sizes[i] = sz_avg
            escvs[i] = escv
            

            print(f'\n\tTop1: {np.median(top1s[0:i+1]):.3f}, Top5: {np.median(top5s[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}, escv: {np.median(escvs[0:i+1]):.3f}\033[F', end='')
        print('')
        
        # Svae the median results
        res_dict={}
        res_dict["Model"] = self.model_name
        res_dict["Dataset"] = self.dataset_name
        res_dict["Predictor"] = self.predictor
        res_dict["alpha"] = self.alpha
        res_dict["post_hc"] = self.post_hoc
        res_dict["Top1"] = np.round(np.median(top1s),4)
        res_dict["Top5"] = np.round(np.median(top5s),4)
        res_dict["Coverage"] = np.round(np.median(coverages),4)
        res_dict["Size"] = np.round(np.median(sizes),4)
        res_dict["ESCV"] = np.round(np.median(escvs),4)
        return res_dict



    def trial(self, n_data_conf, n_data_val, pct_paramtune, bsz):
        alpha = self.alpha
        
        logits_cal, logits_val,self.cal_indices,self.val_indices= split2(self.logits, n_data_conf, len(self.logits)-n_data_conf) 


        ######################
        # psot hoc
        ######################

        # Calibrate the temperature via temperature scaling
        if self.post_hoc == "oTS":
            transformation = OptimalTeamperatureScaling(1.3)
        else:
            raise NotImplementedError
        

        
       # Prepare the loaders
        loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size = bsz, shuffle=False, pin_memory=True)
        loader_val = torch.utils.data.DataLoader(logits_val, batch_size = bsz, shuffle=False, pin_memory=True)
        
        # optimzing the temperature
        transformation =  self.get_optimal_parameters(transformation,loader_cal)
            
        logits_cal = postHocLogits(transformation,loader_cal,self.device,self.num_calsses )
        logits_val = postHocLogits(transformation,loader_val,self.device,self.num_calsses )

        # Prepare the loaders
        loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size = bsz, shuffle=False, pin_memory=True)
        loader_val = torch.utils.data.DataLoader(logits_val, batch_size = bsz, shuffle=False, pin_memory=True)
        

        allow_zero_sets = True
        randomized = True
        
        if self.predictor == "SAPS":
            self.conformal_model = SAPS(loader_cal, alpha=alpha,rank_pen=None,randomized=randomized,allow_zero_sets=allow_zero_sets,batch_size=bsz,pct_paramtune=pct_paramtune)
        else:
            raise NotImplementedError

        return self.validate(loader_val)
    

    def  get_optimal_parameters(self,transformation,calib_loader):
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

    def validate(self,val_loader):
        with torch.no_grad():
            batch_time = AverageMeter('batch_time')
            top1 = AverageMeter('top1')
            top5 = AverageMeter('top5')
            # switch to evaluate mode
            self.conformal_model.eval()
            end = time.time()
            N = 0
            all_S =[]
            targets=[]
            size_array=[]
            correct_array=[]
            topk=[]
            for i, (logits, target) in enumerate(val_loader):
                target = target.cuda()
                logits = logits.cuda()
                
                
                # compute output
                # logits , prediction sets
                _,S = self.conformal_model(logits.cuda())
                all_S.extend(S)
                targets.extend(list(target.detach().cpu().numpy()))
                for i in range(target.shape[0]):
                    size_array.append(len(S[i]))
                    if (target[i].item() in S[i]):
                        correct_array.append(1)
                    else:
                        correct_array.append(0)
                
                # measure accuracy and record loss
                prec1, prec5 = accuracy(logits, target, topk=(1, 5))

                # Update meters
                top1.update(prec1.item()/100.0, n=logits.shape[0])
                top5.update(prec5.item()/100.0, n=logits.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                N = N + logits.shape[0]
                
        escv= self.cal_escv(size_array,correct_array)
        return top1.avg, top5.avg, np.mean(correct_array), np.mean(size_array),escv

    def _fix_randomness(self,seed=0):
        ### Fix randomness 
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)


    def cal_escv(self, size_array,correct_array):
        """
        computing the Each-Size Coverage Violation (escv)
        """
        size_array = np.array(size_array)
        correct_array = np.array(correct_array)

        escv =0
        for i in range(1,self.num_calsses+1):
            temp_index = np.argwhere( size_array == i )
            if len(temp_index)>0:
                temp_index= temp_index[:,0]
                
                stratum_violation = max(0,(1-self.alpha) - np.mean(correct_array[temp_index]))
                escv = max(escv, stratum_violation)
        return escv
    

if __name__ == "__main__":
    """
    The main experiments
    """
    parser = argparse.ArgumentParser(description='Evaluates conformal predictors',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    


    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset')
    parser.add_argument('--model', type=str, default='ResNeXt101', help='model')
    parser.add_argument('--predictor', type=str, default='SAPS', help='the predictor of CP.')
    parser.add_argument('--alpha', type=float, default=0.1, help='the error rate.')
    parser.add_argument('--trials', type=int, default=1, help='number of trials')
    parser.add_argument('--post_hoc', type=str, default="oTS", help='the confidence calibration method.')
    
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
    bsz = 320
    cudnn.benchmark = True
    print(f'Model: {model} | Desired coverage: {1-alpha} | Predictor: {predictor}| Calibration: {post_hoc}')
    this_experiment =  experiment(model,alpha,predictor,dataset_name,post_hoc,num_trials)
    out = this_experiment.run( n_data_conf, n_data_val, pct_paramtune, bsz) 

                
                
    