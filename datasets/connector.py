import os
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F



def build_dataset(dataset, mode="train",transform=None):
    
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir,"data")
    
    
    
    

    if dataset == 'imagenet': 
        if transform==None:
            train_transform = trn.Compose([
                            trn.Resize(256),
                            trn.CenterCrop(224),
                            trn.ToTensor(),
                            trn.Normalize(mean=[0.485, 0.456, 0.406],
                                        std =[0.229, 0.224, 0.225])
                            ])
            
            
            test_transform = trn.Compose([
                            trn.Resize(256),
                            trn.CenterCrop(224),
                            trn.ToTensor(),
                            trn.Normalize(mean=[0.485, 0.456, 0.406],
                                        std =[0.229, 0.224, 0.225])
                            ])
        else:
            train_transform=transform
            test_transform=transform
        if mode=="test":
            data = dset.ImageFolder(data_dir+"/imagenet/val", 
                                    train_transform)
            
        else:
            raise NotImplementedError
        num_classes = 1000  
    
    return data, num_classes


