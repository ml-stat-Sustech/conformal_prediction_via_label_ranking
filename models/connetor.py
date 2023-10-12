import torch
import torch.nn as  nn
from torchvision.models import densenet121
import numpy as np
import torchvision
import os
from datasets.connector import build_dataset
import torch.optim as optim
import sys
import torchvision.transforms as trn

import timm





def build_common_model_imagnet(modelname,mode="test",pre_trained=True,gpus=[0],dataParallel=False):
    if modelname == 'ResNet18':
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1", progress=True)

    elif modelname == 'ResNet50':
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1", progress=True)

    elif modelname == 'ResNet101':
        model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)

    elif modelname == 'ResNet152':
        model = torchvision.models.resnet152(weights="IMAGENET1K_V1", progress=True)

    elif modelname == 'ResNeXt101':
        model = torchvision.models.resnext101_32x8d(weights="IMAGENET1K_V1", progress=True)

    elif modelname == 'VGG16':
        model = torchvision.models.vgg16(weights="IMAGENET1K_V1", progress=True)
    elif modelname == "VGG16_BN":
        model = torchvision.models.vgg16_bn(weights="IMAGENET1K_V1", progress=True)

    elif modelname == 'ShuffleNet':
        model = torchvision.models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1", progress=True)
    elif modelname =="ShuffleNet_v2_x2_0":
        model = torchvision.models.shufflenet_v2_x2_0(weights="IMAGENET1K_V1", progress=True)

    elif modelname == 'Inception':
        model = torchvision.models.inception_v3(weights="IMAGENET1K_V1", progress=True)

    elif modelname == 'DenseNet161':
        model = torchvision.models.densenet161(weights="IMAGENET1K_V1", progress=True)
    elif modelname == "ViT":
        model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1", progress=True)

    elif    modelname == "DeiT":
    

        model = timm.create_model("hf_hub:timm/deit_base_distilled_patch16_224.fb_in1k", pretrained=True)

    else:
        raise NotImplementedError
    

    

    if mode == "test":
        model.eval()
    else:
        model.train()
    if dataParallel :
        model = torch.nn.DataParallel(model) .cuda()
    else:
        model.cuda()
    
    return model



    

def build_common_model(modelname,dataset_name="imagnet",mode="test",pre_trained=True,gpus=[0],dataParallel=False):
   
    if modelname == "CLIP":
        # Load the model
        from  .clip import load as clipload
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clipload('ViT-B/32', device)
        return [model.eval().to(device),preprocess]
        
    if "imagenet" in dataset_name:
        return build_common_model_imagnet(modelname,mode,pre_trained,gpus,dataParallel)
    
    else:
        raise NotImplementedError
    
    
