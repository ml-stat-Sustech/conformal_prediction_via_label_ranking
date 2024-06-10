import torch
import torchvision


def build_common_model_imagnet(modelname, mode="test", pre_trained=True, gpus=[0], dataParallel=False):
    if modelname == 'ResNeXt101':
        model = torchvision.models.resnext101_32x8d(weights="IMAGENET1K_V1", progress=pre_trained)
    else:
        raise NotImplementedError

    if mode == "test":
        model.eval()
    else:
        model.train()
    if dataParallel:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
    return model


def build_common_model(modelname, dataset_name="imagnet", mode="test", pre_trained=True, gpus=[0], dataParallel=False):
    if "imagenet" == dataset_name:
        return build_common_model_imagnet(modelname, mode, pre_trained, gpus, dataParallel)
    else:
        raise NotImplementedError
