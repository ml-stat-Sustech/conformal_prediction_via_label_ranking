import os
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn


def build_dataset(dataset, mode="train", transform=None):
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data")
    if dataset == 'imagenet':
        if transform == None:
            transform = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])
        if mode == "test":
            data = dset.ImageFolder(data_dir + "/imagenet/val",
                                    transform)
        else:
            raise NotImplementedError
        num_classes = 1000

    else:
        raise NotImplementedError

    return data, num_classes
