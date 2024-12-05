from dataset.dataset import Split_SIRSTD, DIL_SIRSTD, DIL_SIRSTD2
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adagrad, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import get_cosine_schedule_with_warmup
import torch

from model.MSHNet import MSHNet

def build_model(model_name):
    if model_name == 'MSHNet':
        model = MSHNet(input_channels=3)
        return model
    else:
        print('Model not found')
        raise NotImplementedError

def build_loss_fn(loss_fn):
    if loss_fn == 'SoftIoULoss':
        return SoftIoULoss
    elif loss_fn == 'Dice':
        return Dice
    else:
        print('Loss not found')
        raise NotImplementedError



def build_dataloader_list(args):
    if args.dataset == 'DIL_SIRSTD':
        train_dataloader_list = []
        test_dataloader_list = []
        for i in [1,2,5,3,6,4]:
            _dataset = DIL_SIRSTD(is_train=True, task_id=i)
            train_dataloader_list.append(
                DataLoader(dataset=_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False))
            _dataset = DIL_SIRSTD(is_train=False, task_id=i)
            test_dataloader_list.append(
                DataLoader(dataset=_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False))
        return train_dataloader_list, test_dataloader_list
    
    elif args.dataset == 'DIL_SIRSTD2':
        train_dataloader_list = []
        test_dataloader_list = []
        for i in [2,1,0]:
            _dataset = DIL_SIRSTD2(is_train=True, task_id=i)
            train_dataloader_list.append(
                DataLoader(dataset=_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False))
            _dataset = DIL_SIRSTD2(is_train=False, task_id=i)
            test_dataloader_list.append(
                DataLoader(dataset=_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False))
        return train_dataloader_list, test_dataloader_list

def build_optimizer(model, args):
    if args.optimizer == 'Adagrad':
        optimizer = Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        print("Optimizer not supported")
        raise NotImplementedError
    return optimizer

def SoftIoULoss(pred, target):
    pred = torch.sigmoid(pred)

    smooth = 1

    intersection = pred * target

    loss = (intersection.sum() + smooth) / \
            (pred.sum() + target.sum() - intersection.sum() + smooth)

    loss = 1 - loss.mean()

    return loss

def Dice(pred, target):
    pred = torch.sigmoid(pred)

    smooth = 1

    intersection = pred * target
    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))

    loss = (2 * intersection_sum + smooth) / \
            (pred_sum + target_sum + intersection_sum + smooth)

    loss = 1 - loss.mean()

    return loss