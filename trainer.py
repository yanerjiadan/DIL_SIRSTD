from tqdm import tqdm
from utils.build import *
from utils.utils import *
from copy import deepcopy


import torch
from torch import nn
import os
from tensorboardX import SummaryWriter
from transformers import get_cosine_schedule_with_warmup




class Trainer:
    def __init__(self, args):
        self.args = args
        self.mode = self.args.mode
        self.device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        self.model = build_model(model_name=self.args.model_name)
        self.model.apply(weights_init_xavier)
        self.model.to(self.device)
        self.model_old = None
        
        self.train_loader_list, self.test_loader_list = build_dataloader_list(args=self.args)
        
        self.criterion = build_loss_fn(loss_fn=args.loss_fn)
        self.best_IoU = 0.
        self.PD_FA = PD_FA(nclass=1, bins=10, size=256)
        # self.ROC = ROCMetric(nclass=1, bins=10)
        self.mIoU = mIoU(nclass=1)
        
        
        self.tsbd = SummaryWriter(logdir=self.args.save_dir)
        

    def incremental_train(self, task_id):
        train_loader = self.train_loader_list[task_id]
        test_loader = self.test_loader_list[task_id]
        train_steps = len(train_loader) * self.args.epochs
        self.optimizer = build_optimizer(model=self.model, args=self.args)
        self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=train_steps * 0.05,
                                                         num_training_steps=train_steps)
        self.best_IoU = 0.
        for epoch in range(self.args.epochs):
            self.model.train()
            train_loader = tqdm(train_loader)
            losses = AverageMeter()
            for i, (images, masks) in enumerate(train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                logits = self.model.forward(images, deep_supervision=True)

                loss = self.criterion(logits, masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                losses.update(loss.item(), images.size(0))
                train_loader.set_description('Epoch {}/{}, Loss {:.4f}, lr {:.6f}'.format(epoch + 1, self.args.epochs, losses.avg, self.optimizer.param_groups[0]['lr']))

            if (epoch + 1) % 20 == 0 or (epoch+1) >= self.args.epochs*0.9:
                mean_IoU, PD, FA = self.test(test_loader=test_loader, epoch=epoch)
                if mean_IoU > self.best_IoU:
                    self.best_IoU = mean_IoU
                self.tsbd.add_scalar(f'{task_id}_best_IoU_training', self.best_IoU, epoch)
                self.tsbd.add_scalar(f'{task_id}_IoU_training', mean_IoU, epoch)
                self.tsbd.add_scalar(f'{task_id}_PD_training', PD, epoch)
                self.tsbd.add_scalar(f'{task_id}_FA_training', FA, epoch)
                
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, f'model_after_{task_id}.pt'))
                
                    
                

    def incremental_test(self, task_id):
        self.model.eval()
        for t in range(task_id+1):
            test_loader = self.test_loader_list[t]
            mean_IoU, PD, FA = self.test(test_loader=test_loader, epoch=-2)
            self.tsbd.add_scalar(f'{t}_IoU_test', mean_IoU, task_id)
            self.tsbd.add_scalar(f'{t}_PD_test', PD, task_id)
            self.tsbd.add_scalar(f'{t}_FA_test', FA, task_id)
                


    def test(self, test_loader, epoch=None):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        test_loader = tqdm(test_loader)
        with torch.no_grad():
            for i, (images, masks) in enumerate(test_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model.forward(images, deep_supervision=True)

                self.mIoU.update(logits, masks)
                self.PD_FA.update(logits, masks)
                _, mean_IoU = self.mIoU.get()

                test_loader.set_description('Epoch {}/{}, mIoU {:.4f}'.format(epoch+1, self.args.epochs, mean_IoU))
            FA, PD = self.PD_FA.get(len(test_loader))

            print(f'mIoU: {mean_IoU:.4f}, Pd: {PD[0]:.4f}, Fa: {FA[0]*1e6:.2f}')
            
            

            return mean_IoU, PD[0], FA[0]*1e6