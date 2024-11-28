from tqdm import tqdm
from utils.build import *
from utils.utils import *
from copy import deepcopy


import torch
from torch import nn



class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device) if self.args.cuda else torch.device('cpu')
        self.model = build_model(model_name=self.args.model_name)
        self.model_old = None
        self.train_loader_list, self.test_loader_list = build_dataloader_list(args=self.args)
        self.optimizer = build_optimizer(model=self.model, args=self.args)
        self.scheduler = None
        self.criterion = build_loss_fn(loss_fn=args.loss_fn)
        self.PD_FA = PD_FA(nclass=1, bins=10, size=args.batch_size)
        self.ROC = ROCMetric(nclass=1, bins=10)
        self.mIoU = mIoU(nclass=1)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.model.to(self.device)

    def incremental_train(self, task_id):
        train_loader = self.train_loader_list[task_id]
        train_steps = len(train_loader) * self.args.epochs
        self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=train_steps * 0.1,
                                                         num_training_steps=train_steps)
        for epoch in range(self.args.epochs):
            self.model.train()
            train_loader = tqdm(train_loader)
            losses = AverageMeter()
            for i, (images, masks) in enumerate(train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                if epoch < 5:
                    logits_deep, logits = self.model.forward(images, deep_supervision=False)
                else:
                    logits_deep, logits = self.model.forward(images, deep_supervision=True)

                loss = self.criterion(logits, masks)
                for j in range(len(logits_deep)):
                    loss = loss + self.criterion(logits_deep[j], masks)
                    masks = self.down(masks)
                loss = loss / (len(logits_deep) + 1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                losses.update(loss.item(), images.size(0))
                train_loader.set_description('Epoch {}/{}, Loss {:.4f}'.format(epoch + 1, self.args.epochs, losses.avg))


    def incremental_test(self, task_id):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        for t in range(task_id):
            test_loader = tqdm(self.test_loader_list[t])
            with torch.no_grad():
                for i, (images, masks) in enumerate(test_loader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    _, logits = self.model.forward(images, deep_supervision=True)

                    self.mIoU.update(logits, masks)
                    self.PD_FA.update(logits, masks)

                    _, mean_IoU = self.mIoU.get()

                    test_loader.set_description(f'Task {t+1}/{task_id}, mIoU: {mean_IoU:.4f}')





    def test(self, test_loader, epoch=None):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        test_loader = tqdm(test_loader)
        with torch.no_grad():
            for i, (images, masks) in enumerate(test_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                if epoch < 5:
                    _, logits = self.model.forward(images, deep_supervision=False)
                else:
                    _, logits = self.model.forward(images, deep_supervision=True)

                self.mIoU.update(logits, masks)
                self.PD_FA.update(logits, masks)
                # self.ROC.update(logits, masks)
                _, mean_IoU = self.mIoU.get()

                test_loader.set_description('Epoch {}/{}, mIoU {:.4f}'.format(epoch+1, self.args.epochs, mean_IoU))
            FA, PD = self.PD_FA.get(len(test_loader))
            # TP_rate, FP_rate, _, _ = self.ROC.get()

            print(f'mIoU: {mean_IoU:.4f}, Pd: {PD[0]:.2f}, Fa: {FA[0]*1000000:.2f}')

