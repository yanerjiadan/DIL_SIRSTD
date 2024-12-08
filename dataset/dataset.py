from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torchvision import transforms
import json
import random
from PIL import ImageOps, ImageFilter
import numpy as np
import random

class Split_SIRSTD(Dataset):
    def __init__(self, is_train, task_id):
        self.is_train = is_train
        self.task_id = task_id
        self.basic_size = 256
        self.crop_size = 256
        mean_list = [0.43912969123762524, 0.4184412863421091, 0.44336409581105646, 0.44391380160904076, 0.42093362094996506, 0.39517910423464103, 0.3919024005292877]
        std_list = [0.14216294851289388, 0.14962109511816776, 0.14331847391058938, 0.15104198486175402, 0.14279883526483003, 0.1584699068773762, 0.15488132569796823]

        self.root_dir = './dataset/DIL_SIRSTD'
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.mask_dir = os.path.join(self.root_dir, 'masks')
        self.image_list = []
        self.image_dir_cur = os.path.join(self.image_dir, str(self.task_id))
        self.mask_dir_cur = os.path.join(self.mask_dir, str(self.task_id))
        if self.is_train:
            with open(os.path.join(self.image_dir_cur, 'train.json'), 'r') as f:
                self.image_list = json.load(f)
        else:
            with open(os.path.join(self.image_dir_cur, 'test.json'), 'r') as f:
                self.image_list = json.load(f)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean_list[self.task_id], mean_list[self.task_id], mean_list[self.task_id]],
                                 std=[std_list[self.task_id], std_list[self.task_id], std_list[self.task_id]])
        ])
        self.mask_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.basic_size * 0.5), int(self.basic_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # mask = self.binarize(mask)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask

    def _testval_sync_transform(self, img, mask):
        img = img.resize((self.basic_size, self.basic_size), Image.BILINEAR)
        mask = mask.resize((self.basic_size, self.basic_size), Image.BILINEAR)
        mask = self.binarize(mask)

        return img, mask

    def binarize(self, img, threshold=127.5):
        img = np.array(img)
        img = (img > threshold).astype(np.uint8) * 255
        img = Image.fromarray(img)
        return img

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, str(self.task_id), self.image_list[idx])
        img = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(self.mask_dir, str(self.task_id), self.image_list[idx])
        mask = Image.open(mask_path)

        if self.is_train:
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._testval_sync_transform(img, mask)

        img = self.transforms(img)
        mask = self.mask_transforms(mask)
        return img, mask

class DIL_SIRSTD(Dataset):
    def __init__(self, is_train, sub_task_list):
        self.dataset_list = []
        for task_id in sub_task_list:
            self.dataset_list.append(Split_SIRSTD(is_train=is_train, task_id=task_id))
    def __len__(self):
        l = 0
        for d in self.dataset_list:
            l += d.__len__()
        return l

    def __getitem__(self, idx):
        if idx < self.dataset_list[0].__len__():
            return self.dataset_list[0].__getitem__(idx)
        else:
            return self.dataset_list[1].__getitem__(idx-self.dataset_list[0].__len__())

class DIL_SIRSTD2(Dataset):
    def __init__(self, is_train, task_id):
        nuaa_dir = './dataset/NUAA-SIRST'
        nudt_dir = './dataset/NUDT-SIRST'
        irstd_dir = './dataset/IRSTD-1k'
        sirst_dir = './dataset/SIRST-v2'

        nuaa_image_dir = os.path.join(nuaa_dir, 'images/images')
        nuaa_mask_dir = os.path.join(nuaa_dir, 'masks/masks')
        nudt_image_dir = os.path.join(nudt_dir, 'images')
        nudt_mask_dir = os.path.join(nudt_dir, 'masks')
        irstd_image_dir = os.path.join(irstd_dir, 'IRSTD1k_Img')
        irstd_mask_dir = os.path.join(irstd_dir, 'IRSTD1k_Label')
        sirst_image_dir = os.path.join(sirst_dir, 'mixed')
        sirst_mask_dir = os.path.join(sirst_dir, 'annotations/masks')

        tmp_list = []
        self.img_list = []
        self.mask_list = []

        self.is_train = is_train
        self.task = task_id
        self.basic_size = 256
        self.crop_size = 256

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
        self.mask_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        random.seed(42)
        for root, dirs, files in os.walk(nudt_image_dir):
            for file in files:
                if file.endswith('.png'):
                    tmp_list.append(file)
        for i in range(10):
            random.shuffle(tmp_list)

        if task_id == 0:
            self.img_dir = nuaa_image_dir
            self.mask_dir = nuaa_mask_dir
        elif task_id == 1:
            self.img_dir = nudt_image_dir
            self.mask_dir = nudt_mask_dir
        elif task_id == 2:
            self.img_dir = irstd_image_dir
            self.mask_dir = irstd_mask_dir
        elif task_id == 3:
            self.img_dir = sirst_image_dir
            self.mask_dir = sirst_mask_dir

        if is_train:
            if task_id == 0:
                txt = os.path.join(nuaa_dir, 'idx_427', 'trainval.txt')
                with open(txt, 'r') as f:
                    lines = f.readlines()
                    self.img_list = [line.strip() + '.png' for line in lines]
                    self.mask_list = [line.strip() + '_pixels0.png' for line in lines]

            elif task_id == 1:
                self.img_list = tmp_list[:1061]
                self.mask_list = self.img_list

            elif task_id == 2:
                txt = os.path.join(irstd_dir, 'trainval.txt')
                with open(txt, 'r') as f:
                    lines = f.readlines()
                    self.img_list = [line.strip() + '.png' for line in lines]
                    self.mask_list = self.img_list

            elif task_id == 3:
                txt = os.path.join(sirst_dir, 'splits/trainval_full.txt')
                with open(txt, 'r') as f:
                    lines = f.readlines()
                    self.img_list = [line.strip() + '.png' for line in lines]
                    self.mask_list = [line.strip() + '_pixels0.png' for line in lines]

        else:
            if task_id == 0:
                txt = os.path.join(nuaa_dir, 'idx_427', 'test_full.txt')
                with open(txt, 'r') as f:
                    lines = f.readlines()
                    self.img_list = [line.strip() + '.png' for line in lines]
                    self.mask_list = [line.strip() + '_pixels0.png' for line in lines]

            elif task_id == 1:
                self.img_list = tmp_list[1061:]
                self.mask_list = self.img_list

            elif task_id == 2:
                txt = os.path.join(irstd_dir, 'test.txt')
                with open(txt, 'r') as f:
                    lines = f.readlines()
                    self.img_list = [line.strip() + '.png' for line in lines]
                    self.mask_list = self.img_list

            elif task_id == 3:
                txt = os.path.join(sirst_dir, 'splits/test_full.txt')
                with open(txt, 'r') as f:
                    lines = f.readlines()
                    self.img_list = [line.strip() + '.png' for line in lines]
                    self.mask_list = [line.strip() + '_pixels0.png' for line in lines]

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.basic_size * 0.5), int(self.basic_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.BILINEAR)
        mask = self.binarize(mask)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask

    def _testval_sync_transform(self, img, mask):
        img = img.resize((self.basic_size, self.basic_size), Image.BILINEAR)
        mask = mask.resize((self.basic_size, self.basic_size), Image.NEAREST)
        # mask = self.binarize(mask)

        return img, mask

    def binarize(self, img, threshold=127.5):
        img = np.array(img)
        img = (img > threshold).astype(np.uint8) * 255
        img = Image.fromarray(img)
        return img

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
        mask = Image.open(mask_path)
        
        if self.is_train:
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._testval_sync_transform(img, mask)

        img = self.transforms(img)
        mask = self.mask_transforms(mask)
        return img, mask









if __name__ == '__main__':
    train_dataset_list = []
    test_dataset_list = []
    for i in range(3):
        train_dataset_list.append(DIL_SIRSTD2(is_train=True, task_id=i))
        test_dataset_list.append(DIL_SIRSTD2(is_train=False, task_id=i))
        print(f'task_id: {i}, train_dataset_len: {len(train_dataset_list[i])}, test_dataset_len: {len(test_dataset_list[i])}')