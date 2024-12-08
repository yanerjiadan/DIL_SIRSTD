import os
import shutil
import random
import json

nuaa_dir = '/mnt/e/Dataset/NUAA-SIRST'
nudt_dir = '/mnt/e/Dataset/NUDT-SIRST'
irstd_dir = '/mnt/e/Dataset/IRSTD-1k'
sirst_dir = '/mnt/e/Dataset/SIRST-v2'
target_dir = '/mnt/e/Dataset/DIL_SIRSTD'

nuaa_image_dir = os.path.join(nuaa_dir, 'images/images')
nuaa_mask_dir = os.path.join(nuaa_dir, 'masks/masks')
nudt_image_dir = os.path.join(nudt_dir, 'images')
nudt_mask_dir = os.path.join(nudt_dir, 'masks')
irstd_image_dir = os.path.join(irstd_dir, 'IRSTD1k_Img')
irstd_mask_dir = os.path.join(irstd_dir, 'IRSTD1k_Label')
target_image_dir = os.path.join(target_dir, 'images')
target_mask_dir = os.path.join(target_dir, 'masks')
sirst_image_dir = os.path.join(sirst_dir, 'mixed')
sirst_mask_dir = os.path.join(sirst_dir, 'annotations/masks')

# if not os.path.exists(os.path.join(target_image_dir, '0')):
#     for i in range(7):
#         os.mkdir(os.path.join(target_image_dir, str(i)))
#         os.mkdir(os.path.join(target_mask_dir, str(i)))
#
# for root, dirs, files in os.walk(nuaa_image_dir):
#     for file in files:
#         if file.endswith('png'):
#             num = file.split('.')[0].split('_')[-1]
#             new_name = num.zfill(6) + '.png'
#             shutil.copy(os.path.join(nuaa_image_dir, file), os.path.join(target_image_dir, '0', new_name))
#
# for root, dirs, files in os.walk(nuaa_mask_dir):
#     for file in files:
#         if file.endswith('png'):
#             num = file.split('_')[1]
#             new_name = num.zfill(6) + '.png'
#             shutil.copy(os.path.join(nuaa_mask_dir, file), os.path.join(target_mask_dir, '0', new_name))
#
# for root, dirs, files in os.walk(os.path.join(target_dir, 'origin', '1')):
#     for file in files:
#         if file.endswith('png'):
#             new_name = file
#             new_name = '1' + new_name[1:]
#             shutil.copy(os.path.join(nudt_image_dir, file), os.path.join(target_image_dir,'1', new_name))
#             shutil.copy(os.path.join(nudt_mask_dir, file), os.path.join(target_mask_dir, '1', new_name))
#
# for root, dirs, files in os.walk(os.path.join(target_dir, 'origin','2')):
#     for file in files:
#         if file.endswith('png'):
#             new_name = file
#             new_name = '1' + new_name[1:]
#             shutil.copy(os.path.join(nudt_image_dir, file), os.path.join(target_image_dir,'2', new_name))
#             shutil.copy(os.path.join(nudt_mask_dir, file), os.path.join(target_mask_dir, '2', new_name))
#
# for root, dirs, files in os.walk(os.path.join(target_dir, 'origin','3')):
#     for file in files:
#         if file.endswith('png'):
#             new_name = file
#             new_name = '1' + new_name[1:]
#             shutil.copy(os.path.join(nudt_image_dir, file), os.path.join(target_image_dir,'3', new_name))
#             shutil.copy(os.path.join(nudt_mask_dir, file), os.path.join(target_mask_dir, '3', new_name))
#
# for root, dirs, files in os.walk(os.path.join(target_dir, 'origin','4')):
#     for file in files:
#         if file.endswith('png'):
#             new_name = file
#             new_name = '1' + new_name[1:]
#             shutil.copy(os.path.join(nudt_image_dir, file), os.path.join(target_image_dir,'4', new_name))
#             shutil.copy(os.path.join(nudt_mask_dir, file), os.path.join(target_mask_dir, '4', new_name))
#
# for root, dirs, files in os.walk(os.path.join(target_dir, 'origin','5')):
#     for file in files:
#         if file.endswith('png'):
#             num = file.split('.')[0].split('U')[1]
#             new_name = num.zfill(6) + '.png'
#             new_name = '2' + new_name[1:]
#             shutil.copy(os.path.join(irstd_image_dir, file), os.path.join(target_image_dir,'5', new_name))
#             shutil.copy(os.path.join(irstd_mask_dir, file), os.path.join(target_mask_dir, '5', new_name))
#
# for root, dirs, files in os.walk(os.path.join(target_dir, 'origin','6')):
#     for file in files:
#         if file.endswith('png'):
#             num = file.split('.')[0].split('U')[1]
#             new_name = num.zfill(6) + '.png'
#             new_name = '2' + new_name[1:]
#             shutil.copy(os.path.join(irstd_image_dir, file), os.path.join(target_image_dir,'6', new_name))
#             shutil.copy(os.path.join(irstd_mask_dir, file), os.path.join(target_mask_dir, '6', new_name))




#-------------划分数据集---------------------

for i in [5, 6]:
    current_dir = os.path.join(target_image_dir, str(i))
    img_list = []
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.png'):
                img_list.append(file)
        train_len = int(len(img_list)*0.8)
        random.seed(10)
        for i in range(random.randint(20,40)):
            random.shuffle(img_list)
        train_files = img_list[:train_len]
        test_files = img_list[train_len:]
        train_files.sort()
        test_files.sort()
        with open(os.path.join(current_dir, 'train.json'), 'w') as f:
            json.dump(train_files, f)
        with open(os.path.join(current_dir, 'test.json'), 'w') as f:
            json.dump(test_files, f)

#---------------导入SIRST-v2---------------------
# test_list = []
# train_list = []
# txt = os.path.join(target_image_dir, '7', 'trainval_full.txt')
# with open(txt, 'r') as f:
#     lines = f.readlines()
#     img_train_list = [line.strip() + '.png' for line in lines]
#     mask_train_list = [line.strip() + '_pixels0.png' for line in lines]
# txt = os.path.join(target_image_dir, '7', 'test_full.txt')
# with open(txt, 'r') as f:
#     lines = f.readlines()
#     img_test_list = [line.strip() + '.png' for line in lines]
#     mask_test_list = [line.strip() + '_pixels0.png' for line in lines]
# for num, img_name in enumerate(img_train_list):
#     new_name = str(num).zfill(6) + '.png'
#     new_name = '3' + new_name[1:]
#     shutil.copy(os.path.join(sirst_image_dir, img_train_list[num]), os.path.join(target_image_dir, '7', new_name))
#     shutil.copy(os.path.join(sirst_mask_dir, mask_train_list[num]), os.path.join(target_mask_dir, '7', new_name))
#     train_list.append(new_name)
# for num, img_name in enumerate(img_test_list):
#     new_name = str(num+len(img_train_list)).zfill(6) + '.png'
#     new_name = '3' + new_name[1:]
#     shutil.copy(os.path.join(sirst_image_dir, img_test_list[num]), os.path.join(target_image_dir, '7', new_name))
#     shutil.copy(os.path.join(sirst_mask_dir, mask_test_list[num]), os.path.join(target_mask_dir, '7', new_name))
#     test_list.append(new_name)
#
# with open(os.path.join(target_image_dir, '7', 'train.json'), 'w') as f:
#     json.dump(train_list, f)
# with open(os.path.join(target_image_dir, '7', 'test.json'), 'w') as f:
#     json.dump(test_list, f)

# for i in '1234':
#     for root, dirs, files in os.walk(os.path.join(target_image_dir, i)):
#         for file in files:
#             if file.endswith('.png'):
#                 mask_name = '0'+file[1:]
#                 shutil.copy(os.path.join(nudt_mask_dir, mask_name), os.path.join(target_mask_dir, i, file))