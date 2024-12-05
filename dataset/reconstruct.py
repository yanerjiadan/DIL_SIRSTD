import os
import shutil
import random
import json

nuaa_dir = '/mnt/e/Dataset/NUAA-SIRST'
nudt_dir = '/mnt/e/Dataset/NUDT-SIRST'
irstd_dir = '/mnt/e/Dataset/IRSTD-1k'
target_dir = '/mnt/e/Dataset/DIL_SIRSTD'

nuaa_image_dir = os.path.join(nuaa_dir, 'images/images')
nuaa_mask_dir = os.path.join(nuaa_dir, 'masks/masks')
nudt_image_dir = os.path.join(nudt_dir, 'images')
nudt_mask_dir = os.path.join(nudt_dir, 'masks')
irstd_image_dir = os.path.join(irstd_dir, 'IRSTD1k_Img')
irstd_mask_dir = os.path.join(irstd_dir, 'IRSTD1k_Label')
target_image_dir = os.path.join(target_dir, 'images')
target_mask_dir = os.path.join(target_dir, 'masks')

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






# for i in range(7):
#     current_dir = os.path.join(target_image_dir, str(i))
#     img_list = []
#     for root, dirs, files in os.walk(current_dir):
#         for file in files:
#             if file.endswith('.png'):
#                 img_list.append(file)
#         train_len = int(len(img_list)*0.8)
#         random.shuffle(img_list)
#         train_files = img_list[:train_len]
#         test_files = img_list[train_len:]
#         train_files.sort()
#         test_files.sort()
#         with open(os.path.join(current_dir, 'train.json'), 'w') as f:
#             json.dump(train_files, f)
#         with open(os.path.join(current_dir, 'test.json'), 'w') as f:
#             json.dump(test_files, f)

for i in '1234':
    for root, dirs, files in os.walk(os.path.join(target_image_dir, i)):
        for file in files:
            if file.endswith('.png'):
                mask_name = '0'+file[1:]
                shutil.copy(os.path.join(nudt_mask_dir, mask_name), os.path.join(target_mask_dir, i, file))