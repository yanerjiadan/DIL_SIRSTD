import PIL.Image as Image
import numpy as np
import time

array = np.array([0.1,0.9])
array = (array>0.5)*255
print(array)

def binarize(img, threshold=127.4):
    img = np.array(img)
    img = (img > threshold).astype(np.uint8)*255
    img = Image.fromarray(img)
    return img


img = Image.open('/mnt/e/Dataset/DIL_SIRSTD/masks/5/200005.png')

nearest = img.resize((384,384), Image.NEAREST)
bilinear = img.resize((384,384), Image.BILINEAR)
bicubic = img.resize((384,384), Image.BICUBIC)
binary = binarize(bicubic)

img.save('img.png')
nearest.save('nearest.png')
bilinear.save('bilinear.png')
bicubic.save('bicubic.png')
binary.save('binary.png')
