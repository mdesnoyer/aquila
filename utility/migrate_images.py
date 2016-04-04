"""
Finding all the images that are referenced and move them to image_data
"""

from glob import glob
import shutil
import os

im2file = dict()

dest = '/data/aquila_training_images/'
ims1 = glob('/data/plutonium/images/*.jpg')
ims4 = glob('/data/plutonium/images2/*.jpg')
ims2 = glob('/data/images/*.jpg')
ims3 = glob('/data/plutonium/orig_images')

for i in ims4:
    im2file[i.split('/')[-1].split('.')[0]] = i

for i in ims3:
    im2file[i.split('/')[-1].split('.')[0]] = i

for i in ims1:
    im2file[i.split('/')[-1].split('.')[0]] = i

for i in ims2:
    im2file[i.split('/')[-1].split('.')[0]] = i

to_move = []
cant_find = []

with open('/repos/aquila/task_data/aggregated_data/idx_2_id', 'r') as f:
    for line in f:
        idx, cid = line.strip().split(',')
        # is fn in the im2file?
        if cid in im2file:
            to_move.append(im2file[cid])
        else:
            cant_find.append(cid)

for n, img in enumerate(to_move):
    dst = os.path.join(dest, img.split('/')[-1])
    shutil.copyfile(img, dst)
    print n,img
