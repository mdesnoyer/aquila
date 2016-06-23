"""
This script gets a win list, just like Aquila, and then attempts to find all of
the images. It then computes the mean R, G, and B values.
"""
from PIL import Image
import numpy as np
from glob import glob
import os

im_src = '/data/aquila_training_images'  # the image source
data_src = '/data/aquila_v2/combined'

unique_ims = set()

print 'Finding unique images'
with open(data_src, 'r') as f:
    for line in f:
        spt = line.strip().split(',')
        im1 = spt[0]
        im2 = spt[1]
        unique_ims.add(im1)
        unique_ims.add(im2)


print 'Computing mean values'
mean_vals = [0, 0, 0]
count = 0
not_avail = set()
for im in unique_ims:
    imfn = os.path.join(im_src, im)
    if os.path.exists(imfn):
        x = np.array(Image.open(imfn).convert('RGB'))
        r, g, b = np.mean(x, (0, 1))
        mean_vals[0] += r
        mean_vals[1] += g
        mean_vals[2] += b
        count += 1
        if not count % 1000:
            mr = mean_vals[0] / count
            mg = mean_vals[1] / count
            mb = mean_vals[2] / count
            print '%i: r:%.3f g:%.3f b:%.3f (unav: %i)' % (count, mr, mg, mb,
                                                           len(not_avail))
    else:
        not_avail.add(im)

mr = mean_vals[0] / count
mg = mean_vals[1] / count
mb = mean_vals[2] / count
print 'r:%g g:%g b:%g' % (mr, mg, mb)