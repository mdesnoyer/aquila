"""
This generates a fake dataset, along with fake images.

UPDATE: NO LONGER MAKES FAKES, IT CROPS IMAGES FROM ACTUAL THUMBNAILS.
"""

from config import *
import os
from PIL import Image
import numpy as np
from collections import defaultdict as ddict
from glob import glob

n = 1000  # the number of images
ncomp = 10000  # the number of datapoints
ndemo = DEMOGRAPHIC_GROUPS
DATA_DIR = '/tmp/aquila_test_data'
all_ims = glob('/data/aquila_training_images/*')

def _make_fake(fn, x, y):
    '''
    generates a fake image with name filename and size x-by-y-by-3 and saves it.
    :param fn: filename
    :param x: height
    :param y: width
    :return: None
    '''
    im = np.random.randint(256, size=(x, y, 3)).astype(np.uint8)
    im = Image.fromarray(im)
    im.save(os.path.join(DATA_DIR, 'images', fn))

def make_fake(fn, x, y):
    im = Image.open(np.random.choice(all_ims))
    ax, ay = im.size
    min_dim = min(x, y, ax, ay)
    im = np.array(im)
    im = im[:min_dim, :min_dim, :]
    im = Image.fromarray(im)
    im = im.resize((x, y))
    im.save(os.path.join(DATA_DIR, 'images', fn))

# select some random scores for the images
image_fns = ['%03i.jpeg' % i for i in range(n)]
im_scores = {x: np.random.rand() for x in image_fns}

data = ddict(lambda: [0] * ndemo * 2)
for cmp in range(ncomp):
    # choose some random images
    a, b = np.random.choice(image_fns, 2, replace=False)
    # choose some random demo
    demo = np.random.randint(ndemo)
    if a > b:
        c = a
        a = b
        b = c
    a_thresh = im_scores[a] / (im_scores[a] + im_scores[b])
    if np.random.rand() < a_thresh:
        # a wins
        data[(a, b)][demo] += 1
    else:
        # b wins
        data[(a, b)][demo + ndemo] += 1

with open(os.path.join(DATA_DIR, 'combined'), 'w') as f:
    for key, cdata in data.iteritems():
        str1 = ','.join(key)
        str2 = ','.join([str(x) for x in cdata])
        f.write('%s,%s\n' % (str1, str2))

for fn in image_fns:
    make_fake(fn, 314, 314)



