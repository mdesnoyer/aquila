"""
Measures the mean image aspect ratio, and stores a histogram of aspect ratios
as width / height.
"""

# from PIL import Image
from glob import glob
from collections import Counter
import numpy as np
import cv2

ims = glob('/data/aquila_training_images/*')
np.random.shuffle(ims)
asp_hist = Counter()
area_hist = Counter()
# further, bin the counts to the second decimal
for n, im in enumerate(ims):
    if not n % 1000:
        print n, len(ims)
    # x = Image.open(im)
    x = cv2.imread(im)
    h, w, c = x.shape
    aspr = np.around(float(w)/h, 2)
    asp_hist[aspr] += 1
    area_hist[w*h] += 1
