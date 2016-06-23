"""
fetch the data from the old training image set, resize, and put them in their
proper place.
"""
from glob import glob
from PIL import Image
import os

im_src = '/data/aquila_training_images_unresized'
im_dst = '/data/aquila_training_images'

ims = glob(os.path.join(im_src, '*'))
max_height = 314
max_width = 558
copied = 0
for n, cim in enumerate(ims):
    if not n % 1000:
        print '%i found, %i moved' % (n, copied)
    _nfn = cim.split('/')[-1].split('.')[0]
    nfn = os.path.join(im_dst, _nfn + '.jpg')
    if os.path.exists(nfn):
        continue
    im = Image.open(cim)
    w, h = im.size
    resize_ratio = min(max_width * 1./w, max_height * 1./h)
    nw, nh = int(w * resize_ratio), int(h * resize_ratio)
    rimd = im.resize((nw, nh), Image.ANTIALIAS)
    rimd.save(nfn)
    copied += 1