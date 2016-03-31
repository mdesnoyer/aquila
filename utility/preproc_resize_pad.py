"""
This replicates the functionality of preproc_resize.py, but also pads the
images (with black) to 558 x 314, then resizes it to 314 x 314.
"""

from PIL import Image
from glob import glob
import os

ims = glob('/data/aquila_training_images_unresized/*')
dst = '/data/images/'

max_height = 314
max_width = 558
new_size = (max_width, max_height)
for n, im in enumerate(ims):
    nfn = os.path.join(dst, im.split('/')[-1])
    imd = Image.open(im)
    w, h = imd.size
    asp = float(w) / h
    if asp >= (16./9):
        # then it's too wide. resize width
        nh = int(max_width / asp)
        rsz_w, rsz_h = max_width, nh
    else:
        # then it's too tall. resize height
        nw = int(max_height * asp)
        rsz_w, rsz_h = nw, max_height
    rimd = imd.resize((rsz_w, rsz_h), Image.ANTIALIAS)
    old_size = rimd.size
    new_im = Image.new("RGB", new_size)
    new_im.paste(rimd, ((new_size[0]-old_size[0])/2,
                        (new_size[1]-old_size[1])/2))
    fin_im = new_im.resize((max_height, max_height), Image.ANTIALIAS)
    fin_im.save(nfn)
    print '%i/%i - %s' % (n+1, len(ims), im[-15:])