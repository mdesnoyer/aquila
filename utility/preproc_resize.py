"""
Resizes the training images appropriately. Here's how it will work:

Desired Aspect Ratio (w / h): da
Desired Width: dw
-------------------------------
- Desired height: dw / da

For an image x with shape w, h:
    - if w/h > da:
        Then it's wider than it should be, which will be cropped off.
         Resize so height is dw / da.
    - if w/h <= da:
        Then it's taller than it should be, which will be cropped off.
        Resize so width is dw.

Result:
    - Images whose width is not more than dw
    - Images whose height is not more than dw / da

- Inception expects images of size 299 x 299
- 60% of our images are aspect ratio 16:9
- We want a ~5 variance in the random cropping
- Resize all images to 314 (max height) or 556 (max width)

- In the net:
    - resize_image_with_crop_or_pad to 556 x 314
    - resize image to 314 x 314
    - random crop to 299 x 299
    - random flip? (??)
        - Yeah, while there probably is a "chirality" in terms of images,
          we don't have enough data to capture it anyway.
"""

from PIL import Image
from glob import glob
import os

ims = glob('/data/aquila_training_images_unresized/*')
dst = '/data/images/'
# ims = glob('/data/aquila_training_images/*')

max_height = 314
max_width = 558

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
    rimd.save(nfn)
    print '%i/%i - %s' % (n+1, len(ims), im[-15:])
