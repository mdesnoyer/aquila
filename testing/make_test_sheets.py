'''
This script produces test sheets of the best/worst for a given attribute.
'''

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from glob import glob
import numpy as np
from collections import defaultdict as ddict
import os

IMG_SRC = '/data/aquila_training_images'
COMP_WIDTH = 2500
MARGIN = 5
CNTR_MARGIN = 20
IM_DEST = '/data/bestworst_tiles'
_FONT_LOC = '/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-R.ttf'
font = ImageFont.truetype(_FONT_LOC, 100)


class PIU(object):
    '''Utilities for dealing with PIL images.'''

    def __init__(self):
        pass

    @classmethod
    def resize(cls, im, im_h=None, im_w=None):
        ''' Resize image.

        If either the desired height or the desired width are not
        specified, the aspect ratio is preserved.
        '''

        #TODO: instance check, PIL.Image check is ambigous,
        #type is JpegImagePlugin?

        if im_h is None and im_w is None:
            return im

        ar = im.size
        #resize on either dimension
        if im_h is None:
            im_h = int(float(ar[1])/ar[0] * im_w)

        elif im_w is None:
            im_w = int(float(ar[0])/ar[1] * im_h)

        image_size = (im_w, im_h)
        image = im.resize(image_size, Image.ANTIALIAS)
        return image

    @classmethod
    def create_random_image(cls, h, w):
        ''' return a random image '''
        return Image.fromarray(np.array(
            np.random.random_integers(0, 255, (h, w, 3)),
            np.uint8))

    @classmethod
    def to_cv(cls, im):
        '''Convert a PIL image to an OpenCV one in BGR format.'''
        if im.mode == 'RGB':
            return np.array(im)[:,:,::-1]
        elif im.mode in ['L', 'I', 'F']:
            return np.array(im)

        raise NotImplementedError(
            'Conversion for mode %s is not implemented' % im.mode)

    @classmethod
    def from_cv(cls, im):
        '''Converts an OpenCV BGR image into a PIL image.'''
        if len(im.shape) == 3:
            return Image.fromarray(im[:,:,::-1])
        else:
            return Image.fromarray(im)

    @classmethod
    def convert_to_rgb(cls, image):
        '''Convert the image to RGB if it is not.'''

        if image.mode == "RGBA":
            # Composite the image to a white background
            new_image = Image.new("RGB", image.size, (255,255,255))
            new_image.paste(image, mask=image)
            image = new_image
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return image

def create_horiz_stack(images):
    '''
    Creates a horizonal stack of images, resizing to the max height.

    :param images: A list of filenames.
    :return: The images, horizontally stacked.
    '''
    imarrs = [Image.open(x) for x in images]
    mxh = max([x.height for x in imarrs])
    # resize to the max height
    imarrs = [PIU.resize(x, im_h=mxh) for x in imarrs]
    tw = np.sum([x.width for x in imarrs]) + MARGIN * (len(imarrs) - 1)
    new_im = Image.new('RGB', (tw, mxh))
    x_offset = 0
    for im in imarrs:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + MARGIN
    return new_im

def create_bw_image(bw, fnum):
    """
    Creates a horizontal stack image

    :param bw: The dictionary of best/worst images.
    :param fnum: The feature number to use.
    :return: A stacked, labeled image.
    """
    goodstack = create_horiz_stack(bw[fnum]['best'][0])
    badstack = create_horiz_stack(bw[fnum]['worst'][0])
    goodstack = PIU.resize(goodstack, im_w=COMP_WIDTH)
    badstack = PIU.resize(badstack, im_w=COMP_WIDTH)
    w, h = font.getsize('Feature %i' % fnum)
    BW_H = goodstack.height + badstack.height + 2*CNTR_MARGIN + h
    textHC = goodstack.height + CNTR_MARGIN
    textWC = (COMP_WIDTH - w) / 2
    im = Image.new('RGB', (COMP_WIDTH, BW_H))
    im.paste(goodstack, (0, 0))
    im.paste(badstack, (0, goodstack.height + 2*CNTR_MARGIN + h))
    draw = ImageDraw.Draw(im)
    draw.text((textWC, textHC), 'Feature %i' % fnum, (255, 255, 255), font=font)
    return im

def create_bw_image_cust(best, worst, label):
    """
    Creates a horizontal stack image

    :param bw: The dictionary of best/worst images.
    :param fnum: The feature number to use.
    :return: A stacked, labeled image.
    """
    goodstack = create_horiz_stack(best[0])
    badstack = create_horiz_stack(worst[0])
    goodstack = PIU.resize(goodstack, im_w=COMP_WIDTH)
    badstack = PIU.resize(badstack, im_w=COMP_WIDTH)
    w, h = font.getsize(label)
    BW_H = goodstack.height + badstack.height + 2*CNTR_MARGIN + h
    textHC = goodstack.height + CNTR_MARGIN
    textWC = (COMP_WIDTH - w) / 2
    im = Image.new('RGB', (COMP_WIDTH, BW_H))
    im.paste(goodstack, (0, 0))
    im.paste(badstack, (0, goodstack.height + 2*CNTR_MARGIN + h))
    draw = ImageDraw.Draw(im)
    draw.text((textWC, textHC), label, (255, 255, 255), font=font)
    return im



print 'Reading files'
abstfiles = glob('/data/bestworst/abst*')
bw = ddict(lambda: dict())
for f in abstfiles:
    _, _, feat, ftype = f.split('_')
    feat = int(feat)
    dats = open(f).read().strip().split('\n')
    dats = [x.split(' ') for x in dats]
    scores = [float(x[1]) for x in dats]
    fns = [os.path.join(IMG_SRC, x[0].split('/')[-1]) for x in dats]
    if ftype == 'best':
        bw[feat]['best'] = [fns, scores]
    else:
        bw[feat]['worst'] = [fns, scores]

print 'creating images'
log_best = '/data/bestworst/logits_best'
log_worst = '/data/bestworst/logits_worst'
dats = open(log_best).read().strip().split('\n')
dats = [x.split(' ') for x in dats]
scores = [float(x[1]) for x in dats]
fns = [os.path.join(IMG_SRC, x[0].split('/')[-1]) for x in dats]
best = [fns, scores]
dats = open(log_worst).read().strip().split('\n')
dats = [x.split(' ') for x in dats]
scores = [float(x[1]) for x in dats]
fns = [os.path.join(IMG_SRC, x[0].split('/')[-1]) for x in dats]
worst = [fns, scores]
cim = create_bw_image_cust(best, worst, 'Neon Score')
cim.save(os.path.join(IM_DEST, 'logits.jpg'))

for i in bw.keys():
    img = create_bw_image(bw, i)
    img.save(os.path.join(IM_DEST, 'abstract_feat_%i.jpg' % i))