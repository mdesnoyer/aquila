"""
This will make some test data for a trial run of Aquila
"""
import numpy as np
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
from glob import glob
from collections import Counter
from collections import defaultdict as ddict
import traceback

n_images = 20
n_trials = 16 * 300

img_dir = 'images'
win_dir = 'win_data'

FONT_PATH = os.environ.get("FONT_PATH", "/Library/Fonts/Menlo.ttc")
font = ImageFont.truetype(FONT_PATH, 100)

# make up some rando images
iw, ih = 320, 240

with open(os.path.join(win_dir, 'id_2_fn'), 'w') as f:
    for i in range(1, n_images+1):
        fn = os.path.join(img_dir, '%03i.jpg' % i)
        f.write('%i,%s\n' % (i-1, fn))
        if os.path.exists(fn):
            continue
        #im = Image.new('RGB', [iw,ih])
        im = np.random.randint(0, 100, [ih, iw]).astype(np.uint8)
        im = Image.fromarray(im)
        draw = ImageDraw.Draw(im)
        w, h = draw.textsize(str(i), font=font)
        cw = (iw - w) / 2
        ch = (ih - h) / 2 
        #draw.text((cw, ch), str(i), (255,255,255),font=font)
        draw.text((cw, ch), str(i), 255,font=font)
        im.save(fn)

# create a win list
outcomes = ddict(lambda: [0,0])
for i in range(n_trials):
    a, b = np.sort(np.random.choice(n_images, 2, replace=False))
    q = np.random.rand()
    if q < (a+1.)/(a+b+2.):
        # a wins
        outcomes[(a,b)][0] += 1
    else:
        outcomes[(a,b)][1] += 1
# let's generate it so it can be perfectly solved
# for i in range(n_trials):
#     a, b = np.random.choice(n_images, 2, replace=False)
#     if a > b:
#         outcomes[(a,b)][0] += 1
#     else:
#         outcomes[(a,b)][1] += 1

out_list = []
for (a, b), (c, d) in outcomes.iteritems():
    out_list.append([a,b,c,d])
np.save(os.path.join(win_dir, 'winlist'), np.array(out_list))
win_matrix = np.zeros((n_images, n_images))
for (a, b), (ab, ba) in outcomes.iteritems():
    win_matrix[a, b] = ab
    win_matrix[b, a] = ba

np.save(os.path.join(win_dir, 'winmatrix'), win_matrix)


