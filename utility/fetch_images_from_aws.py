"""
This fetches the thumbnails from AWS and resize/pads them directly, without
storing an intermediate file.
"""
from PIL import Image
from glob import glob
import os
import boto3
import locale
import cStringIO
import urllib
from threading import Thread
from Queue import Queue

dst = '/data/aquila_training_images/'  # the destination folder
max_height = 314
max_width = 558
new_size = (max_width, max_height)
nthreads = 32
inQ = Queue(maxsize=nthreads*2)
outQ = Queue()

try:
    # for linux
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except:
    # for mac
    locale.setlocale(locale.LC_ALL, 'en_US')

def _parse_key(key):
    vals = key.split('.')[0].split('_')
    sim = float(vals[-1]) / 10000
    ttype = vals[-2]
    vid = '_'.join(vals[:-2])
    return vid, sim, ttype


def _yt_filt_func(key):
    _, sim, _ = _parse_key(key)
    return sim < 0.035 # from 0.027 to include more images.


def _s3sourcer(bucket_name, filter_func=None):
    """
    Constructs an iterator over the items in a bucket, returning those that
    pass the filter function. The iterator returns image id, url tuples.

    :param bucket_name: The name of the bucket to iterate over.
    :param filter_func: The filter function, which checks if an item should be
    yielded based on its key.
    :return: None.
    """
    if filter_func is None:
        filter_func = lambda x: True
    AWS_ACCESS_ID = 'AKIAIS3LLKRK7HDX4XYA'
    AWS_SECRET_KEY = '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'
    s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_ID,
                        aws_secret_access_key=AWS_SECRET_KEY)
    bucket_iter = iter(s3.Bucket(bucket_name).objects.all())
    base_url = 'https://s3.amazonaws.com/%s/%s'
    while True:
        item = bucket_iter.next()
        if filter_func(item.key):
            image_id = item.key.split('.')[0]
            image_url = base_url % (bucket_name, item.key)
            yield (image_id, image_url, item)
    return

sources = [_s3sourcer('neon-image-library'),
           _s3sourcer('mturk-youtube-thumbs', filter_func=_yt_filt_func)]

extant_imgs = glob(os.path.join(dst, '*.jpg'))
extant_imgs = set([x.replace(dst, '').replace('.jpg', '') for x in extant_imgs])

def fetcher(inQ, outQ):
    while True:
        item = inQ.get()
        if item is None:
            return
        imurl, nfn = item
        try:
            file = cStringIO.StringIO(urllib.urlopen(imurl).read())
        except IOError:
            print 'Could not fetch image at: %s' % imurl
            continue
        try:
            im = Image.open(file)
        except:
            print 'Could not convert image to PIL at: %s' % imurl
            continue
        w, h = im.size
        resize_ratio = min(max_width * 1./w, max_height * 1./h)
        nw, nh = int(w * resize_ratio), int(h * resize_ratio)
        rimd = im.resize((nw, nh), Image.ANTIALIAS)
        # # do not pre-pad
        # old_size = rimd.size
        # new_im = Image.new("RGB", new_size)
        # new_im.paste(rimd, ((new_size[0]-old_size[0])/2,
        #                     (new_size[1]-old_size[1])/2))
        # fin_im = new_im.resize((max_height, max_height), Image.ANTIALIAS)
        fin_im = rimd
        fin_im.save(nfn)
        outQ.put(1)

tot = 0
tot_req = 0
tot_obt = 0
threads = []
for t in range(nthreads):
    threads.append(Thread(target=fetcher, args=(inQ, outQ)))

print 'Starting threads'
for t in threads:
    t.start()

for source in sources:
    for imid, imurl, obj in source:
        tot += 1
        if not tot % 1000:
            tot_s = locale.format("%d", tot, grouping=True)
            tr_s = locale.format("%d", tot_req, grouping=True)
            to_s = locale.format("%d", tot_obt, grouping=True)
            print '%s total, %s requested, %s obtained' % (tot_s, tr_s, to_s)
        nfn = os.path.join(dst, imid + '.jpg')
        if os.path.exists(nfn):
            continue
        if imid in extant_imgs:
            continue
        inQ.put((imurl, nfn))
        tot_req += 1
        while True:
            try:
                inc = outQ.get_nowait()
                tot_obt += inc
            except:
                break

for i in range(nthreads * 2):
    inQ.put(None)

print 'Joining threads'
for i in threads:
    i.join()

while True:
    try:
        inc = outQ.get_nowait()
        tot_obt += inc
    except:
        break

tot_s = locale.format("%d", tot, grouping=True)
tr_s = locale.format("%d", tot_req, grouping=True)
to_s = locale.format("%d", tot_obt, grouping=True)
print '%s total, %s requested, %s obtained' % (tot_s, tr_s, to_s)
