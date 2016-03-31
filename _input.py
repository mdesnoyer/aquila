"""
Creates an InputManager to hide the latency behind multithreading (in
python). There may be some overhead still with transferring data from python
to Aquila, but that remains to be seen.

I'm hardcoding the image resolutions. Deal with it.
"""
from glob import glob
import numpy as np
import os
import tensorflow as tf


def _proc_ims(fn_tensor):
    '''
    Returns a tensorflow representation of an image that has been fully
    preprocessed and ready to be placed in a queue.

    :param fn_tensor: A filename tensor.
    :return: The representation of the preprocessed image.
    '''
    # read the image
    raw_im_ = tf.read_file(fn_tensor)
    # decode the image
    im_ = tf.image.decode_jpeg(raw_im_)
    # pad / crop (but really only pad) the image to 16:9 556-by-314 size.
    pc_im_ = tf.image.resize_image_with_crop_or_pad(im_, 314, 556)
    # squeeze the image to 314-by-314
    pc_im_ = tf.image.resize_images(pc_im_, 314, 314)
    # random crop the image
    pc_im_ = tf.random_crop(pc_im_, [314, 314, 3])
    # random flip left/right
    pc_im_ = tf.image.random_flip_left_right(pc_im_)
    return pc_im_


def _enqueue_op(fn_tensors, lab_tensors, tf_queue):
    '''
    Places a batch of images and labels into the tensorflow queue. These
    annoyingly have to be done in lockstep since there will be multiple
    producer threads.

    :param fn_tensors: A list of filename tensors.
    :param lab_tensors: A list of label tensors.
    :param tf_queue: The tensorflow input queue.
    :return: The enqueue operation.
    '''
    pre_resized = [_proc_ims(x) for x in fn_tensors]


def _worker(inq, outq, fnmap, batchsize):
    """
    A worker that loads and preprocesses images for the output tensorflow
    queue.

    :param inq: The input queue of indices to lookup.
    :param outq:
    :return:
    """

def _get_map_dict(mapfile, imdir):
    """
    Creates a mapping dict to map indices to filenames.

    :param mapfile:
    :return: A dictionary of indices to filenames.
    """
    odict = dict()
    files = glob(os.path.join(imdir, '*'))
    file_dict = {x.split('/')[-1].split('.')[0]: x for x in files}

    with open(mapfile, 'r') as f:
        for line in f:
            idx, cid = line.split(',')
            odict[int(idx)] = file_dict[cid]

    return odict


class InputManager(object):
    def __init__(self, imdir, winmtx, mapfile, tfqueue, nepochs, nthreads,
                 batchsize, imw=228, imh=228, randc=False):
        """
        Instantiates the InputManager class.

        :param imdir: String, the path to the image directory.
        :param winmtx: String, path to the win matrix numpy object.
        :param mapfile: String, path to the csv idx-to-id map.
        :param tfqueue: TensorFlow input queue.
        :param nepochs: Int, the number of epochs to run for.
        :param nthreads: Int, the number of threads to spawn to read the files.
        :param batchsize: Int, the batch size.
        :param imw: Int, the image width (in pixels).
        :param imh: Int, the image height (in pixels).
        :return: An InputManager object.
        """
        self.map = _get_map_dict(mapfile, imdir)
        self.winmtx = np.load(winmtx)
        self.tfq = tfqueue
        self.nepochs = nepochs
