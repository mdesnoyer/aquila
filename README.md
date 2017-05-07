# Overview

This repository contains the Aquila model as implemented in [TensorFlow](https://www.tensorflow.org/). It predicts users' relative ranking of images from a large number of noisy individual ranking pairs. Demographic buckets of users can also be specified for those applications where the ranking is likely to vary by demographic. For more details on the mathematics of the model and the training process, see [the white paper](aquila-learning-predict.pdf).

# Architecture

The model consists of an [Inception-v3](https://github.com/tensorflow/models/tree/master/inception) tower with one output node for the final prediction and a second to last layer of 1024 abstract features, which are used for demographic targeting. Using the final prediction, the estimated lift of image A over image B is defined by:

```
lift = exp(A) / exp(B) - 1
```

# Training

In order to train the model, with your own data, you need

TODO: Describe the data formats of all the inputs
* A collection of images
* A win matrix
* A file mapping indices in the win matrix to image ids

Once the data is collected, edit the directories in <init_train.py> and <config.py> to point to your data and run:

```shell
python init_train.py
```

# Pre-trained Model

A pre-trained version of the model is available [here](https://www.dropbox.com/s/3af8auuovksidm7/aquila_model.tar.gz?dl=0). This version is trained on the valence experiments performed by Neon Labs Inc.. It has been trained on approximately 3.6M video frames extracted from random videos on [YouTube](https://www.youtube.com) that have been rated ~25M times by US users of Mechanical Turk.

# Serving Module

The Tensorflow Serving module for this model is available at <https://github.com/mdesnoyer/aquila_serving_module>. It can be used to efficiently serve inference requests for scores from gRPC.