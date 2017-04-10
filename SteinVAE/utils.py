import os
import json
import tempfile
import urllib
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def mnist(datasets_dir='/TMP/'):
    URL_MAP = {
    "train": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
    "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
    "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat"
    }

    PATH_MAP = {
    "train": os.path.join(tempfile.gettempdir(), "binarized_mnist_train.npy"),
    "valid": os.path.join(tempfile.gettempdir(), "binarized_mnist_valid.npy"),
    "test": os.path.join(tempfile.gettempdir(), "binarized_mnist_test.npy")
    }
    for name, url in URL_MAP.items():
        local_path = PATH_MAP[name]
        if not os.path.exists(local_path):
            np.save(local_path, np.loadtxt(urllib.urlretrieve(url)[0]))

    train_set = [x for x in np.load(PATH_MAP['train'])]
    valid_set = [x for x in np.load(PATH_MAP['valid'])]
    test_set =  [x for x in np.load(PATH_MAP['test'])]

    x_train = np.array(train_set).astype(np.float32)
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_valid = np.array(valid_set).astype(np.float32)
    x_valid = x_valid.reshape(x_valid.shape[0], 1, 28, 28)
    x_test = np.array(test_set).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    return x_train, x_valid, x_test


np.random.seed(1234)
rng = np.random.RandomState(1234)
theano_rng = RandomStreams(rng.randint(999999))

def dropout(X, p=0.):

    retain_prob = 1 - p
    return T.switch(T.eq(p, 0.), X, X * theano_rng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX) / retain_prob)


def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]
