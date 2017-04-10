import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import urllib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis
from lib.rng import t_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score


def transform(X):
    return (floatX(X)/255.).reshape(-1, nc, npx, npx)

def inverse_transform(X):
    X = X.reshape(-1, npx, npx)
    return X

desc = 'vae_conv_test3'
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)


import tempfile
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


k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
b1 = 0.1          # momentum term of adam
nc = 1            # # of channels in image
ny = 10           # # of classes
# nbatch = 128      # # of examples in batch
nbatch = 100
npx = 28          # # of pixels width/height of images
nz = 32           # # of dim for Z
ngfc = 512       # # of gen units for fully connected layers
ndfc = 512       # # of discrim units for fully connected layers
ngf = 16          # # of gen filters in first conv layer
ndf = 16          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 200       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 3e-4       # initial learning rate for adam


relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
#gifn = inits.Uniform(scale=0.05)
#difn = inits.Uniform(scale=0.05)
bias_ifn = inits.Constant(c=0.)


ew1 = gifn((ngf, nc, 5, 5), 'ew1')
ew2 = gifn((ngf*2, ngf, 5, 5), 'ew2')
ew3 = gifn((ngf*2, ngf*2, 5, 5), 'ew3')
ew4 = gifn((ngf*2*7*7, ngfc), 'ew4')
eb4 = bias_ifn((ngfc,), 'eb4')
ew_mu = gifn((ngfc, nz), 'ew_mu')
eb_mu = bias_ifn((nz,), 'eb_mu')
ew_sig = gifn((ngfc, nz), 'ew_sig')
eb_sig = bias_ifn((nz,), 'eb_sig')

dw1 = difn((nz, ngfc), 'dw1')
db1 = bias_ifn((ngfc,), 'db1')
dw2 = difn((ngfc, ngf*2*7*7), 'dw2')
db2 = bias_ifn((ngf*2*7*7,), 'db2')
dw3 = difn((ndf*2, ndf*2, 5, 5), 'dw3')
#dw4 = difn((ndf*2, ndf, 5, 5), 'dw4')
#dw5 = difn((ndf, nc, 5, 5), 'dw5')
dw4 = difn((ndf, ndf*2, 5, 5), 'dw4')
dw5 = difn((nc, ndf, 5, 5), 'dw5')

# enc_params = [ew1, ew2, ew3, ew4, eb4, ew_mu, eb_mu, ew_sig, eb_sig]
# dec_params = [dw1, db1, dw2, db2, dw3, dw4, dw5]
enc_params = [ew1, ew2, ew4, eb4, ew_mu, eb_mu, ew_sig, eb_sig]
dec_params = [dw1, db1, dw2, db2, dw4, dw5]

params = enc_params + dec_params


from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv2d
def conv(X, w, s = 2, b = None, activation = relu):
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return activation(z)


def conv_and_pool(X, w, s = 2, b = None, activation=relu, subsample=(2,2)):
    return pool_2d(conv(X, w, s, b, activation=activation), subsample, ignore_border=True)


def deconv(X, w, s=2, b=None):
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return z


def depool(X, factor=2):
    """
    luke perforated upsample
    http://www.brml.org/uploads/tx_sibibtex/281.pdf
    """
    output_shape = [
        X.shape[1],
        X.shape[2]*factor,
        X.shape[3]*factor
    ]
    stride = X.shape[2]
    offset = X.shape[3]
    in_dim = stride * offset
    out_dim = in_dim * factor * factor

    upsamp_matrix = T.zeros((in_dim, out_dim))
    rows = T.arange(in_dim)
    cols = rows*factor + (rows/stride * factor * offset)
    upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)

    flat = T.reshape(X, (X.shape[0], output_shape[0], X.shape[2] * X.shape[3]))

    up_flat = T.dot(flat, upsamp_matrix)
    upsamp = T.reshape(up_flat, (X.shape[0], output_shape[0],
                                 output_shape[1], output_shape[2]))

    return upsamp


def deconv_and_depool(X, w, s=2, b=None, activation=T.nnet.relu):
    return activation(deconv(depool(X, s), w, s, b))


#def conv_encoder(X, w1, w2, w3, w4, b4, w_mu, b_mu, w_sig, b_sig):
def conv_encoder(X, w1, w2, w4, b4, w_mu, b_mu, w_sig, b_sig):
    h1 = conv_and_pool(X, w1, s=2)
    h2 = conv_and_pool(h1, w2, s=2)
    # h3 = conv(h2, w3, s=2)
    #h1 = relu((dnn_conv(X, w1, subsample=(1, 1), border_mode=(1, 1))))
    #h2 = relu((dnn_conv(h1, w2, subsample=(2, 2), border_mode=(2, 2))))
    #h3 = relu((dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))

    h3 = T.flatten(h2, 2)
    h4 = tanh((T.dot(h3, w4) + b4))

    mu = T.dot(h4, w_mu) + b_mu
    log_sigma = T.dot(h4, w_sig) + b_sig

    return mu, log_sigma



#def conv_decoder(X, Z, w1, b1, w2, b2, w3, w4, wx):
def conv_decoder(X, Z, w1, b1, w2, b2, w4, wx):
    h1 = relu((T.dot(Z, w1)+b1))
    h2 = relu((T.dot(h1, w2)+b2))
    h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
    #h3 = relu((deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    #h4 = relu((deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2))))
    #reconstructed_x = sigmoid(deconv(h4, wx, subsample=(1, 1), border_mode=(1, 1)))
    # h3 = deconv(h2, w3, s=2)
    h4 = deconv_and_depool(h2, w4, s=2, activation=T.nnet.relu)
    reconstructed_x = deconv_and_depool(h4, wx, s=2, activation=sigmoid)

    # x1 = T.flatten(reconstructed_x, 2)
    # x2 = T.flatten(X, 2)
    # logpxz = - T.nnet.binary_crossentropy(x1, x2).sum(axis=1)
    logpxz = - T.sum(T.sum(T.nnet.binary_crossentropy(reconstructed_x, X), axis=1))
    return reconstructed_x, logpxz



def sampler(mu, log_sigma):

    eps = t_rng.normal(mu.shape)
    # Reparametrize
    z = mu + T.exp(0.5 * log_sigma) * eps
    # z = mu + T.exp(log_sigma) * eps
    return z



import math
from pylearn2.expr.basic import log_sum_exp
def marginal_loglikelihood(X, num_samples = 512):
    mu, log_sigma = conv_encoder(X, *enc_params)

    epsilon_shape = (num_samples, X.shape[0], mu.shape[1])
    epsilon = t_rng.normal(epsilon_shape)

    mu = mu.dimshuffle('x', 0, 1)
    log_sigma = log_sigma.dimshuffle('x', 0, 1)
    #log_sigma = log_sigma * 2.

    # compute z
    z = mu + T.exp(0.5 * log_sigma) * epsilon

    # Decode p(x | z) in roder to do flatten MLP compatible
    flat_z = z.reshape((epsilon.shape[0] * epsilon.shape[1],
            epsilon.shape[2]))

    reconstructed_x, _ = conv_decoder(X, flat_z, *dec_params)
    reconstructed_x = reconstructed_x.reshape((epsilon.shape[0], epsilon.shape[1], X.shape[1] * X.shape[2] * X.shape[3]))

    # compute log-probabilities
    log_q_z_x = -0.5 * (T.log(2 * math.pi) + log_sigma + (z - mu) ** 2 / T.exp(log_sigma)).sum(axis=2)
    log_p_z = -0.5 * (T.log(2 * math.pi) + (z ** 2)).sum(axis=2)

    # if self.continuous:
    #     # need to rewrite and finish this
    #     log_p_x_z = -0.5 * (T.log(2 * math.pi) + self.gauss_sigma + (X.dimshuffle('x', 0, 1) - reconstructed_x) ** 2 /T.exp(self.gauss_sigma)).sum(axis=2)
    # else:
    X_flatten = X.flatten(2)
    log_p_x_z = - T.nnet.binary_crossentropy(reconstructed_x, X_flatten.dimshuffle('x', 0, 1)).sum(axis=2)

    return T.mean( log_sum_exp(
            log_p_z + log_p_x_z - log_q_z_x,
            axis=0
            ) - T.log(T.cast(num_samples, 'float32'))  )


X = T.tensor4()

mu, log_sigma = conv_encoder(X, *enc_params)

z = sampler(mu, log_sigma)

reconstructed_x, logpxz = conv_decoder(X, z, *dec_params)

# Expectation of (logpz - logqz_x) over logqz_x is equal to KLD (see appendix B):
KLD = 0.5 * T.sum(1 + log_sigma - mu**2 - T.exp(log_sigma))

# Average over batch dimension
# logpx = T.mean(logpxz + KLD)
logpx = logpxz + KLD

cost = -1 * logpx
lrt = sharedX(lr)
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
# g_updater = updates.Adam(lr=lrt)
g_updates = g_updater(params, cost)


X_train, X_valid, X_test = mnist()
ntrain, nvalid, ntest = len(X_train), len(X_valid), len(X_test)
print X_train.shape, X_valid.shape, X_test.shape


print 'COMPILING'
t = time()
_train_g = theano.function([X], cost, updates = g_updates)
_likelihood = theano.function([X], logpx)
_encoder = theano.function([X], z)
_decoder = theano.function([z], reconstructed_x)
_marginal = theano.function([X], marginal_loglikelihood(X))
print '%.2f seconds to compile theano functions'%(time()-t)


def cal_margin():
    ll = 0
    teX = shuffle(X_test)

    m = 1000
    batch_size = ntest // m
    for i in range(m):
        batch = [t % ntest for t in range(i*batch_size, (i+1)*batch_size)]
        imb = floatX(teX[batch])
        ll += _marginal(imb) * len(batch)

    return ll / ntest


n_updates = 0
n_epochs = 0

t = time()
zmb = floatX(np_rng.normal(0, 1, size=(100, nz)))
for epoch in range(1, niter+niter_decay+1):
    X_train = shuffle(X_train)

    ll = 0
    for imb in tqdm(iter_data(X_train, size=nbatch), total=ntrain/nbatch):
        imb = floatX(imb)

        _train_g(imb)

        n_updates+=1

        ll += _likelihood(imb) * len(imb)

    print epoch, 'VLB', ll / ntrain / nbatch


    samples = floatX(_decoder(zmb))
    grayscale_grid_vis(inverse_transform(samples), (10, 10), 'samples/%s/%d.png'%(desc, epoch))


    if epoch == 1 or epoch % 50 == 0:
        print epoch, 'LL', cal_margin()

        joblib.dump([p.get_value() for p in enc_params], 'models/%s/%d_gen_params.jl'%(desc, n_epochs))
        joblib.dump([p.get_value() for p in dec_params], 'models/%s/%d_discrim_params.jl'%(desc, n_epochs))











