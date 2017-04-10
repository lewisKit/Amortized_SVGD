import sys
sys.path.append('..')

from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from utils import *
from svgd import *
import shutil
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

desc = 'steinvae_conv_35'
model_dir = 'models/%s'%desc
images_dir = 'images/%s'%desc
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# copy current file to model_dir
file_name = 'steinvae.py'
dist_dir = os.path.join(model_dir, file_name)
shutil.copy(file_name, dist_dir)


k = 1             # # of discrim updates for each gen update
l2 = 1e-4         # l2 weight decay
b1 = 0.1          # momentum term of adam
nc = 1            # # of channels in image
ny = 10           # # of classes
# nbatch = 128      # # of examples in batch
nbatch = 100
npx = 28          # # of pixels width/height of images
nz = 32           # # of dim for Z
ngfc = 512       # # of en units for fully connected layers
ndfc = 512       # # of de units for fully connected layers
ngf = 16          # # of en filters in first conv layer
ndf = 16          # # of de filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 200       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
de_lrt = 1e-4
en_lrt = 1e-4
drop_p = 0.3   # dropout rate

relu = activations.Rectify()
sigmoid = activations.Sigmoid
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

#enc_params = [ew1, ew2, ew4, eb4, ew_mu, eb_mu, ew_sig, eb_sig]
enc_params = [ew1, ew2, ew4, eb4, ew_mu, eb_mu]
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


def conv_encoder(X, w1, w2, w4, b4, w_mu, b_mu):
    h1 = conv_and_pool(X, w1, s=2)
    h1 = dropout(h1, 0.3)
    h2 = conv_and_pool(h1, w2, s=2)
    h2 = dropout(h2, 0.3)
    # h3 = conv(h2, w3, s=2)
    #h1 = relu((dnn_conv(X, w1, subsample=(1, 1), border_mode=(1, 1))))
    #h2 = relu((dnn_conv(h1, w2, subsample=(2, 2), border_mode=(2, 2))))
    #h3 = relu((dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    h3 = T.flatten(h2, 2)
    h4 = tanh((T.dot(h3, w4) + b4))
    h4 = dropout(h4, 0.3)

    z = T.dot(h4, w_mu) + b_mu
    return z


def conv_decoder(X, Z, w1, b1, w2, b2, w4, wx):
    h1 = relu((T.dot(Z, w1)+b1))
    h2 = relu((T.dot(h1, w2)+b2))
    h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
    #h3 = relu((deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    #h4 = relu((deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2))))
    #reconstructed_x = sigmoid(deconv(h4, wx, subsample=(1, 1), border_mode=(1, 1)))
    # h3 = deconv(h2, w3, s=2)
    h4 = deconv_and_depool(h2, w4, s=2, activation=T.nnet.relu)
    reconstructed_x = deconv_and_depool(h4, wx, s=2, activation=T.nnet.sigmoid)

    logpxz = T.reshape(- T.sum(T.nnet.binary_crossentropy(reconstructed_x, X), [-1, -2]), [-1])

    return reconstructed_x, logpxz


def _vgd_gradient(z, num_z, logpxz):
    tensor_z = T.reshape(z, (-1, num_z, nz))
    Kxy, dxkxy, h = vgd_kernel_tensor(tensor_z)

    dz_logpzx = T.grad(T.sum(logpxz), z) - z
    tensor_grad_z = T.reshape(dz_logpzx, (-1, num_z, nz))


    # vgd_grad_tensor = (T.batched_dot(Kxy, tensor_grad_z) + dxkxy) / T.cast(num_z, 'float32')
    vgd_grad_tensor = (T.batched_dot(Kxy, tensor_grad_z) + 100 * dxkxy) / T.tile(\
         T.mean(Kxy, axis=2).dimshuffle(0, 1, 'x'), (1, 1, nz))

    return vgd_grad_tensor


X = T.tensor4('X')
num_z = T.iscalar('num_z')

# for generatiing image functions

func_x_corrupt = dropout(X, p=drop_p)
func_x_corrupt =T.clip(func_x_corrupt, 1e-6, 1 - 1e-6)

func_z = conv_encoder(func_x_corrupt, *enc_params)
func_res_x, _ = conv_decoder(X, func_z, *dec_params)

# functions for svgd training
x_repeated = T.repeat(X, num_z, axis=0)
x_dropout = dropout(x_repeated, p=drop_p)
x_corrupt = T.clip(x_dropout, 1e-6, 1- 1e-6)
# x_corrupt = x_dropout

z = conv_encoder(x_corrupt, *enc_params)
reconstructed_x, logpxz = conv_decoder(x_repeated, z, *dec_params)

z_vgd_grad = 0. - _vgd_gradient(z, num_z, logpxz)

# L operator
dHdPhi = T.Lop(
            f=z.flatten() / T.cast(num_z * nbatch, 'float32'),
            wrt=enc_params,
            eval_points=z_vgd_grad.flatten())


en_updater = updates.GAdam(lr=sharedX(en_lrt), regularizer=updates.Regularizer(l2=l2))
en_updates = en_updater(enc_params, dHdPhi)

decost = 0 - logpxz.sum() / T.cast(num_z * nbatch, 'float32')
de_updater = updates.Adam(lr=sharedX(de_lrt), regularizer=updates.Regularizer(l2=l2))
de_updates = de_updater(dec_params, decost)

gupdates = en_updates + de_updates



X_train, X_valid, X_test = mnist()
ntrain, nvalid, ntest = len(X_train), len(X_valid), len(X_test)
print X_train.shape, X_valid.shape, X_test.shape


print 'COMPILING'
t = time()
_train = theano.function([X, num_z], decost, updates=gupdates)
_reconstruct = theano.function([X], func_res_x)
_encoder = theano.function([X], func_z)
_decoder = theano.function([func_z], func_res_x)
print '%.2f seconds to compile theano functions'%(time()-t)


n_updates = 0
n_epochs = 0

t = time()
zmb = floatX(np_rng.normal(0, 1, size=(100, nz)))
xmb = floatX(shuffle(X_test)[:100])
number_z = 5

for epoch in range(1, niter+niter_decay+1):
    X_train = shuffle(X_train)

    logpxz = 0
    for imb in tqdm(iter_data(X_train, size=nbatch), total=ntrain/nbatch):
        imb = floatX(imb)

        logpxz += _train(imb, number_z) * len(imb)

        n_updates+=1

    print epoch, 'logpxz', logpxz / ntrain


    if epoch == 1 or epoch % 5 == 0:
        samples = floatX(_decoder(zmb))
        grayscale_grid_vis(inverse_transform(samples), (10, 10), 'images/%s/samples_%d.png'%(desc, epoch))
        rec_x = _reconstruct(xmb)
        grayscale_grid_vis(inverse_transform(rec_x), (10, 10), 'images/%s/rec_x%d.png'%(desc, epoch))


    if epoch == 1 or epoch % 100 == 0:
        joblib.dump([p.get_value() for p in enc_params], 'models/%s/%d_en_params.jl'%(desc, n_epochs))
        joblib.dump([p.get_value() for p in dec_params], 'models/%s/%d_de_params.jl'%(desc, n_epochs))


