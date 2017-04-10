import sys
sys.path.append('..')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
import theano
import theano.tensor as T
import scipy.io
from lib import activations
from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng, t_rng
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file


'''
    Random seed
'''
seed = 42
if "gpu" in theano.config.device:
    srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
else:
    srng = T.shared_randomstreams.RandomStreams(seed=seed)


'''
    For the adult datasets,
    we test on a9a dataset and test on a1a - a8a
'''

X_train_1, y_train_1 = load_svmlight_file('data/a9a.txt', n_features=123, dtype=np.float32)
X_train_2, y_train_2 = load_svmlight_file('data/a9a.t', n_features=123, dtype=np.float32)
X_train = floatX(np.concatenate([X_train_1.toarray(), X_train_2.toarray()], axis=0))
y_train = floatX(np.concatenate([y_train_1, y_train_2], axis=0))


total_dev = []
total_test = []

for i in range(1, 9):
    name_dev = 'data/a' + str(i)+ 'a.txt'
    name_test = 'data/a' + str(i) + 'a.t'

    X_dev, y_dev = load_svmlight_file(name_dev, n_features=123, dtype=np.float32)
    X_test, y_test = load_svmlight_file(name_test, n_features=123, dtype=np.float32)
    total_dev.append((floatX(X_dev.toarray()), floatX(y_dev)))
    total_test.append((floatX(X_test.toarray()), floatX(y_test)))


N_train = X_train.shape[0]
print "size of training=%d" % (N_train)

x_dim = X_train.shape[1]


a0 = 1
b0 = 0.1
nbatch = 100
lr = 5e-1
l2 = 1e1

n_depth = 10

relu = activations.Rectify()
tanh = activations.Tanh()
gifn = inits.Normal(scale=.01)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

npx = x_dim + 1

w_h1 = gifn((npx, npx), 'w_h1')
gg1 = gain_ifn((npx,), 'gg1')
gb1 = bias_ifn((npx,), 'gb1')

w_h2 = gifn((npx, npx), 'w_h2')
gg2 = gain_ifn((npx,), 'gg2')
gb2 = bias_ifn((npx,), 'gb2')

w_h3 = gifn((npx, npx), 'w_h3')
gg3 = gain_ifn((npx,), 'gg3')
gb3 = bias_ifn((npx,), 'gb3')


block_1 = bias_ifn((n_depth, npx), 'block_1')
net_params = [w_h1, gg1, gb1, w_h2, gg2, gb2, block_1]


def rbf_kernel(X):

    XY = T.dot(X, X.T)
    x2 = T.sum(X**2, axis=1).dimshuffle(0, 'x')
    X2e = T.repeat(x2, X.shape[0], axis=1)
    H = X2e +  X2e.T - 2. * XY

    V = H.flatten()
    # median distance
    h = T.switch(T.eq((V.shape[0] % 2), 0),
        # if even vector
        T.mean(T.sort(V)[ ((V.shape[0] // 2) - 1) : ((V.shape[0] // 2) + 1) ]),
        # if odd vector
        T.sort(V)[V.shape[0] // 2])

    h = .5 * h / T.log(T.cast(H.shape[0] + 1., theano.config.floatX))

    # compute the rbf kernel
    kxy = T.exp(-H / h / 2.0)

    dxkxy = -T.dot(kxy, X)
    sumkxy = T.sum(kxy, axis=1).dimshuffle(0, 'x')
    dxkxy = T.add(dxkxy, T.mul(X, sumkxy)) / h

    return kxy, dxkxy


def evaluate(xmb, ymb, theta):

    theta = theta[:, :-1]
    M, n_test = theta.shape[0], ymb.shape[0]

    coff = ymb.dimshuffle('x', 0) * T.sum(-1 * T.tile(theta.dimshuffle(0, 'x', 1), (1, n_test, 1)) * \
                        xmb.dimshuffle('x', 0, 1), axis=2)

    prob = 1. / (1. + T.exp(coff))

    prob = T.mean(prob, axis=0)
    acc = T.mean(prob > 0.5)
    llh = T.mean(T.log(prob))

    return acc, llh


def score_bayes_lr(xmb, ymb, theta, data_N):
    w = theta[:, :-1]
    alpha = T.exp(theta[:, -1])
    d = theta.shape[1] - 1
    M_theta = theta.shape[0]

    wt = T.mul((alpha / 2), T.sum(w ** 2, axis=1))

    coff = T.dot(xmb, w.T)
    y_hat = 1.0 / (1.0 + T.exp(-1 * coff))

    dw_data = T.dot(((T.tile(ymb.dimshuffle(0, 'x'), (1, M_theta)) + 1) / 2.0 - y_hat).T, xmb)

    dw_prior = - T.mul(T.tile(alpha.dimshuffle(0, 'x'), (1, d)), w)

    dw = dw_data * 1.0 * T.cast(data_N, theano.config.floatX) / T.cast(xmb.shape[0], theano.config.floatX) + dw_prior
    dalpha = T.cast(d, theano.config.floatX) / 2.0 - wt + (a0 - 1) - b0 * alpha + 1

    return T.concatenate([dw, dalpha.dimshuffle(0, 'x')], axis=1)


def svgd_gradient(xmb, ymb, theta, data_N):
    grad = score_bayes_lr(xmb, ymb, theta, data_N)
    kxy, dxkxy = rbf_kernel(theta)

    svgd_grad = (T.dot(kxy, grad) + dxkxy) / T.sum(kxy, axis=1).dimshuffle(0, 'x')

    return grad, svgd_grad


def langevin_network(xmb, ymb, Z, data_N, w_h1, gg1, gb1, w_h2, gg2, gb2, block):

    h1 = relu(batchnorm(T.dot(Z, w_h1), g=gg1, b=gb1))
    Z = (batchnorm(T.dot(h1, w_h2), g=gg2, b=gb2))

    prior_samples = Z

    for i in range(n_depth):
        score_x = score_bayes_lr(xmb[i*nbatch:(i+1)*nbatch], ymb[i*nbatch:(i+1)*nbatch], prior_samples, data_N)
        prior_samples = prior_samples + block[i]**2 * score_x / 2. + block[i]**2 * srng.normal(Z.shape)

    return prior_samples


def init_theta(a =1, b0=0.1, n_particle=20):
    alpha0 = np.transpose([np.random.gamma(a0, b0, n_particle)])

    theta0 = np.concatenate([
        np.random.normal(loc=np.zeros((n_particle, 1)), scale=np.sqrt(1 / alpha0),
                size=(n_particle, x_dim)), alpha0], axis=1)

    return theta0.astype(theano.config.floatX)

def _make_svgd_step(theta, lr=1e-2):

    # adagrad with momentum
    fudge_factor = 1e-6
    alpha = 0.9

    updates = []

    grad, grad_theta = svgd_gradient(X, y, theta, data_N)
    acc = theano.shared(theta.get_value() * 0.)
    acc_new = alpha * acc + (1 - alpha) * grad_theta ** 2
    updates.append((acc, T.cast(acc_new, theano.config.floatX)))

    theta_new = theta + (lr / T.sqrt(acc_new  + fudge_factor)) * grad_theta
    updates.append((theta, T.cast(theta_new, theano.config.floatX)))

    svgd_func = theano.function([X, y, data_N], [], updates=updates)

    return svgd_func


def _make_langevin_step(theta, lr = 1e-4):

    i = T.iscalar()
    stepsize = lr * (1 + i) ** (-0.55)
    grad = score_bayes_lr(X, y, theta, data_N)
    update = stepsize * grad / 2. + T.sqrt(stepsize) * srng.normal(size=grad.shape)
    lgvn_func = theano.function([X, y, i, data_N], [], updates=[(theta, T.cast(theta+update, theano.config.floatX))])
    return lgvn_func


def _chunck_eval(X_test, y_test, theta0):
    chunk_size = X_test.shape[0] // 3
    rmse_1, ll_1 = _evaluate(X_test[:chunk_size], y_test[:chunk_size], floatX(theta0))
    # print 'rmse 1', rmse_1, ll_1

    rmse_2, ll_2 = _evaluate(X_test[chunk_size:2*chunk_size], y_test[chunk_size:2*chunk_size], floatX(theta0))
    # print 'rmse 2', rmse_2, ll_2

    rmse_3, ll_3 = _evaluate(X_test[2*chunk_size:], y_test[2*chunk_size:], floatX(theta0))
    # print 'rmse 3', rmse_3, ll_3

    return (rmse_1 + rmse_2 + rmse_3)/ 3., (ll_1 + ll_2 + ll_3)/ 3.



X = T.fmatrix()
y = T.fvector()

theta = T.fmatrix()
deltaX = T.fmatrix() # svgd gradient
data_N = T.scalar('data_N')

block = T.fmatrix()


gX_1 = langevin_network(X, y, theta, data_N, *net_params)
cost_1 = -1 * T.mean(T.sum(gX_1 * deltaX, axis=1))


lrt = sharedX(lr)
g_updater_1 = updates.Adagrad(lr=lr, regularizer=updates.Regularizer(l2=l2))
g_updates_1 = g_updater_1(net_params, cost_1)


print 'COMPILING'
t = time()
_gen_1 = theano.function([X, y, theta, data_N], gX_1)
_train_g_1 = theano.function([X, y, theta, deltaX, data_N], cost_1, updates=g_updates_1)
_svgd_gradient = theano.function([X, y, theta, data_N], svgd_gradient(X, y, theta, data_N))
_score_bayes_lr = theano.function([X, y, theta, data_N], score_bayes_lr(X, y, theta, data_N))
_evaluate = theano.function([X, y, theta], evaluate(X, y, theta))
print '%.2f seconds to compile theano functions'%(time()-t)


n_iter = 10000
n_particle = 100

# first training the network
print "Start Training Langevin Sampler"
# langevin sampler
for iter in tqdm(range(1, n_iter+1)):
    ntrain = X_train.shape[0]

    imb = [t % ntrain for t in range(iter*n_depth*nbatch, (iter+1)*n_depth*nbatch)]
    xmb, ymb = floatX(X_train[imb]), floatX(y_train[imb])

    theta0 = init_theta(n_particle = n_particle)

    b1_samples = floatX(_gen_1(xmb, ymb, theta0, ntrain))
    grad, svgd_grad = _svgd_gradient(xmb, ymb, b1_samples, ntrain)
    _train_g_1(xmb, ymb, theta0, svgd_grad, ntrain)



total_svgd_acc = []
total_svgd_ll = []

total_langevin_acc = []
total_langevin_ll = []

total_m_acc = []
total_m_ll = []

print "Start testing on separete data set"
for data_i in range(0, 8):
    # For each dataset training on dev and testing on test dataset

    X_dev, y_dev = total_dev[data_i]
    X_test, y_test = total_test[data_i]
    dev_N = X_dev.shape[0]
    X_dev, y_dev = shuffle(X_dev, y_dev)

    X_dev, y_dev = shuffle(X_dev, y_dev)
    X_test, y_test = shuffle(X_test, y_test)
    dev_N = X_dev.shape[0]


    ### svgd
    x0 = init_theta(n_particle=n_particle)
    x0 = sharedX(x0)
    _svgd_step = _make_svgd_step(x0, lr=1e-6 * (2 ** 13))
    for i in tqdm(range(n_iter)):
        imb = [t % dev_N for t in range(i*nbatch, (i+1)*nbatch)]
        _svgd_step(X_dev[imb], y_dev[imb], dev_N)

    svgd_rmse, svgd_ll =_chunck_eval(X_test, y_test, x0.get_value())
    total_svgd_acc.append(svgd_rmse)
    total_svgd_ll.append(svgd_ll)

    # langevin
    x0 = init_theta(n_particle = n_particle)
    x0 = sharedX(x0)
    _lgvn_step = _make_langevin_step(x0, lr=1e-6 * (2 ** 11))
    for i in tqdm(range(n_iter)):
        imb = np.asarray([t % dev_N for t in range(i*nbatch, (i+1)*nbatch)]).astype('int32')
        _lgvn_step(X_dev[imb], y_dev[imb], i, dev_N)


    lv_rmse, lv_ll = _chunck_eval(X_test, y_test, x0.get_value())
    total_langevin_acc.append(lv_rmse)
    total_langevin_ll.append(lv_ll)


    # langevin sampler
    imb_dev = np.random.choice(dev_N, min(n_depth * nbatch, dev_N), replace=False)
    xmb_dev, ymb_dev = floatX(X_dev[imb_dev]), floatX(y_dev[imb_dev])
    xmb_dev, ymb_dev = shuffle(xmb_dev, ymb_dev)
    valid_theta = init_theta(n_particle = n_particle)
    samples = floatX(_gen_1(xmb_dev, ymb_dev, valid_theta, dev_N))
    rmse, ll = _chunck_eval(X_test, y_test, samples)
    total_m_acc.append(rmse)
    total_m_ll.append(ll)


    print "Evaluation of dataset=%d" %(data_i)
    print "Evaluation of svgd, ", svgd_rmse, svgd_ll
    print "Evaluation of langevin", lv_rmse, lv_ll
    print "Evaluation of our methods, ", rmse, ll
    print "\n"


print "Final results"
print "\nSGD-----"
print "SVGD acc", np.mean(total_svgd_acc), np.std(total_svgd_acc)
print "SVGD llh", np.mean(total_svgd_ll), np.std(total_svgd_ll)


print "\nConverged Langevin---"
print "langevin acc", np.mean(total_langevin_acc), np.std(total_langevin_acc)
print "langevin llh", np.mean(total_langevin_ll), np.std(total_langevin_ll)


print "\nOur methods----"
print "our acc", np.mean(total_m_acc), np.std(total_m_acc)
print "our ll", np.mean(total_m_ll), np.std(total_m_ll)

