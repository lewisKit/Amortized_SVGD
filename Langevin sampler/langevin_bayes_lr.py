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

    '''
    if i == 1:
        total_dev_X, total_dev_y = floatX(X_dev.toarray()), floatX(y_dev)
        total_test_X, total_test_y = floatX(X_test.toarray()), floatX(y_test)
    else:

        total_dev_X = np.concatenate([total_dev_X, floatX(X_dev.toarray())], axis=0)
        total_dev_y = np.concatenate([total_dev_y, floatX(y_dev)], axis=0)

        total_test_X = np.concatenate([total_test_X, floatX(X_test.toarray())], axis=0)
        total_test_y = np.concatenate([total_test_y, floatX(y_test)], axis=0)
    '''

N_train = X_train.shape[0]
print "size of training=%d" % (N_train)

x_dim = X_train.shape[1]



desc = 'amortized_bayes_lr'
model_dir = 'models/%s' % desc
samples_dir = 'samples/%s' % desc


dir_list = [model_dir, samples_dir]
for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)
print desc


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
#block_2 = bias_ifn((n_depth, npx), 'block_2')

#net_params = [w_h1, gg1, gb1, w_h2, gg2, gb2, w_h3, gg3, gb3, block_1]
net_params = [w_h1, gg1, gb1, w_h2, gg2, gb2, block_1]
#net_params = [block_1]


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


#def langevin_network(xmb, ymb, Z, w_h1, gg1, gb1, w_h2, gg2, gb2, w_h3, gg3, gb3, block):
def langevin_network(xmb, ymb, Z, data_N, w_h1, gg1, gb1, w_h2, gg2, gb2, block):
#def langevin_network(xmb, ymb, Z, block):

    h1 = relu(batchnorm(T.dot(Z, w_h1), g=gg1, b=gb1))
    #h2 = relu(batchnorm(T.dot(h1, w_h2), g=gg2, b=gb2))
    Z = (batchnorm(T.dot(h1, w_h2), g=gg2, b=gb2))

    prior_samples = Z

    for i in range(n_depth):
        score_x = score_bayes_lr(xmb[i*nbatch:(i+1)*nbatch], ymb[i*nbatch:(i+1)*nbatch], prior_samples, data_N)
        prior_samples = prior_samples + block[i]**2 * score_x / 2. + block[i]**2 * srng.normal(Z.shape)
        # grad, svgd_grad = svgd_gradient(xmb[i*nbatch:(i+1)*nbatch], ymb[i*nbatch:(i+1)*nbatch], prior_samples, data_N)
        # prior_samples = prior_samples + (block[i] ** 2) * svgd_grad


    return prior_samples


X = T.fmatrix()
y = T.fvector()

theta = T.fmatrix()
deltaX = T.fmatrix() # svgd gradient
data_N = T.scalar('data_N')

block = T.fmatrix()


gX_1 = langevin_network(X, y, theta, data_N, *net_params)
#gX_2 = langevin_network(X, y, theta, block_2)
cost_1 = -1 * T.mean(T.sum(gX_1 * deltaX, axis=1))
#cost_2 = -1 * T.mean(T.sum(gX_2 * deltaX, axis=1))


lrt = sharedX(lr)
g_updater_1 = updates.Adagrad(lr=lr, regularizer=updates.Regularizer(l2=l2))
g_updates_1 = g_updater_1(net_params, cost_1)

#g_updater_2 = updates.Adagrad(lr=lr, regularizer=updates.Regularizer(l2=l2))
#g_updates_2 = g_updater_2([block_2], cost_2)


print 'COMPILING'
t = time()
_gen_1 = theano.function([X, y, theta, data_N], gX_1)
#_gen_2 = theano.function([X, y, theta], gX_2)
_train_g_1 = theano.function([X, y, theta, deltaX, data_N], cost_1, updates=g_updates_1)
#_train_g_2 = theano.function([X, y, theta, deltaX], cost_2, updates=g_updates_2)
_svgd_gradient = theano.function([X, y, theta, data_N], svgd_gradient(X, y, theta, data_N))
_score_bayes_lr = theano.function([X, y, theta, data_N], score_bayes_lr(X, y, theta, data_N))
# _logp_kernel = theano.function([X, y, theta], logp_kernel(X, y, theta))
_evaluate = theano.function([X, y, theta], evaluate(X, y, theta))
print '%.2f seconds to compile theano functions'%(time()-t)


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




n_iter = 10000
n_particle = 100
# stochastic langevin

'''
for k in range(0, 15):
    lr = 1e-6 * (2 ** k)

    # langevin
    x0 = init_theta(n_particle = n_particle)
    x0 = sharedX(x0)
    _lgvn_step = _make_langevin_step(x0, lr)
    for i in tqdm(range(n_iter)):
        imb = np.asarray([t % ntrain for t in range(i*nbatch, (i+1)*nbatch)]).astype('int32')
        _lgvn_step(X_train[imb], y_train[imb], i)

    print "k=%d, learning_rate=%f" %(k, lr)
    _chunck_eval(x0.get_value())
'''



# first training the network

# langevin network
for iter in tqdm(range(1, n_iter+1)):
    ntrain = X_train.shape[0]

    imb = [t % ntrain for t in range(iter*n_depth*nbatch, (iter+1)*n_depth*nbatch)]
    xmb, ymb = floatX(X_train[imb]), floatX(y_train[imb])

    theta0 = init_theta(n_particle = n_particle)
    #theta0 = floatX(np_rng.normal(0,.1,size=(n_particle,npx)))
    #theta0 = floatX(x0.get_value())

    b1_samples = floatX(_gen_1(xmb, ymb, theta0, ntrain))
    grad, svgd_grad = _svgd_gradient(xmb, ymb, b1_samples, ntrain)
    _train_g_1(xmb, ymb, theta0, svgd_grad, ntrain)
    #_train_g_1(xmb, ymb, theta0, floatX(grad))

    # if iter % 100 == 0:

    #     imb_dev = np.asarray([t % ndev for t in range(iter*n_depth*nbatch, (iter + 1)*n_depth*nbatch)]).astype('int32')
    #     xmb_dev, ymb_dev = floatX(X_dev[imb_dev]), floatX(y_dev[imb_dev])
    #     xmb_dev, ymb_dev = shuffle(xmb_dev, ymb_dev)
    #     valid_theta = init_theta(n_particle = n_particle)
    #     samples = floatX(_gen_1(xmb_dev, ymb_dev, valid_theta))
    #     rmse, ll = _evaluate(X_test, y_test, samples)
    #     print rmse, ll


'''
print "tunning stepsize"
# svgd:
print "tunning svgd"
for k in range(0, 20):
    lr = 1e-6 * (2 ** k)
    print "k=%d, learning_rate=%f" %(k, lr)
    for data_i in range(0, 8):
        # For each dataset training on dev and testing on test dataset
        X_dev, y_dev = total_dev[data_i]
        X_test, y_test = total_test[data_i]
        dev_N = X_dev.shape[0]
        X_dev, y_dev = shuffle(X_dev, y_dev)

        ### svgd
        x0 = init_theta(n_particle=n_particle)
        x0 = sharedX(x0)
        _svgd_step = _make_svgd_step(x0, lr=lr)
        for i in tqdm(range(n_iter)):
            imb = [t % dev_N for t in range(i*nbatch, (i+1)*nbatch)]
            _svgd_step(X_dev[imb], y_dev[imb], dev_N)

        svgd_rmse, svgd_ll =_evaluate(X_test, y_test, x0.get_value())
        print "dataset, ", data_i, svgd_rmse, svgd_ll
    print "\n\n"

print "tunning langevin--------"

for k in range(0, 20):
    lr = 1e-6 * (2 ** k)
    print "k=%d, learning_rate=%f" % (k, lr)

    for data_i in range(0, 8):
        # For each dataset training on dev and testing on test dataset
        X_dev, y_dev = total_dev[data_i]
        X_test, y_test = total_test[data_i]
        dev_N = X_dev.shape[0]
        X_dev, y_dev = shuffle(X_dev, y_dev)

        ### langevin
        x0 = init_theta(n_particle=n_particle)
        x0 = sharedX(x0)
        _lgvn_step = _make_langevin_step(x0, lr=lr)
        for i in tqdm(range(n_iter)):
            imb = np.asarray([t % dev_N for t in range(i*nbatch, (i+1)*nbatch)]).astype('int32')
            _lgvn_step(X_dev[imb], y_dev[imb], i, dev_N)

        lv_rmse, lv_ll = _evaluate(X_test, y_test, x0.get_value())
        print "dataset", data_i, lv_rmse, lv_ll
    print "\n\n"

print "finishing searching"
'''

total_svgd_acc = []
total_svgd_ll = []

total_langevin_acc = []
total_langevin_ll = []

total_langevin_10_acc = []
total_langevin_100_acc = []
total_langevin_1000_acc = []

total_langevin_10_ll = []
total_langevin_100_ll = []
total_langevin_1000_ll = []


total_m_acc = []
total_m_ll = []

print "value of blocks", block_1.get_value()

print "training separete data set"
for data_i in range(0, 8):
    # For each dataset training on dev and testing on test dataset

    X_dev, y_dev = total_dev[data_i]
    X_test, y_test = total_test[data_i]
    dev_N = X_dev.shape[0]
    X_dev, y_dev = shuffle(X_dev, y_dev)
    '''
    X_dev, y_dev = total_dev_X, total_dev_y
    X_test, y_test = total_test_X, total_test_y
    '''
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
        '''
        if i == 10:
            lv_rmse, lv_ll = _chunck_eval(X_test, y_test, x0.get_value())
            total_langevin_10_acc.append(lv_rmse)
            total_langevin_10_ll.append(lv_ll)
        if i == 100:
            lv_rmse, lv_ll = _chunck_eval(X_test, y_test, x0.get_value())
            total_langevin_100_acc.append(lv_rmse)
            total_langevin_100_ll.append(lv_ll)

        if i == 1000:
            lv_rmse, lv_ll = _chunck_eval(X_test, y_test, x0.get_value())
            total_langevin_1000_acc.append(lv_rmse)
            total_langevin_1000_ll.append(lv_ll)
        '''

    lv_rmse, lv_ll = _chunck_eval(X_test, y_test, x0.get_value())
    total_langevin_acc.append(lv_rmse)
    total_langevin_ll.append(lv_ll)


    # langevin network
    imb_dev = np.random.choice(dev_N, min(n_depth * nbatch, dev_N), replace=False)
    xmb_dev, ymb_dev = floatX(X_dev[imb_dev]), floatX(y_dev[imb_dev])
    xmb_dev, ymb_dev = shuffle(xmb_dev, ymb_dev)
    valid_theta = init_theta(n_particle = n_particle)
    samples = floatX(_gen_1(xmb_dev, ymb_dev, valid_theta, dev_N))
    rmse, ll = _chunck_eval(X_test, y_test, samples)
    total_m_acc.append(rmse)
    total_m_ll.append(ll)


    print "Evaluation of experiment=%d" %(data_i)
    print "Evaluation of svgd, ", svgd_rmse, svgd_ll
    print "Evaluation of langevin", lv_rmse, lv_ll
    print "Evaluation of our methods, ", rmse, ll
    print "\n\n"





print "final results"

print "SGD-----"
print "SVGD acc", np.mean(total_svgd_acc), np.std(total_svgd_acc)
print "SVGD llh", np.mean(total_svgd_ll), np.std(total_svgd_ll)

print "langevin 10---"
print "langevin 10 acc", np.mean(total_langevin_10_acc), np.std(total_langevin_10_acc)
print "langevin 10 llh", np.mean(total_langevin_10_ll), np.std(total_langevin_10_ll)


print "langevin 100---"
print "langevin 100 acc", np.mean(total_langevin_100_acc), np.std(total_langevin_100_acc)
print "langevin 100 llh", np.mean(total_langevin_100_ll), np.std(total_langevin_100_ll)


print "langevin 1000---"
print "langevin 1000 acc", np.mean(total_langevin_1000_acc), np.std(total_langevin_1000_acc)
print "langevin 1000 llh", np.mean(total_langevin_1000_ll), np.std(total_langevin_1000_ll)

print "Converged Langevin---"
print "langevin acc", np.mean(total_langevin_acc), np.std(total_langevin_acc)
print "langevin llh", np.mean(total_langevin_ll), np.std(total_langevin_ll)


print "Our methods----"
print "our acc", np.mean(total_m_acc), np.std(total_m_acc)
print "our ll", np.mean(total_m_ll), np.std(total_m_ll)




'''
    for t in tqdm(range(10)):
        imb = np.asarray([t % ndev for t in range(i*nbatch, (i+1)*nbatch)]).astype('int32')
        _lgvn_step(X_dev[imb], y_dev[imb], t)

    _chunck_eval(x0.get_value())

    # converge
    for i in tqdm(range(1000)):
        imb = np.asarray([t % ndev for t in range(i*nbatch, (i+1)*nbatch)]).astype('int32')
        _lgvn_step(X_dev[imb], y_dev[imb], i)

    _chunck_eval(x0.get_value())
    '''


    #b2_samples = floatX(_gen_2(xmb, ymb, b1_samples))
    #grad, svgd_grad = _svgd_gradient(xmb, ymb, b2_samples)
    #_train_g_2(xmb, ymb, b1_samples, floatX(svgd_grad))

    #stepsize = 1e-4 * (1 + iter) ** (-0.55)
    #pred, grad = _score_nn(xmb, ymb, theta0)
    #update  = stepsize * grad / 2. + np.sqrt(stepsize) * np.random.normal(0, 1, grad.shape)
    #theta0 = theta0 + floatX(update)


svgd_x0 = init_theta(n_particle=n_particle)
svgd_x0 = sharedX(svgd_x0)
_svgd_step = _make_svgd_step(svgd_x0)

langevin_x0 = sharedX(init_theta(n_particle=n_particle))
_lgvn_step = _make_langevin_step(langevin_x0)




svgd_x0 = init_theta(n_particle=n_particle)
svgd_x0 = sharedX(svgd_x0)
_svgd_step = _make_svgd_step(svgd_x0)

langevin_x0 = sharedX(init_theta(n_particle=n_particle))
_lgvn_step = _make_langevin_step(langevin_x0)


days = 10

'''
for day in range(0, days):
    # first generate data
    cur_indics = [t % total_X_train.shape[0] for t in range(day * data_X_N, (day + 1) * data_X_N)]


    X_train ,y_train = total_X_train[cur_indics], total_y_train[cur_indics]

    # X_train, X_test, y_train, y_test = train_test_split(X_input[cur_indics], y_input[cur_indics], test_size=0.5)
    X_train, y_train = shuffle(X_train, y_train)
    # normalization
    X_train = (X_train - mean_X_train) / std_X_train
    # X_test = (X_test - mean_X_train) / std_X_train
    y_train = (y_train - mean_y_train) / std_y_train
    ntrain = X_train.shape[0]


    svgd_x0_noi = sharedX(init_theta(n_particle=n_particle))
    _svgd_step_noi = _make_svgd_step(svgd_x0_noi)

    langevin_x0_noi = sharedX(init_theta(n_particle=n_particle))
    _lgvn_step_noi = _make_langevin_step(langevin_x0_noi)



    # langevin network
    for iter in tqdm(range(1, n_iter+1)):

        imb = np.asarray([t % ntrain for t in range(iter*n_depth*nbatch, (iter+1)*n_depth*nbatch)]).astype('int32')
        xmb, ymb = X_train[imb], y_train[imb]

        theta0 = init_theta(n_particle = n_particle)
        #theta0 = floatX(np_rng.normal(0,.1,size=(n_particle,npx)))
        #theta0 = floatX(x0.get_value())

        b1_samples = floatX(_gen_1(xmb, ymb, theta0))
        grad, svgd_grad = _svgd_gradient(xmb, ymb, b1_samples)
        _train_g_1(xmb, ymb, theta0, floatX(svgd_grad))
        #_train_g_1(xmb, ymb, theta0, floatX(grad))

        # svgd training
        _svgd_step(xmb, ymb)

        # langevin training
        _lgvn_step(xmb, ymb, day * (n_iter) + iter)

        # no initial training
        _svgd_step_noi(xmb, ymb)

        # no initial langevin training
        _lgvn_step_noi(xmb, ymb, iter)



    rmse, ll = _evaluate(X_test, y_test, b1_samples)
    print 'Langevin Networks ', rmse, ll
    nn_rmse.append(float(rmse))
    nn_ll.append(float(ll))

    rmse, ll = _evaluate(X_test, y_test, svgd_x0.get_value())
    print "SVGD, ", rmse, ll
    svgd_rmse.append(float(rmse))
    svgd_ll.append(float(ll))

    rmse, ll = _evaluate(X_test, y_test, langevin_x0.get_value())
    print "langevin, ", rmse, ll
    langevin_rmse.append(float(rmse))
    langevin_ll.append(float(ll))

    # no initial version

    rmse, ll = _evaluate(X_test, y_test, svgd_x0_noi.get_value())
    print "SVGD not initial, ", rmse, ll
    svgd_no_rmse.append(float(rmse))
    svgd_no_ll.append(float(ll))

    rmse, ll = _evaluate(X_test, y_test, langevin_x0_noi.get_value())
    print "langevin not initial, ", rmse, ll
    langevin_no_rmse.append(float(rmse))
    langevin_no_ll.append(float(ll))



        #b2_samples = floatX(_gen_2(xmb, ymb, b1_samples))
        #grad, svgd_grad = _svgd_gradient(xmb, ymb, b2_samples)
        #_train_g_2(xmb, ymb, b1_samples, floatX(svgd_grad))

        #stepsize = 1e-4 * (1 + iter) ** (-0.55)
        #pred, grad = _score_nn(xmb, ymb, theta0)
        #update  = stepsize * grad / 2. + np.sqrt(stepsize) * np.random.normal(0, 1, grad.shape)
        #theta0 = theta0 + floatX(update)

print "nn evaluation"
print nn_rmse
print nn_ll
print '\n'

print "svgd evaluation"
print svgd_rmse
print svgd_ll
print '\n'

print "langevin evaluate"
print langevin_rmse
print langevin_ll
print '\n'

print "svgd evaluation without previous initalization"
print svgd_no_rmse
print svgd_no_ll
print '\n'

print "langevin evaluation without previous initliazation"
print langevin_no_rmse
print langevin_no_ll
print '\n'
'''



#rmse, ll = _evaluate(X_test, y_test, b2_samples)
#print rmse, ll



