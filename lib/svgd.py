import theano
import theano.tensor as T
import numpy as np

epsilon = 1e-6

def vgd_kernel(X0):
    XY = T.dot(X0, X0.transpose())
    x2 = T.reshape(T.sum(T.square(X0), axis=1), (X0.shape[0], 1))
    X2e = T.repeat(x2, X0.shape[0], axis=1)
    H = T.sub(T.add(X2e, X2e.transpose()), 2 * XY)

    V = H.flatten()

    # median distance
    h = T.switch(T.eq((V.shape[0] % 2), 0),
        # if even vector
        T.mean(T.sort(V)[ ((V.shape[0] // 2) - 1) : ((V.shape[0] // 2) + 1) ]),
        # if odd vector
        T.sort(V)[V.shape[0] // 2])
    h = T.sqrt(0.5 * h / T.log(X0.shape[0].astype('float32') + 1.0))

    # compute the rbf kernel
    Kxy = T.exp(-H / h ** 2 / 2.0)

    dxkxy = -T.dot(Kxy, X0)
    sumkxy = T.sum(Kxy, axis=1).dimshuffle(0, 'x')
    dxkxy = T.add(dxkxy, T.mul(X0, sumkxy)) / (h ** 2)

    return (Kxy, dxkxy)


'''
    This is svgd version for 3d tensor X0
    The first dimension is number of images,
    second is number of z per images,
    third dimension is the dimension of z
'''

def vgd_kernel_tensor(X0):

    XY = T.batched_dot(X0, X0.transpose(0,2,1))
    x2 = T.reshape(T.sum(T.square(X0),axis=2), (X0.shape[0], X0.shape[1], 1))
    X2e = T.repeat(x2, X0.shape[1], axis=2)
    H = T.sub(T.add(X2e, X2e.transpose(0,2,1)), 2 * XY)

    V = H.flatten(2)

    # median distance
    h = T.switch(T.eq((V.shape[1] % 2), 0),
            # if even vector
            T.mean(T.sort(V)[:, ((V.shape[1] // 2) - 1): ((V.shape[1] // 2) + 1)], axis=1),
             # if odd vector
            T.sort(V)[:, V.shape[1] // 2])

    h = T.sqrt(0.5 * h / T.log(X0.shape[1].astype('float32') + 1.0))
    # h = T.maximum(h, T.zeros_like(h) + 1e-4)

    # h = h / 2
    Kxy = T.exp(-H / T.tile(h.dimshuffle(0, 'x', 'x'), (1, X0.shape[1], X0.shape[1])) ** 2 / 2.0)

    dxkxy = - T.batched_dot(Kxy, X0)
    sumkxy = T.sum(Kxy, axis=2).dimshuffle(0, 1, 'x')
    dxkxy = T.add(dxkxy, T.mul(X0, sumkxy)) / (T.tile(h.dimshuffle(0, 'x', 'x'), (1, X0.shape[1], X0.shape[2])) ** 2)

    return (Kxy, dxkxy, h)








