import scipy as sp
from numpy.linalg import inv
from numpy.linalg import solve
from scipy.io import loadmat
import numpy as np
from scipy.spatial.distance import cdist


def load_data(fname):
    """
    Loads two dimensional data from <fname>
    """
    # load the data
    data = loadmat(fname)
    # extract data for training
    X_train = data['training_data']
    X_train = sp.log(X_train)
    X_train = X_train[:, :1000]
    # extract positions
    Y_train = data['training_labels']
    Y_train = Y_train[:, :1000]
    return X_train,Y_train

X_train, Y_train = load_data('file.mat')


def gaussian_kernel(x1, x2, kwidth):
    """
    Compute Gaussian Kernel
    Input: X1    - DxN1 array of N1 data points with D features
           X2    - DxN2 array of N2 data points with D features
           kwidth - Kernel width
    Output K     - N1 x N2 Kernel matrix
    """
    assert(x1.shape[0] == x2.shape[0])
    K = cdist(x1.T, x2.T, 'euclidean')
    K = np.exp(-(K ** 2) / (2. * kwidth ** 2))
    return K

K = gaussian_kernel(X_train, X_train, 1)


def train_kernel_ridge(x_train, y_train, kwidth, llambda):
    """
    Trains kernel ridge regression (krr)
    Input:       X_train  -  DxN array of N data points with D features
                 Y        -  D2xN array of length N with D2 multiple labels
                 kwdith   -  kernel width
                 llambda    -  regularization parameter
    Output:      alphas   -  NxD2 array, weighting of training data
    used for apply_krr
    """
    K = gaussian_kernel(X_train, X_train, kwidth)
    alphas = sp.dot(inv(K + llambda * np.eye(np.shape(K)[0])),
                    Y_train.transpose())
    return alphas
