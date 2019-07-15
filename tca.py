# encoding=utf-8

import numpy as np
import pickle
import sklearn.metrics
import scipy.linalg
import os

from tsne import *

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
       
    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1/ns*np.ones((ns, 1)), -1/nt*np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n,n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

def main():
    # kernel_type='primal'
    kernel_type = 'linear'
    # tca_path = 'E:/python/FeatureProcessing-master/linear/'
    tca_path = 'E:/python/FeatureProcessing-master/normalized/'+kernel_type+'/'
    if not os.path.exists(tca_path):
    	os.mkdir(tca_path)
    casia_tca = tca_path+'casia_tca.pkl'
    iemo_tca = tca_path+'iemo_tca.pkl' 
    casia_X, casia_y = read_file(casia_file)
    iemo_X, iemo_y = read_file(iemo_file)    
    casia_X = normalize(casia_X)
    iemo_X = normalize(iemo_X)

    tca = TCA(kernel_type=kernel_type, dim=len(casia_X[0]), lamb=1, gamma=1)
    iemo_X, casia_X = tca.fit(iemo_X, casia_X)
    # iemo_X, casia_X = tca.fit(iemo_X[:20], casia_X[:10])
    print('tca end')
    feature_emo = []
    for (x, y) in zip(casia_X, casia_y):
        feature_emo.append((np.reshape(x,(300, 23)), y))
    pickle.dump(feature_emo, open(casia_tca, 'wb'))

    feature_emo = []
    for (x, y) in zip(iemo_X, iemo_y):
        feature_emo.append((np.reshape(x,(300, 23)), y))
    pickle.dump(feature_emo, open(iemo_tca, 'wb'))

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    try:
        casia_X = tsne.fit_transform(casia_X)
        iemo_X = tsne.fit_transform(iemo_X)
    except ValueError:
        casia_X = np.real(casia_X)
        iemo_X = np.real(iemo_X)
        casia_X = tsne.fit_transform(casia_X)
        iemo_X = tsne.fit_transform(iemo_X)
        print('complex occured.')

    m = np.max(casia_X) if np.max(casia_X)>np.max(iemo_X) else np.max(iemo_X)

    fig = plt.figure()
    ax = plt.subplot(111)
    m = np.max(casia_X) if np.max(casia_X) > np.max(iemo_X) else np.max(iemo_X)
    plt.axis([-m, m, -m, m])
    plt_text(casia_X, casia_y, 'c', plt)
    plt_text(iemo_X, iemo_y, 'i', plt)
    plt.xticks([])
    plt.yticks([])
    plt.show(fig)

if __name__ == '__main__':
    main()
        



