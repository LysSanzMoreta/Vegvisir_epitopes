"""
SVD: https://towardsdatascience.com/svd-8c2f72e264f
PCA: https://towardsdatascience.com/implementing-pca-from-scratch-fb434f1acbaa

a) First of all, we need to compute the covariance matrix.
b) Once we obtain this matrix, we need to decompose it, using eigendecomposition.
c) Next, we can select the most important eigenvectors based on the eigenvalues, to finally project the original matrix into its reduced dimension.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from pylab import plt
import time

class R_pca:
    """Deep PCA from https://github.com/blengerich/drpca"""
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')

def get_differential(data, numComponents=None):
    m, n = data.shape
    data -= data.mean(axis=0)
    pca = PCA(n_components=numComponents)
    data_components = pca.fit_transform(data)
    return data_components, pca.singular_values_, pca.components_.T

class PCA_custom:
    def __init__(self,n_components):
        self.n_components = n_components
    def standardize_data(self, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X-mean)/std

    def get_covariance_matrix(self,X,ddof=0):
        """
        Function equivalent to covar_np = np.cov(pca.standardize_data(X),rowvar=False)
        :param int ddof : degrees of freedom, if == 0, all variables are independent
        """

        n = X.shape[0]
        #out = np.dot(X.T,X)
        out = X.T @ X
        return out/(n-ddof)
    def get_eigenvectors(self, C):
        eigenvals, eigenvectors = np.linalg.eig(C)
        n_cols = np.argsort(eigenvals)[::-1][:self.n_components]
        max_eigenvectors = eigenvectors[:, n_cols]
        return max_eigenvectors

    def project_matrix(self, eigenvectors):
        pass
    def fit(self,X):
        standarized_data = self.standardize_data(X)
        covariance = self.get_covariance_matrix(X)
        max_eigenvectors = self.get_eigenvectors(covariance)
        data_proj = np.dot(standarized_data, max_eigenvectors)
        return data_proj


def plot_projection(projection):

    fig, ax = plt.subplots(1, 1, figsize=(10,6))

    sns.scatterplot(
        x = projection[:,0],
        y = projection[:,1],
        hue=y_train
    )
    ax.set_title('Iris Dataset')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    sns.despine()
    plt.show()


# load iris dataset
iris = datasets.load_iris()
X_train = iris.data
y_train = iris.target

#Highlight: Singular Value Decomposition
# U,S,VT = np.linalg.svd(X_train)
# projection_svd = X_train@VT
# plot_projection(projection_svd)

#Highlight: PCA
# pca_projection = PCA_custom(n_components=2).fit(X_train)
# # plot results
# fig, ax = plt.subplots(1, 1, figsize=(10,6))
#
# sns.scatterplot(
#     x = pca_projection[:,0],
#     y = pca_projection[:,1],
#     hue=y_train
# )
# ax.set_title('Iris Dataset')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# #plt.show()
# sns.despine()



print("Fitting rPCA...", end='')
t = time.time()
max_n_components = 2
rpca = R_pca(X_train)
L, S = rpca.fit(max_iter=5000, iter_print=100)
print(L.shape)
print(S.shape)
plot_projection(S)
exit()
_, rpca_evals, rpca_evecs = get_differential(L, max_n_components)
rpca_components = rpca_evecs.T
rpca_train_reduced = X_train.dot(rpca_evecs)
#rpca_test_reduced  = X_test.dot(rpca_evecs)
print("Took {:.3f} seconds.".format(time.time() - t))




