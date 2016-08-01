# Toolboxes used
import numpy as np # For linear algebra
import scipy.io # For loading matlab datasets
from sklearn import decomposition as decomp # PCA-like methods
from sklearn import manifold # Manifold-based methods
from sklearn import neighbors

import matplotlib # Matlab style plotting package for python
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)

import dimred_funcs as df # Importing the custom subfunctions from a separate file

# Loading an example dataset
X = scipy.io.loadmat('freyface.mat')['X'].astype(float)
print('The dimensions of our dataset are: {}').format(X.shape)

# Built-in PCA results
pca = decomp.PCA(n_components=min(X.shape))
pca.fit(X.T)
eigvec = pca.components_.T
eigval = pca.explained_variance_
print_n = 16
print('The first {} PCA eigenvalues are {}.\n').format(print_n, eigval[0:print_n])
# df.showfreyface(eigvec.T)

# Wrong implementation
Dun,Vun = np.linalg.eig(np.dot(X,X.T)) # Get eigenvalues and eigenvectors of XX^T

order = Dun.argsort()[::-1] # Get the descending ordering of eigenvalues
Dun = Dun[order]
Vun = Vun[:,order]

print_n = 16;
print('The first {} incorrect PCA eigenvalues are {}.\n').format(print_n, Dun[0:print_n])

# df.showfreyface(Vun[:,0:16])