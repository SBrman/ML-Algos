#! python3

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"


import numpy as np


def PCA(X, num_dim=None):
    # X_pca, num_dim = X, len(X[0]) # placeholder

    # Getting the standard deviation and replacing the zeros with 1 to avoid divide by zero errors.
    # Since the goal is to normalize the data by dividing by std, std of 0 would mean the 
    # feature value is equal to average everywhere and would not contribute to analysis. In that case
    # leaving the value as is.
    x_std = X.std(axis=0)
    x_std[x_std == 0.0] = 1.0

    # Standardize the features by subtracting the feature means and dividing by the feature stds.
    X_pca = (X - X.mean(axis=0)) / x_std

    # Get the covariance matrix for X_pca^T. Taking transpose to ensure the covariance matrix 
    # gives the covariance between pairs of features
    covariance_matrix = X_pca.T.dot(X_pca) / X_pca.shape[0]

    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # select the reduced dimensions that keep >90% of the variance
    if num_dim is None:
        for i, _ in enumerate(eigen_values, 1):
            num_dim = i
            if eigen_values[:i].sum() / eigen_values.sum() > 0.9:
                break

    # Get the top eigen vectors for >90% variance
    top_eigen_vectors = eigen_vectors[:, :num_dim]

    # project the high-dimensional data to low-dimensional one
    X_pca = X_pca.dot(top_eigen_vectors)
    
    return X_pca, num_dim


if __name__ == "__main__":
    data = np.genfromtxt("../data/Digits089.csv", delimiter=",")
    X = data[:, 2:]
    y = data[:, 1]

    X_pca_, num_dim_pca = PCA(X)
    print(X_pca_, num_dim_pca)
