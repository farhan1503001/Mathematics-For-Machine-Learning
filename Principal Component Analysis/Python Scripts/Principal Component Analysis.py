import numpy as np

#First we will normalize
def normalize(X):
    """Normalize X such that it lies in zero mean

    Args:
        X (matrix): Input Matrix
    """
    N,D=X.shape
    mu=np.mean(X,axis=0)
    Xbar=X-mu
    return Xbar,mu
def eig(S):
    """Finds the eigenvector and eigen values
    
        S (_type_): Covariance matrix of X
    """
    eig_vals,eig_vecs=np.linalg.eig(S)
    sort_indices=np.argsort(eig_vals)[::-1]
    return eig_vals[sort_indices],eig_vecs[:,sort_indices]

def projection_matrix(B):
    """Compute the projection matrix on to the subspace B

    Args:
        B (_type_): _description_
    """
    p=B@np.linalg.inv(B.T@B)@B.T
    return p

def PCA(X,num_components):
    x_normalized,mean=normalize(X)
    #Now covariance matrix
    S=(x_normalized.T@ x_normalized)/x_normalized.shape[0]
    
    eig_vals,eig_vecs=eig(S)
    #Now finding principal values and principal vectors
    principal_vals,principal_vecs=eig_vals[:num_components],eig_vecs[:,:num_components]
    #Now reconstrunction
    principal_components=np.real(principal_vecs)
    reconst = (projection_matrix(principal_components)@x_normalized.T).T+mean
    return reconst, mean, principal_vals, principal_components


if __name__ == '__main__':
    X = np.array([[3, 6, 7],
              [8, 9, 0],
              [1, 5, 2]])

reconst, mean, principal_vals, principal_components = PCA(X, 1)

print('Cheacking mean...')
mean_exp = np.array([4, 20 / 3, 3])
np.testing.assert_allclose(mean, mean_exp, rtol=1e-5)
print('Mean is computed correctly!')

print('Checking principal values...')
principal_vals_exp = np.array([15.39677773])
np.testing.assert_allclose(principal_vals, principal_vals_exp, rtol=1e-5)
print('Principal Values are computed correctly!')

print('Checking principal components...')
principal_components_exp = np.array([[-0.68811066],
                                     [-0.40362611],
                                     [ 0.60298398]])
np.testing.assert_allclose(principal_components, principal_components_exp, rtol=1e-5)
print("Principal components are computed correctly!")

print('Checking reconstructed data...')
reconst_exp = np.array([[ 1.68166528,  5.30679755,  5.03153182],
                        [ 7.7868029 ,  8.8878974 , -0.31833472],
                        [ 2.53153182,  5.80530505,  4.2868029 ]])
np.testing.assert_allclose(reconst, reconst_exp, rtol=1e-5)
print("Reconstructed data is computed correctly!")