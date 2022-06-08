import numpy as np

#Gram schimdt Process is used for orthogonalizing a vector or matrix by finding out orthonormal basis
very_small_number=1e-14
def gsbasis4(mat_a):
    """Here we will perform gsbasis process for making each coloumn orthonormal

    Args:
        mat_a (numpy): _description_
    """
    B=np.array(mat_a,dtype=np.float_)
    #Copied our matrix for calculation
    #Now first column just normalize
    B[:,0]=B[:,0]/np.linalg.norm(B[:,0])
    #Now second column we have to remove 1st column's projection
    B[:,1]=B[:,1]-B[:,1]@ B[:,0]*B[:,0]
    if np.linalg.norm(B[:,1])>very_small_number:
        B[:,1]=B[:,1]/np.linalg.norm(B[:,1])
    else:
        B[:,1]=np.zeros_like(B[:,1])
    #Now third row we have to remove projection of 1st and 2nd column
    B[:,2]=B[:,2]-B[:,2]@B[:,0]*B[:,0]-B[:,2]@B[:,1]*B[:,1]
    if np.linalg.norm(B[:,2])>very_small_number:
        B[:,2]=B[:,2]/np.linalg.norm(B[:,2])
    else:
        B[:,2]=np.zeros_like(B[:,2])
    #Now fourth row we have to remove projection 1st 2nd 3rd column
    B[:,3]=B[:,3]-B[:,3]@B[:,0]*B[:,0]-B[:,3]@B[:,1]*B[:,1]-B[:,3]@B[:,2]*B[:,2]
    
    if np.linalg.norm(B[:,3])>very_small_number:
        B[:,3]=B[:,3]/np.linalg.norm(B[:,3])
    else:
        B[:,3]=np.zeros_like(B[:,3])
    
    #Finally return B
    return B

def gsBasis(Mat):
    """Generalized Function for finding orthonormal basis vector of the matrix mat

    Args:
        Mat (_type_): _description_
    """
    B=np.array(Mat,dtype=np.float_)
    #We will iterate over columns 
    for i in range(B.shape[1]):
        for j in range(i):
            B[:,i]=B[:,i]-B[:,i]@B[:,j]*B[:,j]
        if np.linalg.norm(B[:,i]>very_small_number):
            B[:,i]=B[:,i]/np.linalg.norm(B[:,i])
        else:
            B[:,i]=np.zeros_like(B[:,i])
    return B

if __name__ == '__main__':
    V = np.array([[1,0,2,6],
              [0,1,8,2],
              [2,8,3,1],
              [1,-6,2,3]], dtype=np.float_)
    
    print(gsbasis4(V))
    U = gsbasis4(V)
    print(gsbasis4(U))
    A = np.array([[3,2,3],
              [2,5,-1],
              [2,4,8],
              [12,2,1]], dtype=np.float_)
    print(gsBasis(A))


    
    