import numpy as np
very_small_number=1e-4
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
def build_reflection_matrix(Basis):
    """Our reflection matrix will first reflect by this matrix to the
    it's dimension then it will inversed to the real dimension 
    Args:
        Basis (_type_): _description_
    """
    orthobasis=gsBasis(Basis)
    #Here we perform orthogonalisation so that it's inverse is Transpose so it becomes easier
    #Now reflection matrix in ideal axis
    mirror=np.array([
        [1,0],
        [0,-1]
    ])
    #Now will will mirror it in it's own then transform to ideal
    
    Transformer=orthobasis@mirror@np.transpose(orthobasis)
    
    return Transformer

if __name__ == '__main__':
    bearbasis=np.array(
        [[1,-1],
        [1.5,2]
        ]
    )
    Transform_matrix=build_reflection_matrix(bearbasis)
    
    print(Transform_matrix)
    