import numpy as np

#projec_1d=bb.T/norm(b)**2
def projection_matrix_1d(b):
    """Projection matrix for a 1d subspace with basis vector b

    Args:
        b (vector): Basis vector b
    """
    dimension,=b.shape
    Project=np.outer(b,b)/(np.linalg.norm(b)**2)
    return Project

def project_1d(x,b):
    """Compute the projection associated with projection matrix
    Args:
        x (_type_): vector to be projected
        b (_type_): basis for subspace 
    """
    projection=np.dot(projection_matrix_1d(b),x)
    return projection

def projection_matrix_general(B):
    """Projection matrix for projecting to the subspace

    Args:
        B (_type_): Basis vector of the subspace
    """
    Projection_mat=B @ np.linalg.inv(B.T @ B) @ B.T
    return Projection_mat
def Projection_general(x,B):
    """Compute the projected value of the vector x

    Args:
        x (_type_): _description_
        B (_type_): _description_
    """
    projected=projection_matrix_general(B) @ x
    return projected
if __name__ == '__main__':
    #Will view the projection matrix for 1D
    mat=np.array([1,2,2])
    projection_mat=projection_matrix_1d(mat)
    print(projection_mat)
    projected_val=project_1d(np.ones(3),mat)
    print(projected_val)
    B = np.array([[1, 0],
              [1, 1],
              [1, 2]])
    print(projection_matrix_general(B))
    print(Projection_general(np.array([6,0,0]).reshape(-1,1),B))