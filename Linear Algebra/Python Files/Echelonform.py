import numpy as np
from typing import List
#Here we are writing script which will convert a 4*4 matrix to echelon form
#At first we will elimante first row 
class MatrixSingularException(Exception):pass
def fixrowzero(mat:list[int])->List[int]:
    """
    first value a[0,0] will be 1
    we should check if the value is 0 or not
    """
    if mat[0,0]==0:
        mat[0]=mat[0]+mat[1]
    if mat[0,0]==0:
        mat[0]=mat[0]+mat[2]
    if mat[0,0]==0:
        mat[0]=mat[0]+mat[3]
    if mat[0,0]==0:
        raise MatrixSingularException()
    #Now we will make a[0,0]=1
    mat[0]=mat[0]/mat[0,0]
    return mat
#Now we will write a function for fixing row 1
def fixrowone(mat:List[int])->List[int]:
    """
    Here we have to make a[1,0]=0 and a[1,1]=1 that's our goal
    """
    mat[1]=mat[1]-mat[1,0]*mat[0]
    if mat[1,1]==0:
        mat[1]=mat[1]+mat[2]
        mat[1]=mat[1]-mat[1,0]*mat[0]
    if mat[1,1]==0:
        mat[1]=mat[1]+mat[3]
        mat[1]=mat[1]-mat[1,0]*mat[0]
    if mat[1,1]==0:
        raise MatrixSingularException()
    mat[1]=mat[1]/mat[1,1]
    return mat
#Now we will fix row-2 a[2,0]=0,a[2,1]=0,a[2,2]=1
def fixrowtwo(mat:List[int])->List[int]:
    mat[2]=mat[2]-mat[2,0]*mat[0]
    mat[2]=mat[2]-mat[2,1]*mat[1]
    
    if mat[2,2]==0:
        mat[2]=mat[2]+mat[3]
        mat[2]=mat[2]-mat[2,0]*mat[0]
        mat[2]=mat[2]-mat[2,1]*mat[1]
    if mat[2,2]==0:
        raise MatrixSingularException()
    
    mat[2]=mat[2]/mat[2,2]
    return mat 
#Now we will fix our row 3
def fixrowthree(mat:List[int])->List[int]:
    mat[3]=mat[3]-mat[3,0]*mat[0]
    mat[3]=mat[3]-mat[3,1]*mat[1]
    mat[3]=mat[3]-mat[3,2]*mat[2]
    
    if mat[3,3]==0:
        raise MatrixSingularException()
    mat[3]=mat[3]/mat[3,3]
    return mat

#Now just a singularity checker function
def isSingular(a:List[int])->List[int]:
    #Copy it
    b=np.array(a,dtype=np.float_)
    try:
        fixrowzero(b)
        fixrowone(b)
        fixrowtwo(b)
        fixrowthree(b)
    except MatrixSingularException:
        return False
    return True

if __name__ == '__main__':
    
    A = np.array([
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 5, 5]
        ], dtype=np.float_)
    print(isSingular(A))
    

    A = np.array([
            [0, 7, -5, 3],
            [2, 8, 0, 4],
            [3, 12, 0, 5],
            [1, 3, 1, 3]
        ], dtype=np.float_)
    
    print(fixrowzero(A))
    print(fixrowone(A))
    print(fixrowtwo(A))
    print(fixrowthree(A))
    print("Final value")
    print(A)



