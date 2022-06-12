import numpy as np

def distance(x0,x1):
    """Finds the orthogonal distance between two vector x0 and x1

    Args:
        x0 (_type_): _description_
        x1 (_type_): _description_
    """
    
    diff=x0-x1
    distance=np.sqrt(diff @ diff.T)
    return distance

def angle(x0,x1):
    """Finds the angle between two vector x0 and x1

    Args:
        x0 (_type_): _description_
        x1 (_type_): _description_
    """
    dot_prod=np.dot(x0,x1.T)
    norm_x=np.linalg.norm(np.dot(x0.T,x0))
    norm_y=np.linalg.norm(np.dot(x1.T,x1))
    res=dot_prod/(norm_x*norm_y)
    angle=np.arccos(res)
    return angle   

if __name__ == '__main__':
    x1=np.array([
        [1,3]   
    ])
    x2=np.array([-1,4])
    
    print(distance(x1,x2))
    print(angle(x1,x2))