import numpy as np

def page_rank(L,num_pages):
    """L is the link matrix or graph adjacency matrix
    where r is the rank matrix we will try to update r until lastr and r 
    becomes same

    Args:
        L (_type_): Link Matrix
        r (_type_): rank matrix
    """
    #Defining r
    r=100*np.ones(num_pages)/num_pages
    lastr=r
    r=L@r
    iteration=0
    print("starting")
    while np.linalg.norm(lastr-r)>0.01:
        lastr=r
        r=L@r
        iteration+=1
        print(iteration)
    print("Number of iterations needed to converge ",iteration)
    print("Rank vector is :",r)
    return r

def page_rank_dumpfactor(L,d,num_pages):
    r=100*np.ones(num_pages)/num_pages
    L2=d*L+(1-d)/num_pages * np.ones([num_pages,num_pages])
    lastr=r
    r=L2@r
    iteration=0
    while np.linalg.norm(lastr-r)>0.01:
        lastr=r
        r=L2@r
        iteration+=1
        print(iteration)
    print("Number of iterations needed to converge ",iteration)
    print("Rank vector is: ",r)
    return r
def eigenrank(link):
    eigen_val,eigen_vec=np.linalg.eig(link)
    #Now sort eigen values
    order=np.absolute(eigen_val).argsort()[::-1]
    evals=eigen_val[order]
    evecs=eigen_vec[:,order]
    r = evecs[:, 0]
    final_r=100 * np.real(r / np.sum(r))
    print("Rank Matrix")
    print(final_r)
    return final_r

if __name__ == '__main__':
    link=np.array([[0,   1/2, 1/3, 0, 0,   0, 0 ],
               [1/3, 0,   0,   0, 1/2, 0, 0 ],
               [1/3, 1/2, 0,   1, 0,   1/3, 0 ],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0 ],
               [0,   0,   0,   0, 0,   0, 0 ],
               [0,   0,   1/3, 0, 0,   0, 0 ],
               [0,   0,   0,   0, 0,   1/3, 1 ]])
    
    print("Using dumping factor")
    print()
    page_rank_dumpfactor(link,0.7,link.shape[0])
    print("Not Using Dumping Factor")
    print()
    page_rank(link,link.shape[0])
    print("Using eigen vectors")
    eigenrank(link)