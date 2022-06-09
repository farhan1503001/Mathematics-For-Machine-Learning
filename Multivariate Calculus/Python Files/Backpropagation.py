import numpy as np
#Backpropagation depends actually on three processes
#we just have to remember three formulas which are
#.................................................a
#a(n)=sigma(z(n)) 
#z(n)=w(n)a(n-1)+b(n)
#sigma(z)=1/(1+exp(-z))
sigma=lambda z: 1/(1+np.exp(-z))
dsigma=lambda z: (np.cosh(z/2)**2)/4
#Setting parameters of the network
def reset_network(n1=6,n2=7):
    """Here, we will take only one input and feed it to the network

    Args:
        n1 (int, optional): Number of neurons in first hidden layer. Defaults to 6.
        n2 (int, optional): Number of neurons in second hidden layer. Defaults to 7.
    """
    global w1,w2,w3,b1,b2,b3
    w1=np.random.randn(n1,1)/2 #as only one input
    w2=np.random.randn(n2,n1)/2 #Middle layer will have n1 inputs
    w3=np.random.randn(2,n2) #Final layer with two outputs n2 inputs
    b1=np.random.randn(n1,1) #First layer bias
    b2=np.random.randn(n2,1) #second layer bias
    b3=np.random.randn(2,1) #Third layer

#Now we are defining our feedforward function
def feed_forward(a0):
    z1=w1@a0+b1
    a1=sigma(z1)
    z2=w1@a1+b2
    a2=sigma(z2)
    z3=w3@a2+b3
    a3=sigma(z3)
    
    return a0,z1,a1,z2,a2,z3,a3 

#Now we will perform the most important part backpropagation
#J(w3)=dc/dw(3)
#dc/dw(3)=dc/da3*da3/dz3*dz3/dw3
#dc/db(3)=dc/da3**da3/dz3*dz3/db3
#Now finding jacobian for w3
def j_w3(x,y):
    a0,z1,a1,z2,a2,z3,a3=feed_forward(x)
    #dc/da3=2(a3-y)
    j=2*(a3-y)
    #da3/dz3=dsigma(z3)
    j=j*dsigma(z3)
    #dz3/dw3=a2
    j=j@a2.T/x.size
    
    return j

def j_b3(x,y):
    a0,z1,a1,z2,a2,z3,a3=feed_forward(x)
    j=2*(a3-y)
    j=j*dsigma(z3)
    #dz3/db3=1
    j=np.sum(j,axis=1,keepdims=True)/x.size
    
    return j

#Now backprop for layer 2 
#...............................
#dc/dw2=dc/da3*da3/da2*da2/dz2*dz2/dw2
#dc/db2=dc/da3*da3/da2*da2/dz2*dz2/db2
#da3/da2=dsigma(z3)w3

def j_w2(x,y):
    a0, z1, a1, z2, a2, z3, a3 = feed_forward(x)
    j=2*(a3-y)
    j=j*dsigma(z3)
    j=(j.T@w3).T
    j=j*dsigma(z2)
    j=j@a1.T/x.size
    
    return j
def d_b2(x,y):
    a0, z1, a1, z2, a2, z3, a3 = feed_forward(x)
    j=2*(a3-y)
    j=j*dsigma(z3)
    j=(j.T@w3).T
    j=j*dsigma(z2)
    j=np.sum(j,axis=1,keepdims=True)/x.size
    return j

#Now backpropagation for first layer
#dc/dw1=dc/da3*da3/da2*da2/da1*da1/dz1*dz1/dw1
#dc/db1=dc/da3*da3/da2*da2/da1*da1/dz1*dz1/db1
def j_w1(x,y):
    a0, z1, a1, z2, a2, z3, a3 = feed_forward(x)
    j=2*(a3-y)
    j=j*dsigma(z3)
    j=(j.T@w3).T
    j=j*dsigma(z2)
    j=(j.T@w2).T
    j=j*dsigma(z1)
    j=(j@a1.T)/x.size
    return j
def d_b1(x,y):
    a0, z1, a1, z2, a2, z3, a3 = feed_forward(x)
    j=2*(a3-y)
    j=j*dsigma(z3)
    j=(j.T@w3).T
    j=j*dsigma(z2)
    j=(j.T@w2).T
    j=j*dsigma(z1)
    
    j=np.sum(j,axis=1,keepdims=True)/x.size
    return j


    
    
    
    