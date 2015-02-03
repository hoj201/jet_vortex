import numpy as np
from scipy.integrate import odeint

# FOR DATA PLOTTING
#import matplotlib.pyplot as plt

# FOR 3D PLOTTING UNCOMMENT THE FOLLOWING LINES IN ADDITION TO THE ABOVE LINE
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')

N = 3 #number of vortices
SIGMA = 1.
def kernel(Z,z):
    #z_c is usually a (N, 2) np arrays
    X = z[:,0]
    Y = z[:,1]
    x = z[:,0]
    y = z[:,1]
    dx = np.transpose(np.tile(X, (1, N) ))- np.tile( x , (1 , N))
    dy = np.transpose(np.tile(Y, (1, N) ))- np.tile( y , (1 , N))
    r_sq = dx**2 + dy**2
    K = np.exp( -r_sq/(2*SIGMA**2))
    DK = np.zeros( ( np.size(X) , np.size(x) , 2) )
    DK[:,:,0] = dx*K / SIGMA**2
    DK[:,:,1] = dy*K / SIGMA**2
    return K, DK

def Hermite( k , x):
    #Calculate the 'statisticians' Hermite polynomials
    if k==0:
        return 1.
    elif k==1:
        return x
    elif k==2:
        return x**2 -1
    elif k==3:
        return x**3 - 3*x
    elif k==4:
        return x**4 - 6*x**2 + 3
    elif k==5:
        return x**5 - 10*x**3 + 15*x
    else:
        print 'error in Hermite function, unknown formula for k=' + str(k)
        
def stream_function(Z,z):
    K,DK =  kernel(Z,z)
    return 0

def velocity_field(Z,z,gamma):
    u = np.zeros( (np.size(Z[:,0]), 2) )
    K,DK = kernel(Z,z)
    grad_psi = np.einsum( 'j,ija', gamma, DK) #has indices ia
    u[:,0] = -grad_psi[:,1]
    u[:,1] = grad_psi[:,0]
    return u


    
z = np.random.rand(N,2)
gamma = np.random.rand(N)
print np.shape(z)
K = kernel(z,z)
print np.shape(K)
    
