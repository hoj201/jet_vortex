import numpy as np
import time
import os
import pickle
from scipy.special import expi
import sympy as sp

class Jet_vortex:
    # A class for jet-vortices
    def __init__( self, order=0, N=1 ):
        self.order = order
        self.N = N
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.gamma = np.zeros(N)
        if order > 0:
            self.gamma_x = np.zeros(N)
            self.gamma_y = np.zeros(N)
            if order > 1:
                self.gamma_xx = np.zeros(N)
                self.gamma_xy = np.zeros(N)
                self.gamma_yy = np.zeros(N)
    def use_state( self, state ):
        #converts numpy-array into a jet-vortex
        N = self.N
        self.x = np.copy( state[0:N] )
        self.y = np.copy( state[N:(2*N)] )
        self.gamma = np.copy( state[(2*N):(3*N)] )
        if self.order > 0:
            self.gamma_x = np.copy( state[(3*N):(4*N)] )
            self.gamma_y = np.copy( state[(4*N):(5*N)] )
            if self.order > 1:
                self.gamma_xx = np.copy( state[(5*N):(6*N)] )
                self.gamma_xy = np.copy( state[(6*N):(7*N)] )
                self.gamma_yy = np.copy( state[(7*N):(8*N)] )

    def state(self):
        # converts a jet-vortex into a numpy array
        N = self.N
        out = np.zeros( (3 + 2*(self.order > 0)\
                         + 3*(self.order>1))*N )
        out[0:N] = self.x
        out[N:(2*N)] = self.y 
        out[(2*N):(3*N)] = self.gamma
        if self.order > 0:
            out[(3*N):(4*N)] = self.gamma_x
            out[(4*N):(5*N)] = self.gamma_y
            if self.order > 1:
                out[(5*N):(6*N)] = self.gamma_xx
                out[(6*N):(7*N)] = self.gamma_xy
                out[(7*N):(8*N)] = self.gamma_yy
        return out

    def dx(self , X = None):
        if X is None:
            X = self.x
        store =  np.tile(X ,(np.size(self.x),1))
        return store.T - np.tile(self.x, (np.size(X),1) )

    def dy(self , Y = None):
        if Y is None:
            Y = self.y
        store =  np.tile(Y ,(np.size(self.y),1))
        return store.T - np.tile(self.y, (np.size(Y),1) )    

# THE FOLLOWING BLOCK OF CODE DERIVES DERIVATIVES OF THE KERNEL G_DELTA

# delta should always be normalized to 1.0
# so that 5th order derivatives do not blow-up
delta = 1.
x = sp.Symbol('x')
y = sp.Symbol('y')
prefix = './symbolic_expressions/sym_'
num_precomputed_formulas = 2*(2 + 3 + 4 + 5 + 6)
if len(os.listdir('./symbolic_expressions')) == num_precomputed_formulas:
    print "Loading symbolic kernel expressions..."
    sym_DxG = pickle.load(open(prefix+'DxG.dat','r'))
    sym_DyG = pickle.load(open(prefix+'DyG.dat','r'))
    sym_DxxG = pickle.load(open(prefix+'DxxG.dat','r'))
    sym_DxyG = pickle.load(open(prefix+'DxyG.dat','r'))
    sym_DyyG = pickle.load(open(prefix+'DyyG.dat','r'))
    sym_DxxxG = pickle.load(open(prefix+'DxxxG.dat','r'))
    sym_DxxyG = pickle.load(open(prefix+'DxxyG.dat','r'))
    sym_DxyyG = pickle.load(open(prefix+'DxyyG.dat','r'))
    sym_DyyyG = pickle.load(open(prefix+'DyyyG.dat','r'))
    sym_DxxxxG = pickle.load(open(prefix+'DxxxxG.dat','r'))
    sym_DxxxyG = pickle.load(open(prefix+'DxxxyG.dat','r'))
    sym_DxxyyG = pickle.load(open(prefix+'DxxyyG.dat','r'))
    sym_DxyyyG = pickle.load(open(prefix+'DxyyyG.dat','r'))
    sym_DyyyyG = pickle.load(open(prefix+'DyyyyG.dat','r'))
    sym_DxxxxxG = pickle.load(open(prefix+'DxxxxxG.dat','r'))
    sym_DxxxxyG = pickle.load(open(prefix+'DxxxxyG.dat','r'))
    sym_DxxxyyG = pickle.load(open(prefix+'DxxxyyG.dat','r'))
    sym_DxxyyyG = pickle.load(open(prefix+'DxxyyyG.dat','r'))
    sym_DxyyyyG = pickle.load(open(prefix+'DxyyyyG.dat','r'))
    sym_DyyyyyG = pickle.load(open(prefix+'DyyyyyG.dat','r'))
    sym_DxG_near = pickle.load(open(prefix+'DxG_near.dat','r'))    
    sym_DyG_near = pickle.load(open(prefix+'DyG_near.dat','r'))    
    sym_DxxG_near = pickle.load(open(prefix+'DxxG_near.dat','r'))    
    sym_DxyG_near = pickle.load(open(prefix+'DxyG_near.dat','r'))
    sym_DyyG_near = pickle.load(open(prefix+'DyyG_near.dat','r'))
    sym_DxxxG_near = pickle.load(open(prefix+'DxxxG_near.dat','r'))    
    sym_DxxyG_near = pickle.load(open(prefix+'DxxyG_near.dat','r'))    
    sym_DxyyG_near = pickle.load(open(prefix+'DxyyG_near.dat','r'))    
    sym_DyyyG_near = pickle.load(open(prefix+'DyyyG_near.dat','r'))    
    sym_DxxxxG_near = pickle.load(open(prefix+'DxxxxG_near.dat','r'))    
    sym_DxxxyG_near = pickle.load(open(prefix+'DxxxyG_near.dat','r'))    
    sym_DxxyyG_near = pickle.load(open(prefix+'DxxyyG_near.dat','r'))    
    sym_DxyyyG_near = pickle.load(open(prefix+'DxyyyG_near.dat','r'))    
    sym_DyyyyG_near = pickle.load(open(prefix+'DyyyyG_near.dat','r'))    
    sym_DxxxxxG_near = pickle.load(open(prefix+'DxxxxxG_near.dat','r'))    
    sym_DxxxxyG_near = pickle.load(open(prefix+'DxxxxyG_near.dat','r'))    
    sym_DxxxyyG_near = pickle.load(open(prefix+'DxxxyyG_near.dat','r'))    
    sym_DxxyyyG_near = pickle.load(open(prefix+'DxxyyyG_near.dat','r'))    
    sym_DxyyyyG_near = pickle.load(open(prefix+'DxyyyyG_near.dat','r'))    
    sym_DyyyyyG_near = pickle.load(open(prefix+'DyyyyyG_near.dat','r'))    
    print "Complete."
else:
    print "Calculating derivatives of kernel symbolically..."
    rr = x**2 + y**2
    rho = rr / delta**2
    sym_DxG = x*(sp.exp( - rho ) - 1 )/(2*sp.pi*rr)
    sym_DyG = y*(sp.exp( - rho ) - 1 )/(2*sp.pi*rr)
    sym_DxG_near = sp.series(sp.series(sym_DxG,x).removeO()\
                             ,y).removeO()
    sym_DyG_near = sp.series(sp.series(sym_DyG,x).removeO()\
                             ,y).removeO()
    pickle.dump(sym_DxG,open(prefix+'DxG.dat','w'))
    pickle.dump(sym_DyG,open(prefix+'DyG.dat','w'))
    pickle.dump(sym_DxG_near,open(prefix+'DxG_near.dat','w'))
    pickle.dump(sym_DyG_near,open(prefix+'DyG_near.dat','w'))
    print "Computed first order derivatives"
    sym_DxxG = sp.diff( sym_DxG , x )
    sym_DxyG = sp.diff( sym_DxG , y )
    sym_DyyG = sp.diff( sym_DyG , y )
    sym_DxxG_near = sp.series(sp.series(sym_DxxG,x).removeO()\
                             ,y).removeO()
    sym_DxyG_near = sp.series(sp.series(sym_DxyG,x).removeO()\
                             ,y).removeO()
    sym_DyyG_near = sp.series(sp.series(sym_DyyG,x).removeO()\
                             ,y).removeO()
    pickle.dump(sym_DxxG,open(prefix+'DxxG.dat','w'))
    pickle.dump(sym_DxyG,open(prefix+'DxyG.dat','w'))
    pickle.dump(sym_DyyG,open(prefix+'DyyG.dat','w'))
    pickle.dump(sym_DxxG_near,open(prefix+'DxxG_near.dat','w'))
    pickle.dump(sym_DxyG_near,open(prefix+'DxyG_near.dat','w'))
    pickle.dump(sym_DyyG_near,open(prefix+'DyyG_near.dat','w'))
    print "Computed second order derivatives"
    sym_DxxxG = sp.diff( sym_DxxG , x )
    sym_DxxyG = sp.diff( sym_DxxG , y )
    sym_DxyyG = sp.diff( sym_DxyG , y )
    sym_DyyyG = sp.diff( sym_DyyG , y )
    sym_DxxxG_near = sp.series(sp.series(sym_DxxxG,x).removeO()\
                             ,y).removeO()
    sym_DxxyG_near = sp.series(sp.series(sym_DxxyG,x).removeO()\
                             ,y).removeO()
    sym_DxyyG_near = sp.series(sp.series(sym_DxyyG,x).removeO()\
                             ,y).removeO()
    sym_DyyyG_near = sp.series(sp.series(sym_DyyyG,x).removeO()\
                             ,y).removeO()
    pickle.dump(sym_DxxxG,open(prefix+'DxxxG.dat','w'))
    pickle.dump(sym_DxxyG,open(prefix+'DxxyG.dat','w'))
    pickle.dump(sym_DxyyG,open(prefix+'DxyyG.dat','w'))
    pickle.dump(sym_DyyyG,open(prefix+'DyyyG.dat','w'))
    pickle.dump(sym_DxxxG_near,open(prefix+'DxxxG_near.dat','w'))
    pickle.dump(sym_DxxyG_near,open(prefix+'DxxyG_near.dat','w'))
    pickle.dump(sym_DxyyG_near,open(prefix+'DxyyG_near.dat','w'))
    pickle.dump(sym_DyyyG_near,open(prefix+'DyyyG_near.dat','w'))
    print 'Computed third order derivatives'
    sym_DxxxxG = sp.diff( sym_DxxxG , x )
    sym_DxxxyG = sp.diff( sym_DxxyG , x )
    sym_DxxyyG = sp.diff( sym_DxyyG , x )
    sym_DxyyyG = sp.diff( sym_DyyyG , x )
    sym_DyyyyG = sp.diff( sym_DyyyG , y )
    sym_DxxxxG_near = sp.series(sp.series(sym_DxxxxG,x).removeO()\
                             ,y).removeO()
    sym_DxxxyG_near = sp.series(sp.series(sym_DxxxyG,x).removeO()\
                             ,y).removeO()
    sym_DxxyyG_near = sp.series(sp.series(sym_DxxyyG,x).removeO()\
                             ,y).removeO()
    sym_DxyyyG_near = sp.series(sp.series(sym_DxyyyG,x).removeO()\
                             ,y).removeO()
    sym_DyyyyG_near = sp.series(sp.series(sym_DyyyyG,x).removeO()\
                             ,y).removeO()
    pickle.dump(sym_DxxxxG,open(prefix+'DxxxxG.dat','w'))
    pickle.dump(sym_DxxxyG,open(prefix+'DxxxyG.dat','w'))
    pickle.dump(sym_DxxyyG,open(prefix+'DxxyyG.dat','w'))
    pickle.dump(sym_DxyyyG,open(prefix+'DxyyyG.dat','w'))
    pickle.dump(sym_DyyyyG,open(prefix+'DyyyyG.dat','w'))
    pickle.dump(sym_DxxxxG_near,open(prefix+'DxxxxG_near.dat','w'))
    pickle.dump(sym_DxxxyG_near,open(prefix+'DxxxyG_near.dat','w'))
    pickle.dump(sym_DxxyyG_near,open(prefix+'DxxyyG_near.dat','w'))
    pickle.dump(sym_DxyyyG_near,open(prefix+'DxyyyG_near.dat','w'))
    pickle.dump(sym_DyyyyG_near,open(prefix+'DyyyyG_near.dat','w'))
    print 'Computed fourth order derivatives'
    sym_DxxxxxG = sp.diff( sym_DxxxxG , x )
    sym_DxxxxyG = sp.diff( sym_DxxxyG , x )
    sym_DxxxyyG = sp.diff( sym_DxxyyG , x )
    sym_DxxyyyG = sp.diff( sym_DxyyyG , x )
    sym_DxyyyyG = sp.diff( sym_DxyyyG , y )
    sym_DyyyyyG = sp.diff( sym_DyyyyG , y )
    sym_DxxxxxG_near = sp.series(sp.series(sym_DxxxxxG,x).removeO()\
                             ,y).removeO()
    sym_DxxxxyG_near = sp.series(sp.series(sym_DxxxxyG,x).removeO()\
                             ,y).removeO()
    sym_DxxxyyG_near = sp.series(sp.series(sym_DxxxyyG,x).removeO()\
                             ,y).removeO()
    sym_DxxyyyG_near = sp.series(sp.series(sym_DxxyyyG,x).removeO()\
                             ,y).removeO()
    sym_DxyyyyG_near = sp.series(sp.series(sym_DxyyyyG,x).removeO()\
                             ,y).removeO()
    sym_DyyyyyG_near = sp.series(sp.series(sym_DyyyyyG,x).removeO()\
                             ,y).removeO()
    pickle.dump(sym_DxxxxxG,open(prefix+'DxxxxxG.dat','w'))
    pickle.dump(sym_DxxxxyG,open(prefix+'DxxxxyG.dat','w'))
    pickle.dump(sym_DxxxyyG,open(prefix+'DxxxyyG.dat','w'))
    pickle.dump(sym_DxxyyyG,open(prefix+'DxxyyyG.dat','w'))
    pickle.dump(sym_DxyyyyG,open(prefix+'DxyyyyG.dat','w'))
    pickle.dump(sym_DyyyyyG,open(prefix+'DyyyyyG.dat','w'))
    pickle.dump(sym_DxxxxxG_near,open(prefix+'DxxxxxG_near.dat','w'))
    pickle.dump(sym_DxxxxyG_near,open(prefix+'DxxxxyG_near.dat','w'))
    pickle.dump(sym_DxxxyyG_near,open(prefix+'DxxxyyG_near.dat','w'))
    pickle.dump(sym_DxxyyyG_near,open(prefix+'DxxyyyG_near.dat','w'))
    pickle.dump(sym_DxyyyyG_near,open(prefix+'DxyyyyG_near.dat','w'))
    pickle.dump(sym_DyyyyyG_near,open(prefix+'DyyyyyG_near.dat','w'))
    print 'Computed fifth order derivatives'

# Convert symbolic expressions into vectorized numpy functions
print "Lambdifying..."
DxG = sp.lambdify( (x,y) , sym_DxG , "numpy" ) 
DyG = sp.lambdify( (x,y) , sym_DyG , "numpy" ) 
DxxG = sp.lambdify( (x,y) , sym_DxxG , "numpy" ) 
DxyG = sp.lambdify( (x,y) , sym_DxyG , "numpy" ) 
DyyG = sp.lambdify( (x,y) , sym_DyyG , "numpy" ) 
DxxxG = sp.lambdify( (x,y) , sym_DxxxG , "numpy" ) 
DxxyG = sp.lambdify( (x,y) , sym_DxxyG , "numpy" ) 
DxyyG = sp.lambdify( (x,y) , sym_DxyyG , "numpy" ) 
DyyyG = sp.lambdify( (x,y) , sym_DyyyG , "numpy" ) 
DxxxxG = sp.lambdify( (x,y) , sym_DxxxxG , "numpy" ) 
DxxxyG = sp.lambdify( (x,y) , sym_DxxxyG , "numpy" ) 
DxxyyG = sp.lambdify( (x,y) , sym_DxxyyG , "numpy" ) 
DxyyyG = sp.lambdify( (x,y) , sym_DxyyyG , "numpy" ) 
DyyyyG = sp.lambdify( (x,y) , sym_DyyyyG , "numpy" ) 
DxxxxxG = sp.lambdify( (x,y) , sym_DxxxxxG , "numpy" ) 
DxxxxyG = sp.lambdify( (x,y) , sym_DxxxxyG , "numpy" ) 
DxxxyyG = sp.lambdify( (x,y) , sym_DxxxyyG , "numpy" ) 
DxxyyyG = sp.lambdify( (x,y) , sym_DxxyyyG , "numpy" ) 
DxyyyyG = sp.lambdify( (x,y) , sym_DxyyyyG , "numpy" ) 
DyyyyyG = sp.lambdify( (x,y) , sym_DyyyyyG , "numpy" ) 

DxG_near = sp.lambdify( (x,y) , sym_DxG_near , "numpy" ) 
DyG_near = sp.lambdify( (x,y) , sym_DyG_near , "numpy" ) 
DxxG_near = sp.lambdify( (x,y) , sym_DxxG_near , "numpy" ) 
DxyG_near = sp.lambdify( (x,y) , sym_DxyG_near , "numpy" ) 
DyyG_near = sp.lambdify( (x,y) , sym_DyyG_near , "numpy" ) 
DxxxG_near = sp.lambdify( (x,y) , sym_DxxxG_near , "numpy" ) 
DxxyG_near = sp.lambdify( (x,y) , sym_DxxyG_near , "numpy" ) 
DxyyG_near = sp.lambdify( (x,y) , sym_DxyyG_near , "numpy" ) 
DyyyG_near = sp.lambdify( (x,y) , sym_DyyyG_near , "numpy" ) 
DxxxxG_near = sp.lambdify( (x,y) , sym_DxxxxG_near , "numpy" ) 
DxxxyG_near = sp.lambdify( (x,y) , sym_DxxxyG_near , "numpy" ) 
DxxyyG_near = sp.lambdify( (x,y) , sym_DxxyyG_near , "numpy" ) 
DxyyyG_near = sp.lambdify( (x,y) , sym_DxyyyG_near , "numpy" ) 
DyyyyG_near = sp.lambdify( (x,y) , sym_DyyyyG_near , "numpy" ) 
DxxxxxG_near = sp.lambdify( (x,y) , sym_DxxxxxG_near , "numpy" ) 
DxxxxyG_near = sp.lambdify( (x,y) , sym_DxxxxyG_near , "numpy" ) 
DxxxyyG_near = sp.lambdify( (x,y) , sym_DxxxyyG_near , "numpy" ) 
DxxyyyG_near = sp.lambdify( (x,y) , sym_DxxyyyG_near , "numpy" ) 
DxyyyyG_near = sp.lambdify( (x,y) , sym_DxyyyyG_near , "numpy" ) 
DyyyyyG_near = sp.lambdify( (x,y) , sym_DyyyyyG_near , "numpy" ) 
print "Complete."

def kernel( jv , X=None , Y=None , m=0 , n=0 ):
    # computes \partial_x^m \partial_y^n G_\delta( Z_i - z_j)
    # output should be of shape [size(X),size(jv.x)]
    if X is None:
        X = jv.x
    if Y is None:
        Y = jv.y
    dx = jv.dx(X)
    dy = jv.dy(Y)
    r2 = dx**2 + dy**2
    r = np.sqrt(r2)
    rho = r2 / (delta**2)
    store = np.zeros( [np.size(X), jv.N] )
    near = (r < delta/10.)
    far = (r >= delta/10.)
    r_far = r*far+2.0*near
    r2_far = r_far**2
    dx_far = dx + near*1.0
    dx_near = near*dx
    dy_far = dy + near*1.0
    dy_near = near*dy
    if (m==0) & (n==0):
        t0 =  time.time()
        kernel_far = (expi( -r2_far/((delta)**2) ) \
                         - 2.0*np.log(r_far))/(4.0*np.pi)
        factorial_k = 1
        for k in range(1,6):
            factorial_k = k*factorial_k
            store = store + ((-1)**k)*(rho**k) / (k*factorial_k)
        euler_mascheroni = 0.5772156649
        kernel_near = (1./(4*np.pi))*( euler_mascheroni \
                        - 2.*np.log(delta) \
                        + store)
    if (m==1) & (n==0):
        kernel_far = DxG(dx_far,dy_far)
        kernel_near = DxG_near(dx_near,dy_near)
    if (m==0) & (n==1):
        kernel_far = DyG(dx_far,dy_far)
        kernel_near = DyG_near(dx_near,dy_near)
    if (m==2) & (n==0):
        kernel_far = DxxG(dx_far,dy_far)
        kernel_near = DxxG_near(dx_near,dy_near)
    if (m==1) & (n==1):
        kernel_far = DxyG(dx_far,dy_far)
        kernel_near = DxyG_near(dx_near,dy_near)
    if (m==0) & (n==2):
        kernel_far = DyyG(dx_far,dy_far)
        kernel_near = DyyG_near(dx_near,dy_near)
    if (m==3) & (n==0):
        kernel_far = DxxxG(dx_far,dy_far)
        kernel_near = DxxxG_near(dx_near,dy_near)
    if (m==2) & (n==1):
        kernel_far = DxxyG(dx_far,dy_far)
        kernel_near = DxxyG_near(dx_near,dy_near)
    if (m==1) & (n==2):
        kernel_far = DxyyG(dx_far,dy_far)
        kernel_near = DxyyG_near(dx_near,dy_near)
    if (m==0) & (n==3):
        kernel_far = DyyyG(dx_far,dy_far)
        kernel_near = DyyyG_near(dx_near,dy_near)
    if (m==4) & (n==0):
        kernel_far = DxxxxG(dx_far,dy_far)
        kernel_near = DxxxxG_near(dx_near,dy_near)
    if (m==3) & (n==1):
        kernel_far = DxxxyG(dx_far,dy_far)
        kernel_near = DxxxyG_near(dx_near,dy_near)
    if (m==2) & (n==2):
        kernel_far = DxxyyG(dx_far,dy_far)
        kernel_near = DxxyyG_near(dx_near,dy_near)
    if (m==1) & (n==3):
        kernel_far = DxyyyG(dx_far,dy_far)
        kernel_near = DxyyyG_near(dx_near,dy_near)
    if (m==0) & (n==4):
        kernel_far = DyyyyG(dx_far,dy_far)
        kernel_near = DyyyyG_near(dx_near,dy_near)
    if (m==5) & (n==0):
        kernel_far = DxxxxxG(dx_far,dy_far)
        kernel_near = DxxxxxG_near(dx_near,dy_near)
    if (m==4) & (n==1):
        kernel_far = DxxxxyG(dx_far,dy_far)
        kernel_near = DxxxxyG_near(dx_near,dy_near)
    if (m==3) & (n==2):
        kernel_far = DxxxyyG(dx_far,dy_far)
        kernel_near = DxxxyyG_near(dx_near,dy_near)
    if (m==2) & (n==3):
        kernel_far = DxxyyyG(dx_far,dy_far)
        kernel_near = DxxyyyG_near(dx_near,dy_near)
    if (m==1) & (n==4):
        kernel_far = DxyyyyG(dx_far,dy_far)
        kernel_near = DxyyyyG_near(dx_near,dy_near)
    if (m==0) & (n==5):
        kernel_far = DyyyyyG(dx_far,dy_far)
        kernel_near = DyyyyyG_near(dx_near,dy_near)
    return far*kernel_far + near*kernel_near

def stream_function( jv , X=None , Y=None , m=0 , n=0 ):
    # Computes \partial^m_x \partial^n_y \psi(X,Y)
    # induced by the jet vortex configuration, jv.
    # Note m + n <= 2 must hold
    if X is None:
        X = jv.x
        Y = jv.y
    store = np.zeros( np.size(X) )
    store += np.dot(kernel(jv, X, Y, m, n), jv.gamma)
    if jv.order > 0:
        store += np.dot(kernel(jv, X, Y, m+1, n), jv.gamma_x)
        store += np.dot(kernel(jv, X, Y, m, n+1), jv.gamma_y)
        if jv.order > 1:
            store += np.dot(kernel(jv, X, Y, m+2, n), jv.gamma_xx)
            store += np.dot(kernel(jv, X, Y, m+1, n+1), jv.gamma_xy)
            store += np.dot(kernel(jv, X, Y, m, n+2), jv.gamma_yy)
    return store

def velocity_field( jv , X=None , Y=None , m=0 , n=0 ):
    # Computes \partial^m_x \partial^n_y \psi(X,Y)
    # induced by the jet vortex configuration, jv.
    # Note m + n <= 2 must hold
    u = stream_function(jv,X,Y,m,n+1)
    v = -stream_function(jv,X,Y,m+1,n)
    return u,v

def get_gamma( psi , x , y , order ):
    # constructs gamma variables from a stream-function and vortex locations
    # psi is a function psi(X,Y,m=0,n=0)
    # outputs a jet-vortex configuration, jv.
    N = np.size(x)
    jv = Jet_vortex( order=order , N = N )
    jv.x = x
    jv.y = y
    Dx = jv.dx()
    Dy = jv.dy()
    G0 = kernel(jv)
    if order == 0:
        b = psi(x,y)
        jv.gamma = np.linalg.solve( G0 , b )
    elif order > 0:
        G_x = kernel(jv,m=1)
        G_y = kernel(jv,n=1)
        G_xx = kernel(jv,m=2)
        G_xy = kernel(jv,m=1,n=1)
        G_yy = kernel(jv,n=2)
        if order == 1:
            matrix = np.zeros( [3*N,3*N] )
        elif order == 2:
            matrix = np.zeros( [6*N,6*N] )
        matrix[0:N,0:N] = G0
        matrix[0:N,N:(2*N)] = G_x
        matrix[0:N,(2*N):(3*N)] = G_y
        matrix[N:(2*N),0:N] = G_x
        matrix[N:(2*N),N:(2*N)] = G_xx
        matrix[N:(2*N),(2*N):(3*N)] = G_xy
        matrix[(2*N):(3*N),0:N] = G_y
        matrix[(2*N):(3*N),N:(2*N)] = G_xy
        matrix[(2*N):(3*N),(2*N):(3*N)] = G_yy
        if order == 2:
            G_xxx = kernel(jv,m=3)
            G_xxy = kernel(jv,m=2,n=1)
            G_xyy = kernel(jv,m=1,n=2)
            G_yyy = kernel(jv,n=3)
            G_xxxx = kernel(jv,m=4)
            G_xxxy = kernel(jv,m=3,n=1)
            G_xxyy = kernel(jv,m=2,n=2)
            G_xyyy = kernel(jv,m=1,n=3)
            G_yyyy = kernel(jv,n=4)
            matrix[0:N,(3*N):(4*N)] = G_xx
            matrix[0:N,(4*N):(5*N)] = G_xy
            matrix[0:N,(5*N):(6*N)] = G_yy
            matrix[N:(2*N),(3*N):(4*N)] = G_xxx
            matrix[N:(2*N),(4*N):(5*N)] = G_xxy
            matrix[N:(2*N),(5*N):(6*N)] = G_xyy
            matrix[(2*N):(3*N),(3*N):(4*N)] = G_xxy
            matrix[(2*N):(3*N),(4*N):(5*N)] = G_xyy
            matrix[(2*N):(3*N),(5*N):(6*N)] = G_yyy
            matrix[(3*N):(4*N),0:N] = G_xx
            matrix[(3*N):(4*N),N:(2*N)] = G_xxx
            matrix[(3*N):(4*N),(2*N):(3*N)] = G_xxy
            matrix[(3*N):(4*N),(3*N):(4*N)] = G_xxxx
            matrix[(3*N):(4*N),(4*N):(5*N)] = G_xxxy
            matrix[(3*N):(4*N),(5*N):(6*N)] = G_xxyy
            matrix[(4*N):(5*N),0:N] = G_xy
            matrix[(4*N):(5*N),N:(2*N)] = G_xxy
            matrix[(4*N):(5*N),(2*N):(3*N)] = G_xyy
            matrix[(4*N):(5*N),(3*N):(4*N)] = G_xxxy
            matrix[(4*N):(5*N),(4*N):(5*N)] = G_xxyy
            matrix[(4*N):(5*N),(5*N):(6*N)] = G_xyyy
            matrix[(5*N):(6*N),0:N] = G_yy
            matrix[(5*N):(6*N),N:(2*N)] = G_xyy
            matrix[(5*N):(6*N),(2*N):(3*N)] = G_yyy
            matrix[(5*N):(6*N),(3*N):(4*N)] = G_xxyy
            matrix[(5*N):(6*N),(4*N):(5*N)] = G_xyyy
            matrix[(5*N):(6*N),(5*N):(6*N)] = G_yyyy            
        if order == 1:
            b = np.zeros(3*N)
        elif order == 2:
            b = np.zeros(6*N)
        b[0:N] = psi(x,y)
        b[N:(2*N)] = psi(x,y,m=1)
        b[(2*N):(3*N)] = psi(x,y,n=1)
        if order == 2:
            b[(3*N):(4*N)] = psi(x,y,m=2)
            b[(4*N):(5*N)] = psi(x,y,m=1,n=1)
            b[(5*N):(6*N)] = psi(x,y,n=2)
        store = np.linalg.solve( matrix , b )
        jv.gamma = store[0:N]
        jv.gamma_x = store[N:(2*N)]
        jv.gamma_y = store[(2*N):(3*N)]
        if order == 2:
            jv.gamma_xx = store[(3*N):(4*N)]
            jv.gamma_xy = store[(4*N):(5*N)]
            jv.gamma_yy = store[(5*N):(6*N)]
    return jv

def ode_func( state , t , pars): #pars = (order,N)
    jv = Jet_vortex( order = pars[0] , N = pars[1] )
    djv = Jet_vortex( order = pars[0] , N = pars[1] )
    jv.use_state(state)
    u,v = velocity_field(jv)
    djv.x = u
    djv.y = v
    if jv.order > 0:
        u_x,v_x = velocity_field(jv,m=1)
        u_y,v_y = velocity_field(jv,n=1)
        djv.gamma_x = jv.gamma_x*u_x + jv.gamma_y*u_y
        djv.dgamma_y = jv.gamma_x*v_x + jv.gamma_y*v_y
        if jv.order ==2:
            u_xx,v_xx = velocity_field(jv,m=2)
            u_xy,v_xy = velocity_field(jv,m=1,n=1)
            u_yy,v_yy = velocity_field(jv,n=2)
            djv.dgamma_xx = 2*jv.gamma_xx*u_x + jv.gamma_xy*u_y
            djv.dgamma_yy = 2*jv.gamma_yy*v_y + jv.gamma_xy*v_x
            djv.dgamma_xy = 2*jv.gamma_xx*v_x + 2*jv.gamma_yy*u_y
            djv.gamma_x += - jv.gamma_xx*u_xx \
                   - jv.gamma_xy*u_xy - jv.gamma_yy*u_yy
            djv.gamma_y += - jv.gamma_xx*v_xx \
                   - jv.gamma_xy*v_xy - jv.gamma_yy*v_yy
    return djv.state()

def energy(jv):
    # computes the conserved energy as a sanity check
    Dx = jv.dx() ; Dy = jv.dy()
    gamma = jv.gamma
    G0 = kernel(jv)
    out = 0.5*np.einsum('i,j,ij',gamma,gamma,G0)
    if jv.order > 0:
        G_x = kernel(jv,m=1)
        G_y = kernel(jv,n=1)
        G_xx = kernel(jv,m=2)
        G_yy = kernel(jv,n=2)
        G_xy = kernel(jv,m=1,n=1)
        gamma_x = jv.gamma_x
        gamma_y = jv.gamma_y
        out -= 0.5*np.einsum('i,j,ij',gamma_x,gamma_x,G_xx)
        out -= np.einsum('i,j,ij',gamma_x,gamma_y,G_xy)
        out -= 0.5*np.einsum('i,j,ij',gamma_y,gamma_y,G_yy)
        out += np.einsum('i,j,ij',gamma,gamma_x,G_x)
        out += np.einsum('i,j,ij',gamma,gamma_y,G_y)
        if jv.order > 1:
            G_xxxx = kernel(jv,m=4)
            G_xxxy = kernel(jv,m=3,n=1)
            G_xxyy = kernel(jv,m=2,n=2)
            G_xyyy = kernel(jv,m=1,n=3)
            G_yyyy = kernel(jv,n=4)
            gamma_xx = jv.gamma_xx 
            gamma_xy = jv.gamma_xy
            gamma_yy = jv.gamma_yy 
            out += np.einsum('i,j,ij',gamma,gamma_xx,G_xx)
            out += np.einsum('i,j,ij',gamma,gamma_xy,G_xy)
            out += np.einsum('i,j,ij',gamma,gamma_yy,G_yy)
            out += 0.5*np.einsum('i,j,ij',gamma_xx,gamma_xx,G_xxxx)
            out += np.einsum('i,j,ij',gamma_xx,gamma_xy,G_xxxy)
            out += 0.5*np.einsum('i,j,ij',gamma_xy,gamma_xy,G_xxyy)
            out += np.einsum('i,j,ij',gamma_xx,gamma_yy,G_xxyy)
            out += np.einsum('i,j,ij',gamma_xy,gamma_yy,G_xyyy)
            out += 0.5*np.einsum('i,j,ij',gamma_yy,gamma_yy,G_yyyy)
    return out

def linear_momentum(jv):
    # Returns the symplectic linear momentum map
    if jv.order == 0:
        return np.sum(np.array([-jv.y*jv.gamma,\
                                jv.x*jv.gamma]))
    else:
        return np.sum(np.array([-jv.y*jv.gamma+jv.gamma_y,\
                                jv.x*jv.gamma-jv.gamma_x]))

def angular_momentum(jv):
    # Returns the symplectic angular momentum map
    store = 0.5*jv.gamma*((jv.x)**2+(jv.y)**2)
    if jv.order > 0:
        store -= jv.gamma_x*jv.x + jv.gamma_y*jv.y
        if jv.order > 1:
            store += jv.gamma_xx + jv.gamma_yy
    return store.sum()
