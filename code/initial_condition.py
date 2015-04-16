import matplotlib.pyplot as plt
import numpy as np
import jet_vortex_lib as jvl

N=2
order = 1

def get_ic():
    jv = jvl.Jet_vortex(order=order,N=N)
    jv.x[0] = 10.0
    jv.y[0] = 0.5
    jv.x[1] = 0.0
    jv.y[1] = -0.5
    jv.gamma[0] = 0.0
    jv.gamma[1] = 1.0
    jv.gamma_x[0] = 0.0
    jv.gamma_y[0] = 1.0
    return jv

#  THE FOLLOWING BLOCK OF CODE IS TO MANUALLY DESCRIBE THE INITIAL CONDITION
def get_random_ic():
    jv = jvl.Jet_vortex(order=order,N=N)
    for i in range(0,N):
        jv.gamma[i] = 0.5 + np.random.randn()
        jv.x[i] = np.random.randn()
        jv.y[i] = np.random.randn()
        if order > 0:
            jv.gamma_x[i] = 0.1*np.random.randn()
            jv.gamma_y[i] = 0.1*np.random.randn()
    return jv

def get_ic_from_keyboard():
    N = input("How many vortices would you like to see?\n")
    while N not in {1,2,3,4,5}:
        N = input("please answer with 1, 2, or 3.")
    order = input("order = ?\n")
    while order not in {0,1,2}:
        order = input("please answer with 0, 1, or 2.")
    jv = jvl.Jet_vortex(order=order,N=N)
    for i in range(0,N):
        jv.x[i] = input("x[%d] = " % i)
        jv.y[i] = input("y[%d] = " % i)
        jv.gamma[i] = input("gamma[%d] = " % i)
        if jv.order > 0:
            jv.gamma_x[i] = input("gamma_x[%d] = " % i)
            jv.gamma_y[i] = input("gamma_y[%d] = " % i)
            if jv.order > 1:
                jv.gamma_xx[i] = input("gamma_xx[%d] = " % i)
                jv.gamma_xy[i] = input("gamma_xy[%d] = " % i)
                jv.gamma_yy[i] = input("gamma_yy[%d] = " % i)
    return jv

def psi_0(x,y,m=0,n=0):
    k=3.0
    g = np.exp( -(x**2+y**2)/2.)
    if m==0:
        if n==0:
            return np.sin(k*x)*g
        elif n==1:
            return -y*np.sin(k*x)*g
        elif n==2:
            return -np.sin(k*x)*g+(y**2)*np.sin(k*x)*g
    elif m==1:
        if n==0:
            return (-x*np.sin(k*x) + k*np.cos(k*x))*g
        elif n==1:
            return (x*y*np.sin(k*x) - k*y*np.cos(k*x))*g
    elif m==2:
        return (-np.sin(k*x)-x*k*np.cos(k*x)-k*k*np.sin(k*x)\
                +(x**2)*np.sin(k*x) - k*x*np.cos(k*x))*g

def psi_1(x,y,m=0,n=0):
    var = 5.
    return -0.3*Gaussian(x+var,y,sigma=var,m=m,n=n)\
        + 0.5*Gaussian(x,y,m=m,n=n,sigma=var)\
        - 0.3*Gaussian(x-var,y+0.1,sigma=var,m=m,n=n)

def psi_2(x,y,m=0,n=0):
    s = 5.
    return 0.5*Gaussian(x,y,m=m+1,n=n,sigma=s) +\
        0.2*Gaussian(x,y,m=m,n=n+1,sigma=s) +\
        0.5*Gaussian(x,y,m=m,n=n,sigma=0.8*s)

# THE FOLLOWING BLOCK OF CODE IS TO DETERMINE JET-VORTICES USING
# ONE OF THE STREAM FUNCTIONS IN THE PREAMBLE
def get_ic_from_psi(psi):
    rt_N = 10
    N = rt_N**2
    h = (x_max-x_min)/rt_N
    order = input("What order of jet-vortex?\n")
    while order not in {0,1,2}:
        print 'Order must be 0 , 1 , or 2.'
        order = input("What order of jet-vortex?\n")
    print 'order = ' + str(order)
    print 'N = ' + str(N)
    s = np.linspace(x_min,x_max,rt_N)
    x_mesh,y_mesh = np.meshgrid(s,s)
    x = x_mesh.flatten()
    y = y_mesh.flatten()

    psi_wild = psi_1
    jv = jvl.get_gamma( psi_wild , x , y , order)
    state_0 = jv.state()

    max_gamma  = np.max(np.abs(jv.gamma))
    if max_gamma > 10.0:
        print 'max(gamma) = ' + str(max_gamma) + '!  Abort!'
        exit()

    psi = np.zeros([res,res])
    psi_flat = jvl.stream_function(jv,X=x_grid.flatten()\
                                   ,Y=y_grid.flatten())
    psi_exact = psi_wild(x_grid.flatten(),y_grid.flatten())
    levels = np.linspace( psi_flat.min() , psi_flat.max() ,30)
    print "Plotting initial stream_function"
    plt.contourf(x_grid,y_grid,psi_flat.reshape([res,res]),levels)
    plt.axis('equal')
    plt.show()
    print "max error = " + str(np.max(np.abs(psi_flat - psi_exact)))
    string = raw_input("Continue? (y/n):")
    while string not in {'y','n'}:
        string = raw_input("please answer with \'y\' or \'n\'.  Continue?")

    if string is 'n':
        exit()
    else:
        return jv
