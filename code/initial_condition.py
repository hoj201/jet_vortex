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
