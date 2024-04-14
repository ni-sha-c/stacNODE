import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.append('../examples')
from KS import *

# from torch import *

def test_KuramotoSivashinsky():
    '''
    state vector, u, has following index i

    i = 0, 1, 2, ..., n, n+1
    where i = 1, ..., n are internal nodes
          i = 0, n+1 are boundary nodes
          i = -1, n+2 are ghost nodes
    '''

    L = 128 # signal from [0, L]
    n = 127 # n = number of interior nodes: 127, 511
    dx = L/(n+1) # 0.25
    c = 0.4
    x = torch.arange(0, L+dx, dx) # [0, 0+dx, ... 128] shape: L + 1
    # u = -0.5 + torch.rand(n+2)

    u = x**4  
    u[0], u[-1] = 0, 0 # u_0, u_n = 0, 0

    # print((7*u[n] - 4*u[n-1] + u[n-2])/(dx**4))
    up = 4*x**3
    upup = 12*x**2
    upupup = 24*x**1


    # --- ana_rhs_KS: -(u + c)*up - upup - upupupup --- #
    ana_rhs_KS = -upup - 24.0*torch.ones_like(x)
    # ana_rhs_KS = 24.0*torch.ones_like(x)
    # --- num_rhs_KS: rhs_KS(u, c, dx) --- #
    num_rhs_KS = torch.matmul(rhs_KS_implicit(u, dx), u)
    # num_rhs_KS = rhs_KS_explicit_nl(u, c,dx) # rhs_KS_explicit_linear(u, c, dx) #rhs_KS_explicit_nl(u, c,dx) + 
 
    # Testing for inner nodes
    print("answer", ana_rhs_KS[:10], ana_rhs_KS[-10:])
    print("predicted", num_rhs_KS[:10], num_rhs_KS[-10:])

    print(torch.norm(ana_rhs_KS[1:-1]))
    print(torch.norm(num_rhs_KS[1:-1]))
    
    return u, num_rhs_KS, ana_rhs_KS


def KS_FD_Simulate_ensemble(c, L, T, N):
    '''
    L: the last boundary node of signal [0, L]
    N : number of realization
    '''

    n = L - 1 # n = number of interior nodes
    dx = L/(n+1) # 0.25
    dt = 0.25 # 0.1
    x = torch.arange(0, L+dx, dx) # [0, 0+dx, ... 128] shape: L + 1
    u0 = torch.zeros(N, n+2)
    u = torch.zeros(N, n+2)
    realization = torch.zeros(N)

    for i in torch.arange(0, N):
        # initial signal of 'i'th realization

        # u0[i, :] = -0.5 + torch.rand(n+2)
        u0[i, :] = torch.tensor(2.71828**(-(x-64)**2/512), requires_grad=True)
        
        # boundary condition
        u0[i, 0], u0[i, -1] = 0, 0 

        u[i,:] = u0[i]
        u1, time_avg = run_KS_timeavg(u[i,:], c, dx, dt, T, mean=True)
        print(u1)
        u1.sum().backward()
        print("u1", u1.grad)
        u[i,:] = u1
        realization[i] = time_avg
        print(i, time_avg)

    return x, u0, u, realization






if __name__ == '__main__':

    # test_KuramotoSivashinsky()

    fig, ax = plt.subplots()

    C = torch.arange(0, 2.1, 0.1)
    n = torch.tensor([256]) # n = [128, 256, 512, 700]
    T = 500
    num_samples = 1
    
    result = torch.zeros(C.shape[0], 1)
    r_mean = torch.zeros(n.shape[0], C.shape[0])

    for i in torch.arange(0, n.shape[0]):
        print("n", n[i])
        for c in torch.arange(0, C.shape[0]):
            print("c", C[c])
            x, u, u1, realization = KS_FD_Simulate_ensemble(C[c], n[i], T, num_samples)
            r_mean[i, c] = torch.mean(realization)
            x_axis = torch.ones(realization.shape[0]) * C[c]
            ax.scatter(x_axis, realization, c=x_axis*int(i), cmap='viridis')
        ax.plot(C, np.array(r_mean[i, :].detach().cpu()), label = 'n = ' + str(n.item()))

    ax.legend(fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    path = '../plot/KS_validate.png'
    fig.savefig(path, format='png', dpi=400)







    '''L = 128
    n = 511 # number of interior nodes
    c = 0.
    dx = L/(n+1) # 0.25
    x = torch.arange(0, L+dx, dx) # 0, 0 + dx, ... 128 # shape: L + 1
    dt = 0.1
    T = torch.arange(0, dt*5, dt) # 300

    u = sin(2*pi*x/L) # only the internal nodes
    # u_next_exp = explicit_rk(u, c, dx, dt)
    # u_next_imp = implicit_rk(u, c, dx, dt)
    # u_next = u_next_exp + u_next_imp
    u_bar = []
    
    for i in T:
        print(i, u)
        u_next_exp = explicit_rk(u, c, dx, dt)
        u_next_imp = implicit_rk(u, c, dx, dt)
        u_next = u + u_next_exp + u_next_imp
        u = u_next
        u_bar.append(torch.mean(u_next))

    print(u_bar)'''


