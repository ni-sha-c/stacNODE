import torch
import torch.sparse as tosp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D


def rhs_KS_implicit(u, dx, device):

    n = u.shape[0]     # u contains boundary nodes i = 0, 1, 2, ... , n, n+1

    # ----- second derivative ----- #
    A = tosp.spdiags(torch.vstack((torch.ones(n), -2*torch.ones(n), torch.ones(n)))/(dx*dx), torch.tensor([-1, 0, 1]), (n, n)).to(device)
    A = A.to_dense().double()
    A[0], A[-1], A[:, 0], A[:, -1] = 0, 0, 0, 0 

    # ----- fourth derivative ----- #


    dx4 = dx*dx*dx*dx
    B = tosp.spdiags(torch.vstack((torch.ones(n), -4*torch.ones(n), 6*torch.ones(n), -4*torch.ones(n), torch.ones(n)))/dx4, torch.tensor([-2, -1, 0, 1, 2]), (n-2, n-2)).to(device)
    B = B.to_dense().double()

    # Create the pad 
    C = torch.zeros(n, n).to(device).double()
    C[1:n-1, 1:n-1] = B

    # Boundary Condition (i = 2, 3, ... , n-1)
    C[1, 1] = 7/dx4
    C[1, 2] = -4/dx4
    C[1, 3] = 1/dx4
    C[-2, -2] = 7/dx4
    C[-2, -3] = -4/dx4
    C[-2, -4] = 1/dx4

    return -(A+C)


def rhs_KS_explicit_nl(u, c, dx, device):
    # u contains boundary nodes
    n = u.shape[0]

    B = tosp.spdiags(torch.vstack((torch.ones(n), -torch.ones(n)))/(2*dx), torch.tensor([1,-1]), (n, n)).to(device)
    B = B.to_dense().double()
    B[0], B[-1] = 0., 0. # du_0/dx = 0, du_n/dx = 0
    
    exp_term = - torch.matmul(B, u*u)/2
    # exp_term[0], exp_term[-1] = 0., 0.

    return exp_term


def rhs_KS_explicit_linear(u, c, dx, device):
    # u contains boundary nodes
    n = u.shape[0]

    B = tosp.spdiags(torch.vstack((torch.ones(n), -torch.ones(n)))/(2*dx), torch.tensor([1,-1]), (n, n)).to(device)
    B = B.to_dense().double()

    # du_0/dx = 0, du_n/dx = 0
    B[0], B[-1] = 0., 0.

    exp_term = - torch.matmul(B, u)*c
 
    return exp_term

def explicit_rk(u, c, dx, dt, device):
    k1 = rhs_KS_explicit_nl(u, c, dx, device) + rhs_KS_explicit_linear(u, c, dx, device)
    k2 = rhs_KS_explicit_nl(u + dt/3*k1, c, dx, device) + rhs_KS_explicit_linear(u + dt/3*k1, c, dx, device)
    k3 = rhs_KS_explicit_nl(u + dt*k2, c, dx, device) + rhs_KS_explicit_linear(u + dt*k2, c, dx, device)
    k4 = rhs_KS_explicit_nl(u + dt*(0.75*k2 + 0.25*k3), c, dx, device) + rhs_KS_explicit_linear(u + dt*(0.75*k2 + 0.25*k3), c, dx, device)
    return dt*(3/4*k2 - 1/4*k3 + 1/2*k4)

def implicit_rk(u, c, dx, dt, device):
    n = u.shape[0]
    A = rhs_KS_implicit(u, dx, device)
    Au = torch.matmul(A, u)
    k2 = torch.linalg.solve(torch.eye(n).to(device) - dt/3*A, Au)
    k3 = torch.linalg.solve(torch.eye(n).to(device) - dt/2*A, Au + dt/2*torch.matmul(A, k2))
    k4 = torch.linalg.solve(torch.eye(n).to(device) - dt/2*A, Au + dt/4*torch.matmul(A, 3*k2-k3))
    return dt * (3/4*k2 - 1/4*k3 + 1/2*k4)


def plot_KS(u_list, dx, n, c, T, dt, train, test, loss_type):
    # plot the result
    if torch.is_tensor(u_list):
        u_list = np.array(u_list.detach().cpu())
    else:
         u_list = np.array(u_list)
    # print("shape", u_list.shape)

    fig, ax = plt.subplots(figsize=(12,12))
    x = np.arange(0, n, dx)
    t = np.arange(0, T, dt)

    xx, tt = np.meshgrid(x, t)
    levels = np.arange(-4, 4, 0.01)
    cs = ax.contourf(xx, tt, u_list, cmap=cm.jet)
    cbar = fig.colorbar(cs)
    cbar.ax.tick_params(labelsize=34)

    ax.set_xlabel("X", fontsize=35)
    ax.set_ylabel("T", fontsize=35)
    ax.xaxis.set_tick_params(labelsize=34)
    ax.yaxis.set_tick_params(labelsize=34)
    plt.tight_layout()

    # ax.set_title(f"Kuramoto-Sivashinsky: L = {n-1}")
    if train == True:
        fig.savefig('../plot/Phase_plot/KS/true_train_'+str(loss_type)+'{0:.1f}.png'.format(c))
    elif test == True:
        fig.savefig('../plot/Phase_plot/KS/pred_train_'+str(loss_type)+'{0:.1f}.png'.format(c))
    elif (train == False) and (test == False):
        None
    else:
        fig.savefig('../plot/KS/KS_'+str(loss_type)+'{0:.1f}.png'.format(c))
    return


def run_KS(u, c, dx, dt, T, mean, device):
    ''' plot time averages of spatial average of u for a set of c '''

    # u contains boundary nodes
    n = u.shape[0]
    t = 0.
    spatial_avg = 0
    # spatial_avg = torch.zeros(u.shape)
    denominator = 0
    time_avg = 0
    u_list = []
    while t < T:
        if t % 10 == 0:
            print("run_KS", t)
        u = u + explicit_rk(u, c, dx, dt, device) + implicit_rk(u, c, dx, dt, device) 
        u[0], u[-1] = 0., 0.
        u_list.append(u.clone())
        t += dt

    if torch.is_tensor(u):
        u_list = torch.stack(u_list)
    elif isinstance(u, list):
        print('list')
        list_tensors = [torch.tensor(np.array(u)) for u in u_list]
        u_list = torch.stack(list_tensors) # shape:  time x x_grid
    elif isinstance(variable, np.ndarray):
        list_tensors = [torch.tensor(u) for u in u_list]
        u_list = torch.stack(list_tensors) # shape:  time x x_grid
    else:
        print("The variable has an unknown type.")
    
    return u_list

def run_KS_timeavg(u, c, dx, dt, T, mean):
    ''' plot time averages of spatial average of u for a set of c '''

    # u contains boundary nodes
    n = u.shape[0]
    t = 0.
    spatial_avg = 0
    # spatial_avg = torch.zeros(u.shape)
    denominator = 0
    time_avg = 0
    u_list = []
    while t < T:
        u = u + explicit_rk(u, c, dx, dt) + implicit_rk(u, c, dx, dt) 
        u[0], u[-1] = 0., 0.
        u_list.append(u)
        t += dt

        if mean == True:
            if t >= (T - 200 + dt):
                denominator += 1
                spatial_avg += torch.mean(u)
                # spatial_avg += u
                # print(u)
                # print(torch.mean(u))
            # print("At time", t, " Max u", torch.max(u))
            # print("Min u", torch.min(u))'''


    # plot_KS(u_list, dx, n, c, T, dt)

    if mean == True:
        # time_avg = torch.mean(spatial_avg) / denominator
        time_avg = spatial_avg / denominator
    
    if torch.is_tensor(u):
        u_list = torch.stack(u_list)
        print('tensor')
    elif isinstance(u, list):
        print('list')
        list_tensors = [torch.tensor(np.array(u)) for u in u_list]
        u_list = torch.stack(list_tensors) # shape:  time x x_grid
    elif isinstance(variable, np.ndarray):
        list_tensors = [torch.tensor(u) for u in u_list]
        u_list = torch.stack(list_tensors) # shape:  time x x_grid
    else:
        print("The variable has an unknown type.")
    
    # return u_list
    return u_list, time_avg
    
