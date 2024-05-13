import torch
import torch.nn as nn
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy.fft import fft, rfft
from scipy.integrate import odeint
from scipy.signal import argrelextrema
from scipy.signal import correlate
from scipy.stats import wasserstein_distance
import seaborn as sns

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
sys.path.append('..')

from dyn_sys.dim1 import *
from dyn_sys.dim2 import *
from dyn_sys.dim3 import *
from dyn_sys.dim4 import *
from dyn_sys.KS import *


class ODE_MLP(nn.Module):
    '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''

    def __init__(self, y_dim=3, n_hidden=512, n_layers=5):
        super(ODE_MLP, self).__init__()
        layers = [nn.Linear(y_dim, n_hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.GELU()])
        layers.append(nn.Linear(n_hidden, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        res = self.net(y)
        return res
    

class ODE_MLP_skip(nn.Module):
    def __init__(self, y_dim=3, n_hidden=512, n_layers=5):
        super(ODE_MLP_skip, self).__init__()
        layers = [nn.Linear(y_dim, n_hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.skip = nn.Sequential(
            nn.Linear(y_dim, n_hidden),
            nn.ReLU(),
        )
        self.output = nn.Linear(n_hidden, y_dim)
    
    def forward(self, t, y):
        res = self.net(y) + self.skip(y)
        return self.output(res)   


#################
##### Train #####
#################


def create_data(dyn_info, n_train, n_test, n_val, n_trans):
    dyn, dim, time_step = dyn_info
    # Adjust total time to account for the validation set
    tot_time = time_step * (n_train + n_test + n_val + n_trans + 1)
    t_eval_point = torch.arange(0, tot_time, time_step)

    # Generate trajectory using the dynamical system
    traj = torchdiffeq.odeint(dyn, torch.randn(dim), t_eval_point, method='rk4', rtol=1e-8)
    traj = traj[n_trans:]  # Discard transient part

    # Create training dataset
    X_train = traj[:n_train]
    Y_train = traj[1:n_train + 1]

    # Shift trajectory for validation dataset
    traj = traj[n_train:]
    X_val = traj[:n_val]
    Y_val = traj[1:n_val + 1]

    # Shift trajectory for test dataset
    traj = traj[n_val:]
    X_test = traj[:n_test]
    Y_test = traj[1:n_test + 1]

    return [X_train, Y_train, X_val, Y_val, X_test, Y_test]


def define_dyn_sys(dyn_sys):
    DYNSYS_MAP = {'tilted_tent_map' : [tilted_tent_map, 1],
                  'pinched_tent_map' : [pinched_tent_map, 1],
                  'plucked_tent_map' : [plucked_tent_map, 1],
                  'KS': [run_KS, 127],
                  'baker' : [baker, 2],
                  'lorenz' : [lorenz, 3],
                  'rossler' : [rossler, 3],
                  'hyperchaos': [hyperchaos, 4]}
    dyn_sys_info = DYNSYS_MAP[dyn_sys]
    dyn_sys_func, dim = dyn_sys_info
    return dyn_sys_func, dim

#################
### Vectorize ###
#################

def one_step_rk4(f, y0, t):
    h = t[1] - t[0]
    k1 = f(t, y0)
    k2 = f(t + h/2, y0 + k1 * h / 2.)
    k3 = f(t + h/2, y0 + k2 * h / 2.)
    k4 = f(t + h, y0 + k3 * h)
    new_y = y0 + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return new_y

def vectorized_simulate(model, X, t_eval_point, len_T, device):
    torch.cuda.empty_cache()

    integrated_model = lambda x: one_step_rk4(model, x, t_eval_point).to(device)
    compute_batch = torch.func.vmap(integrated_model, in_dims=(0), chunk_size=5000)
    
    traj = torch.zeros(len_T, X.shape[0], X.shape[1]) # len_T x num_init x dim
    traj[0] = X
    for i in range(1, len_T):
        if i % 1000 == 0: print(i)
        traj[i] = compute_batch(X.to(device)).detach() 
        X = traj[i]
    return traj

def vectorized_simulate_map(model, X, t_eval_point, len_T, device):
    torch.cuda.empty_cache()

    integrated_model = lambda x: model(x).to(device)
    compute_batch = torch.func.vmap(integrated_model, in_dims=(0), chunk_size=3000)
    
    traj = torch.zeros(len_T, X.shape[0], X.shape[1]) # len_T x num_init x dim
    traj[0] = X
    for i in range(1, len_T):
        print(i)
        traj[i] = compute_batch(X.double().to(device)).detach() 
        X = traj[i]
    return traj


#################
##### Metric ####
#################


def auto_corr(traj, tau, ind_func, dt, len_integration):
    # Iterate over from 0 ... tau-1
    # traj_mean_sq = np.dot(traj, traj)/((tau*int(1/dt))**2)
    num_corr_ele = tau*int(1/dt)-len_integration
    corr = np.zeros(num_corr_ele)
    for j in range(num_corr_ele):
        # # z(0 + Tau: t + Tau)
        base_traj = traj[:len_integration]
        tau_traj = traj[j: j+len_integration]
        traj_mean_sq = np.mean(tau_traj)*np.mean(base_traj)

        # # compute corr between
        corr[j] = np.dot(tau_traj,base_traj)/(len_integration) - traj_mean_sq
    return np.abs(corr)

def rk4_KS(f, y0, t):
    h = t[1] - t[0]
    k1 = f(t, y0)
    k2 = f(t + h/2, y0 + k1 * h / 2.)
    k3 = f(t + h/2, y0 + k2 * h / 2.)
    k4 = f(t + h, y0 + k3 * h)
    new_y = y0 + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return new_y

def rk4(x, f, dt):
    k1 = f(0, x)
    k2 = f(0, x + dt*k1/2)
    k3 = f(0, x + dt*k2/2)
    k4 = f(0, x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
def lyap_exps(dyn_sys_info, s, traj, iters):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dyn_type, model, dim, time_step = dyn_sys_info
    LE = torch.zeros(dim).to(device)
    traj_gpu = traj.to(device)

    if dim in (1,2):
        le = 0
        for t in range(traj_gpu.shape[0]):
            if (model == tilted_tent_map) or (model == plucked_tent_map) or (model == pinched_tent_map) or (model == baker):
                le += torch.log(abs(F.jacobian(lambda x: model(x, s), traj_gpu[t]))) # true model
            else:
                le += torch.log(abs(F.jacobian(lambda x: model(0, x), traj_gpu[t]))) # learned model
        return le/traj_gpu.shape[0]
    else:
        if dyn_type == "KS":
            lower_dim = 15
            Q = torch.eye(*(dim, lower_dim)).to(device).double()
            # True
            if model == run_KS:
                L = 128 #128 # n = [128, 256, 512, 700]
                n = L-1 # num of internal node
                dx = 1
                dt = time_step
                c = 0.4
                lyap_exp = []
                U = torch.eye(*(dim, lower_dim)).double()
                for i in range(iters):
                    if i % 1000 == 0: print("rk4", i) 
                    x0 = traj_gpu[i].requires_grad_(True)
                    cur_J = F.jacobian(lambda x: run_KS(x, c, dx, dt, dt*2, False, device), x0, vectorize=True)[-1]
                    J = torch.matmul(cur_J.to(device).double(), U.to(device).double())
                    Q, R = torch.linalg.qr(J)
                    lyap_exp.append(torch.log(abs(R.diagonal())))
                    U = Q.double() #new axes after iteration
                lyap_exp = torch.stack(lyap_exp).detach().cpu().numpy()
                LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (dt*iters) for j in range(lower_dim)]
                return torch.tensor(LE)
            # Learned
            else:
                ts = torch.linspace(0, time_step, 2)
                f = lambda x: rk4_KS(model, x, ts)
                Jac = torch.vmap(torch.func.jacrev(f), chunk_size=5)(traj_gpu)
            LE = torch.zeros(lower_dim).to(device)
        else:
            f = lambda x: rk4(x, model, time_step)
            Jac = torch.vmap(torch.func.jacrev(f))(traj_gpu)
            Q = torch.rand(dim,dim).to(device)
        for i in range(iters):
            if i > 0 and i % 1000 == 0:
                print("Iteration: ", i, ", LE[0]: ", LE[0].detach().cpu().numpy()/i/time_step)
            Q = torch.matmul(Jac[i], Q)
            Q, R = torch.linalg.qr(Q)
            LE += torch.log(abs(torch.diag(R)))
        return LE/iters/time_step