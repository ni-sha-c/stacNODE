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