import ctypes
import itertools
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import multiprocessing
import numpy as np
import scipy
import torch
from scipy.fft import fft, rfft
from scipy.integrate import odeint
from scipy.signal import argrelextrema
from scipy.signal import correlate
from scipy.stats import wasserstein_distance
from test_metrics import *
import seaborn as sns

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from src.NODE_solve import *
from src.NODE import *
from examples.Brusselator import *
from examples.Lorenz import *
from examples.Lorenz_periodic import *
from examples.Sin import *
from examples.Tilted_tent_map import *
from examples.Pinched_tent_map import *
from examples.Plucked_tent_map import *

def vectorized_simulate(model, X, t_eval_point, len_T, device):
    torch.cuda.empty_cache()

    integrated_model = lambda x: one_step_rk4(model, x, t_eval_point).to(device)
    compute_batch = torch.func.vmap(integrated_model, in_dims=(0), chunk_size=1000)
    
    traj = torch.zeros(len_T, X.shape[0], X.shape[1]) # len_T x num_init x dim
    traj[0] = X
    for i in range(1, len_T):
        print(i)
        traj[i] = compute_batch(X.double().to(device)).detach() 
        X = traj[i]
    return traj


def vectorized_simulate_map(model, X, t_eval_point, len_T, device):
    torch.cuda.empty_cache()

    integrated_model = lambda x: model(x).to(device)
    compute_batch = torch.func.vmap(integrated_model, in_dims=(0), chunk_size=1000)
    
    traj = torch.zeros(len_T, X.shape[0], X.shape[1]) # len_T x num_init x dim
    traj[0] = X
    for i in range(1, len_T):
        print(i)
        traj[i] = compute_batch(X.double().to(device)).detach() 
        X = traj[i]
    return traj



if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. define system
    dyn_sys= "tilted_tent_map"
    dyn_sys_f, dim = define_dyn_sys(dyn_sys)
    time_step= 0.01
    len_T = 1000*int(1/time_step)
    ind_func = 0
    s = 0.2

    # 2. define num init points
    N = 2000
    if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map") or (dyn_sys == "baker"):
        inits = torch.rand(N, dim).double().to(device)
    else:
        inits = torch.randn(N, dim).double().to(device)

    # 3. call models
    if (dyn_sys == "baker"):
        MSE_path = "../test_result/expt_"+str(dyn_sys)+"/" + str(s)+"(MSE)/model.pt"
        JAC_path = "../test_result/expt_"+str(dyn_sys)+"/" + str(s)+"(JAC)/model.pt"
    elif (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        MSE_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+ str(s)+"(MSE)/model.pt"
        JAC_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+ str(s)+"(JAC)/model.pt"
    else:
        MSE_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+'MSE_0/model.pt'
        JAC_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+'JAC_0/model.pt'

    MSE = create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
    JAC = create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
    MSE.load_state_dict(torch.load(MSE_path))
    JAC.load_state_dict(torch.load(JAC_path))
    MSE.eval()
    JAC.eval()

    # 4. generate 3 trajectories
    one_step = torch.linspace(0, time_step, 2).to(device)

    if (dyn_sys == "henon") or (dyn_sys == "baker") or (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        true_traj = torch.zeros(len_T, inits.shape[0], inits.shape[1])

        for j in range(inits.shape[0]):
            print("j: ", j)
            if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
                x = torch.abs(inits[j][0])
            else:
                x = inits[j]
            
            for i in range(len_T):
                next_x = dyn_sys_f(x)
                true_traj[i, j] = next_x
                x = next_x
        MSE_traj = vectorized_simulate_map(MSE, inits, one_step, len_T, device)
        JAC_traj = vectorized_simulate_map(JAC, inits, one_step, len_T, device)
    else:
        true_traj = vectorized_simulate(dyn_sys_f, inits, one_step, len_T, device)
        MSE_traj = vectorized_simulate(MSE, inits, one_step, len_T, device)
        JAC_traj = vectorized_simulate(JAC, inits, one_step, len_T, device)
    print(JAC_traj.shape)

    # 4-1. Remove exploding traj
    MSE_traj = np.asarray(MSE_traj)
    mask = MSE_traj < 10**4  # np.max(np.array(JAC_traj)[:, :, ind_func]) + 5
    if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map") or (dyn_sys == "baker"):
        mask = MSE_traj < 10**1
    row_sums = np.sum(mask, axis=0)

    columns_with_all_true = np.where(row_sums[:, ind_func] == mask.shape[0])
    valid_col = np.unique(columns_with_all_true[0])
    MSE_traj_cleaned = MSE_traj[:, valid_col, :]
    print("MSE cleaned", MSE_traj_cleaned.shape)

    if (dyn_sys == "baker"):
        # 4-1. Remove exploding traj
        JAC_traj = np.asarray(JAC_traj)
        mask = JAC_traj < 10**1  # np.max(np.array(JAC_traj)[:, :, ind_func]) + 5
        row_sums = np.sum(mask, axis=0)
        # print("row", row_sums, row_sums.shape)

        columns_with_all_true = np.where(row_sums[:, ind_func] == mask.shape[0])
        valid_col = np.unique(columns_with_all_true[0])
        JAC_traj = JAC_traj[:, valid_col, :]
        print("JAC cleaned", JAC_traj.shape)


    # 5. indicator function
    # len_T x num_init x dim
    true_avg_traj = np.mean(true_traj[:, :, ind_func].detach().cpu().numpy(), axis=1)
    MSE_avg_traj = np.mean(MSE_traj_cleaned[:, :, ind_func], axis=1)
    JAC_avg_traj = np.mean(JAC_traj[:, :, ind_func].detach().cpu().numpy(), axis=1)
    print("avg traj shape:", JAC_avg_traj.shape)
    print(true_traj[:, :, ind_func].detach().cpu().numpy())

    # 6. plot dist
    pdf_path = '../plot/dist_'+str(dyn_sys)+'_all'+'_'+str(N)+'_'+str(len_T)+'.jpg'

    fig, ax1 = subplots(1,figsize=(16,8)) #, sharey=True
    sns.set_theme()

    lorenz_range_x = [-1.5, 2.5]
    lorenz_range_z = [5, 25]

  
    if str(dyn_sys) == "lorenz":     # lorenz (before -> 2)
        if ind_func == 0:
            kwargs = dict(hist_kws={'alpha':.5, 'range':lorenz_range_x}, kde_kws={'linewidth':3})  

            ax = sns.distplot(JAC_avg_traj, bins=200, color="slateblue", hist=True,   **kwargs)
            ax1 = sns.distplot(true_avg_traj, bins=200, color="salmon", hist=True, **kwargs)
            ax2 = sns.distplot(MSE_avg_traj, bins=200, color="turquoise", hist=True, **kwargs) #histtype='step', linewidth=2., 
            ax.set_xlim(-1.5, 2.)
            ax1.set_xlim(-1.5, 2.)
            ax2.set_xlim(-1.5, 2.)
        elif ind_func == 2:
            kwargs = dict(hist_kws={'alpha':.7, 'range':lorenz_range_z}, kde_kws={'linewidth':3})  

            sns.displot(JAC_avg_traj, bins=300, color="slateblue", density=True, histtype='step', linewidth=3., range=lorenz_range_z)
            sns.distplot(true_avg_traj, bins=300, color="salmon", density=True, histtype='step', linewidth=3., range=lorenz_range_z)
            sns.displot(MSE_avg_traj, bins=300, color="turquoise", density=True, histtype='step', linewidth=3., range=lorenz_range_z)
        elif ind_func == 1:
            sns.displot(JAC_avg_traj, bins=200, color="slateblue", density=True, histtype='step', linewidth=2., range=lorenz_range_x) #range=lorenz_range_x
            sns.displot(true_avg_traj, bins=200, color="salmon", density=True, histtype='step', linewidth=2., range=lorenz_range_x)
            sns.displot(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2., range=lorenz_range_x)
    elif str(dyn_sys) == "rossler": 
        sns.displot(JAC_avg_traj, bins=200, color="slateblue", density=True, histtype='step', linewidth=2.)
        sns.displot(true_avg_traj, bins=200, color="salmon", density=True, histtype='step', linewidth=2.)
        sns.displot(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2.)
    elif str(dyn_sys) == "hyperchaos":
        ax1.hist(JAC_avg_traj, bins=300, color="slateblue", density=True, histtype='step', linewidth=2.)
        ax1.hist(true_avg_traj, bins=300, color="salmon", density=True, histtype='step', linewidth=2.)
        ax1.hist(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2.)
    elif (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        ax1.hist(JAC_avg_traj, bins=200, color="slateblue", density=True, histtype='step', linewidth=2.)
        ax1.hist(true_avg_traj, bins=200, color="salmon", density=True, histtype='step', linewidth=2.)
        ax1.hist(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2.)
    elif str(dyn_sys) == "baker":
        ax1.hist(JAC_avg_traj, bins=200, color="slateblue", density=True, histtype='step', linewidth=2.)
        ax1.hist(true_avg_traj, bins=200, color="salmon", density=True, histtype='step', linewidth=2.)
        ax1.hist(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2.)


    ax1.grid(True)
    ax1.legend(['JAC', 'True', 'MSE'], fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=34)
    ax1.yaxis.set_tick_params(labelsize=34)
    tight_layout()
    savefig(pdf_path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)