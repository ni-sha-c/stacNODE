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
from examples.Tent_map import *

def vectorized_simulate(model, X, t_eval_point, len_T, device):
    torch.cuda.empty_cache()

    integrated_model = lambda x: one_step_rk4(model, x, t_eval_point).to(device)
    compute_batch = torch.func.vmap(integrated_model, in_dims=(0), chunk_size=2000)
    
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
    compute_batch = torch.func.vmap(integrated_model, in_dims=(0), chunk_size=3000)
    
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
    dyn_sys= "lorenz"
    dyn_sys_f, dim = define_dyn_sys(dyn_sys)
    time_step= 0.01
    len_T = 50*int(1/time_step)

    # 2. define num init points
    N = 1000
    inits = torch.randn(N, dim).double().to(device)
    every = 10
    ind_func = 0

    # 3. call models
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

    true_traj = vectorized_simulate(dyn_sys_f, inits, one_step, len_T, device)
    MSE_traj = vectorized_simulate(MSE, inits, one_step, len_T, device)
    JAC_traj = vectorized_simulate(JAC, inits, one_step, len_T, device)
    print(JAC_traj.shape)
    print(MSE_traj.shape)
    print(true_traj.shape)

    # 4-1. Remove exploding traj
    MSE_traj = np.asarray(MSE_traj)
    # mask = 10**-4 < MSE_traj < 10**4  # np.max(np.array(JAC_traj)[:, :, ind_func]) + 5
    mask = MSE_traj < 10**6
    if (dyn_sys == "tent_map") or (dyn_sys == "baker"):
        mask = MSE_traj < 10**1
    row_sums = np.sum(mask, axis=0)
    # print("row", row_sums, row_sums.shape)

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

    # 5-1. Compute time_avg
    true_timeavg, MSE_timeavg, JAC_timeavg = [], [], []
    for t in range(0, true_avg_traj.shape[0], every):
        true_timeavg.append(np.mean(true_avg_traj[:t]))
        MSE_timeavg.append(np.mean(MSE_avg_traj[:t]))
        JAC_timeavg.append(np.mean(JAC_avg_traj[:t]))

    # 6. plot dist
    pdf_path = '../plot/timeavg_'+str(dyn_sys)+'_all'+'_'+str(N)+'_'+str(len_T)+'.jpg'

    fig, ax1 = subplots(1,figsize=(16,8)) #, sharey=True
    sns.set_theme()

    true_timeavg = np.asarray(true_timeavg)
    num_tau = true_timeavg.shape[0]
    tau_x = np.linspace(0, len_T*time_step, num_tau)

    # tent_map
    # ax1.hist(jac_data[:100, 0].detach().cpu(), bins=50, alpha=0.5, color="slateblue") #density=True,
    # ax1.hist(true_data[:100, 0], bins=50, alpha=0.5, color="salmon") #density=True, 
    # ax1.hist(node_data[:100, 0].detach().cpu(), bins=50, alpha=0.5, color="turquoise") #density=True, 

    # baker
    # ax1.hist(jac_data[:80, 1].detach().cpu(), bins=50, alpha=0.5, color="slateblue") #density=True,
    # ax1.hist(true_data[:80, 1], bins=50, alpha=0.5, color="salmon") #density=True, 
    # ax1.hist(node_data[:80, 1].detach().cpu(), bins=50, alpha=0.5, color="turquoise") #density=True, 
    transition = 0
    if str(dyn_sys) == "lorenz":     # lorenz (before -> 2)
        ax1.semilogy(tau_x[transition:], JAC_timeavg[transition:], color="slateblue", linewidth=2., alpha=0.8, marker='o')
        ax1.semilogy(tau_x[transition:], true_timeavg[transition:], color="salmon", linewidth=2., alpha=0.8, marker='o')
        ax1.semilogy(tau_x[transition:], MSE_timeavg[transition:], color="turquoise",  linewidth=2., alpha=0.8, marker='o')
    elif str(dyn_sys) == "rossler": 
        print("rossler!!!")
        ax1.plot(tau_x, JAC_timeavg, color="slateblue", linewidth=2., alpha=0.8, marker='o')
        ax1.plot(tau_x, true_timeavg, color="salmon", linewidth=2., alpha=0.8, marker='o')
        ax1.plot(tau_x, MSE_timeavg, color="turquoise",  linewidth=2., alpha=0.8, marker='o')
    elif str(dyn_sys) == "hyperchaos":
        ax1.plot(tau_x, JAC_timeavg, color="slateblue", linewidth=2., alpha=0.8, marker='o')
        ax1.plot(tau_x, true_timeavg, color="salmon", linewidth=2., alpha=0.8, marker='o')
        ax1.plot(tau_x, MSE_timeavg, color="turquoise",  linewidth=2., alpha=0.8, marker='o')


    ax1.grid(True)
    ax1.legend(['JAC', 'True', 'MSE'], fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=34)
    ax1.yaxis.set_tick_params(labelsize=34)
    tight_layout()
    savefig(pdf_path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)