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

# def auto_corr(traj):
#     # res = np.correlate(traj, traj, mode='full')
#     # return res[res.size // 2:]
#     res = scipy.signal.correlate(traj, traj, mode='same')
#     return res[res.size // 2:]/np.max(res)

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


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. define system
    dyn_sys= "lorenz"
    dyn_sys_f, dim = define_dyn_sys(dyn_sys)
    time_step= 0.01
    tau=500
    len_T = tau*int(1/time_step)

    # 2. define num init points
    N = 2000
    inits = torch.randn(N, dim).double().to(device)
    every = 100
    ind_func = 0
    vec_len = 100 #300

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
    print("JAC shape", JAC_traj.shape)
    print("True shape", true_traj.shape)

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
    # true_mean = np.mean(true_avg_traj)
    # print("avg traj shape:", JAC_avg_traj.shape)

    # 5-1. Compute autocorr
    true_ac = auto_corr(true_avg_traj, tau, ind_func, time_step, vec_len)
    MSE_ac = auto_corr(MSE_avg_traj, tau, ind_func, time_step, vec_len)
    JAC_ac = auto_corr(JAC_avg_traj, tau, ind_func, time_step, vec_len)

    # 6. plot dist
    pdf_path = '../plot/corr_'+str(dyn_sys)+'_'+str(N)+'_'+str(len_T)+'.jpg'

    fig, ax1 = subplots(1,figsize=(16,8)) #, sharey=True
    sns.set_theme()

    num_tau = true_ac.shape[0]
    tau_x = np.linspace(0, tau, num_tau)
    # print("true_mean", true_mean)


    transition = 0
    if str(dyn_sys) == "lorenz":     # lorenz (before -> 2)
        ax1.semilogy(tau_x, JAC_ac, color="slateblue", linewidth=2., alpha=0.8, marker='o', markevery=every)
        ax1.semilogy(tau_x, true_ac, color="salmon", linewidth=2., alpha=0.8, marker='o', markevery=every)
        ax1.semilogy(tau_x, MSE_ac, color="turquoise",  linewidth=2., alpha=0.8, marker='o', markevery=every)
    elif str(dyn_sys) == "rossler": 
        print("rossler!!!")
        ax1.plot(tau_x, JAC_ac, color="slateblue", linewidth=2., alpha=0.8, marker='o', markevery=every)
        ax1.plot(tau_x, true_ac, color="salmon", linewidth=2., alpha=0.8, marker='o', markevery=every)
        ax1.plot(tau_x, MSE_ac, color="turquoise",  linewidth=2., alpha=0.8, marker='o', markevery=every)
    elif str(dyn_sys) == "hyperchaos":
        ax1.plot(tau_x, JAC_ac, color="slateblue", linewidth=2., alpha=0.8, marker='o', markevery=every)
        ax1.plot(tau_x, true_ac, color="salmon", linewidth=2., alpha=0.8, marker='o', markevery=every)
        ax1.plot(tau_x, MSE_ac, color="turquoise",  linewidth=2., alpha=0.8, marker='o', markevery=every)


    ax1.grid(True)
    ax1.legend(['JAC', 'True', 'MSE'], fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=34)
    ax1.yaxis.set_tick_params(labelsize=34)
    tight_layout()
    savefig(pdf_path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)