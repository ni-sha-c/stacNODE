import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
import datetime
import numpy as np
import argparse
import json
import logging
import os
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d

import sys
sys.path.append('..')

from src.util import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. define system
dyn_sys= "lorenz"
dyn, dim = define_dyn_sys(dyn_sys)
time_step= 0.01
ind_func = 0
s = 0.2
hidden = 256
model = 'FNO'
num_trajectories = 5000
long_len_T = 3000*int(1/time_step)
init = "inside"

if init == "inside":
    true_initial_condition = torch.tensor([-9.116407, -3.381641, 33.748295]).reshape(1, dim)
    pdf_path = '../plot/dist_inside_'+str(model)+'.jpg'
else:
    true_initial_condition = torch.tensor([-15, -15, -5.]).reshape(1, dim)
    pdf_path = '../plot/dist_outside_all'+str(model)+'.jpg'
    pdf_path_2 = '../plot/dist_outside_all_Res.jpg'

model='MLP_skip'
MSE_MS_path = "../plot/Vector_field/"+str(dyn_sys)+"/best_model_MSE.pth"
JAC_MS_path = "../plot/Vector_field/"+str(dyn_sys)+"/best_model_JAC.pth"
mse_ms_model = ODE_MLP_skip(y_dim=dim, n_hidden=512, n_layers=5).to(device)
best_ms_model = ODE_MLP_skip(y_dim=dim, n_hidden=1024, n_layers=5).to(device)

# model='MLP'
# MSE_mlp_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model.pth"
# JAC_mlp_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
# mse_mlp_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=7).to(device)
# best_mlp_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=7).to(device)


mse_ms_model.load_state_dict(torch.load(MSE_MS_path))
best_ms_model.load_state_dict(torch.load(JAC_MS_path))
# mse_mlp_model.load_state_dict(torch.load(MSE_mlp_path))
# best_mlp_model.load_state_dict(torch.load(JAC_mlp_path))
mse_ms_model.eval()
best_ms_model.eval()
# mse_mlp_model.eval()
# best_mlp_model.eval()

# Function to generate data
def generate_data(model, initial_condition, shortinitial_condition, is_dynamical=True):
    # Long orbit generation
    one_step = torch.linspace(0, time_step, 2).to(device)
    long_orbit = vectorized_simulate(model, initial_condition, one_step, long_len_T, device).detach().cpu().numpy()
    
    # short_orbit = vectorized_simulate(model, shortinitial_condition, one_step, short_len_T, device).detach().cpu().numpy()
    # print("short orbit length", short_orbit.shape)
    short_orbit = None
    
    return long_orbit, short_orbit

def calculate_relative_error(model, dyn, device):
    # Simulate an orbit using the true dynamics
    time_step = 0.01  # Example timestep, adjust as needed
    orbit = torchdiffeq.odeint(dyn, torch.randn(3), torch.arange(0, 100, time_step), method='rk4', rtol=1e-8)
    
    orbit = orbit.to('cuda')
    # Compute vector field from model and true dynamics
    vf_nn = model(0, orbit).detach()
    vf_true = torch.stack([dyn(0, orbit[i]) for i in range(orbit.size(0))])

    # Calculate relative error
    err = torch.linalg.norm(vf_nn - vf_true, dim=1)
    mag = torch.linalg.norm(vf_true, dim=1)
    # relative_error = torch.mean(err / mag).item() * 100  # As percentage
    print("err", err.shape, mag.shape)
    relative_error = err / mag
    print("max", torch.max(relative_error))
    print("min", torch.min(relative_error))
    print("mean", torch.mean(relative_error))
    return relative_error

rel = calculate_relative_error(mse_ms_model, dyn, 'cuda')
rel = calculate_relative_error(best_ms_model, dyn, 'cuda')

# # Generate true dynamics long orbit and its short orbits
# trueshort_initial_condition = torch.randn(num_trajectories,dim) 
# true_long, true_short= generate_data(dyn, true_initial_condition, trueshort_initial_condition, is_dynamical=True)
# true_long = np.transpose(true_long, (1, 0, 2))
# print("true long inside initial orbit", true_long[0, -1])
# # true_short = np.transpose(true_short, (1, 0, 2))

# print("orbitshape", true_long.shape) #orbitshape (100000, 3) short (500, 5000, 3)

# learned_ms_long, learned_ms_short = generate_data(best_ms_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)      
# mse_ms_long, mse_ms_short = generate_data(mse_ms_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)
# # learned_mlp_long, learned_mlp_short = generate_data(best_mlp_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)      
# # mse_mlp_long, mse_mlp_short = generate_data(mse_mlp_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)

# learned_ms_long = np.transpose(learned_ms_long, (1, 0, 2))
# mse_ms_long = np.transpose(mse_ms_long, (1, 0, 2))
# # learned_mlp_long = np.transpose(learned_mlp_long, (1, 0, 2))
# # mse_mlp_long = np.transpose(mse_mlp_long, (1, 0, 2))



# # Function to plot histograms for three models in one subplot
# def plot_histograms(ax, data_true, data_learned, data_mse, title, first, idx):
#     bins = np.linspace(min(np.min(data_true), np.min(data_learned), np.min(data_mse)), max(np.max(data_true), np.max(data_learned), np.max(data_mse)), 500)

#     # ax.hist(data_mse, bins=bins, alpha=0.8, label='MSE', color='turquoise', histtype='step', linewidth=2., density=True)
#     if (idx == 0):
#         ax.hist(data_true, bins=bins, alpha=0.6, label='True', color='black', histtype='step', linewidth=5., density=True)
#         ax.hist(data_learned, bins=bins, alpha=0.7, label='Model', color='red', histtype='step', linewidth=5., density=True)
#     else:
#         ax.hist(data_true, bins=bins, alpha=0.6, label='True', color='black', histtype='step', linewidth=5., density=True)
#         ax.hist(data_learned, bins=bins, alpha=0.7, label='Model', color='blue', histtype='step', linewidth=5., density=True)

#     ax.set_title(title, fontsize=45)
#     ax.xaxis.set_tick_params(labelsize=45)
#     ax.yaxis.set_tick_params(labelsize=45)
#     ax.legend(fontsize=45)

# fig, axes = plt.subplots(1, 2, figsize=(21, 7))  # 2 rows (time, ensemble) x 3 columns (x, y, z)
# dimensions = ['X', 'Y', 'Z']
# title = ['MSE_MLP', 'JAC_MLP']

# true_long = np.squeeze(true_long)
# learned_ms_long = np.squeeze(learned_ms_long)
# mse_ms_long = np.squeeze(mse_ms_long)
# # learned_mlp_long = np.squeeze(learned_mlp_long)
# # mse_mlp_long = np.squeeze(mse_mlp_long)



# models = [mse_mlp_long, learned_mlp_long]

# for j in range(2): 
#     # print("true", true_long.shape, learned_long.shape, mse_long.shape, axes.shape, dimensions)
#     m = models[j]
#     index = 0
#     if j == 0:
#         plot_histograms(axes[j], true_long[:, index], m[:, index], m[:, index], f'{title[j]}', True, j)
#     else:
#         plot_histograms(axes[j], true_long[:, index], m[:, index], m[:, index], f'{title[j]}', False, j)
#     # plot_histograms(axes[1, j], true_short[:, :, j].flatten(), learned_short[:, :, j].flatten(), mse_short[:, :, j].flatten(), f'Ensemble Avg - {dimensions[j]}')

# plt.tight_layout()
# plt.savefig(pdf_path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
# # plt.show()


# fig, axes = plt.subplots(1, 2, figsize=(21, 7))  # 2 rows (time, ensemble) x 3 columns (x, y, z)
# title = ['MSE_Res', 'JAC_Res']

# models = [mse_ms_long, learned_ms_long]

# for j in range(2): 
#     m = models[j]
#     index = 0
#     if j == 0:
#         plot_histograms(axes[j], true_long[:, index], m[:, index], m[:, index], f'{title[j]}', True, j)
#     else:
#         plot_histograms(axes[j], true_long[:, index], m[:, index], m[:, index], f'{title[j]}', False, j)
#     # plot_histograms(axes[1, j], true_short[:, :, j].flatten(), learned_short[:, :, j].flatten(), mse_short[:, :, j].flatten(), f'Ensemble Avg - {dimensions[j]}')

# plt.tight_layout()
# plt.savefig(pdf_path_2, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
# # plt.show()
