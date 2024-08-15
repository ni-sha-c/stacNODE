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
model = 'MLP_skip'
num_trajectories = 5000
long_len_T = 1000*int(1/time_step) #5000
init = "outside"


if init == "inside":
    true_initial_condition = torch.tensor([-9.116407, -3.381641, 33.748295]).reshape(1, dim)
    pdf_path = '../plot/dist_inside_'+str(model)+'.jpg'
else:
    true_initial_condition = torch.tensor([-15, -15, 0.]).reshape(1, dim)
    pdf_path = '../plot/dist_outside_all'+str(model)+'.jpg'
    pdf_path_2 = '../plot/dist_outside_all_res.jpg'


if model =='MLP_skip':
    MSE_MS_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model.pth"
    JAC_MS_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
    mse_model = ODE_MLP_skip(y_dim=dim, n_hidden=512, n_layers=5).to(device)
    best_model = ODE_MLP_skip(y_dim=dim, n_hidden=1024, n_layers=5).to(device)

    mse_model.load_state_dict(torch.load(MSE_MS_path))
    best_model.load_state_dict(torch.load(JAC_MS_path))
    mse_model.eval()
    best_model.eval()
else:
    # model='MLP'
    MSE_mlp_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model.pth"
    JAC_mlp_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
    mse_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=7).to(device)
    best_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=3).to(device)

    m = torch.load(JAC_mlp_path)
    print("state", m.keys())
    mse_model.load_state_dict(torch.load(MSE_mlp_path))
    best_model.load_state_dict(torch.load(JAC_mlp_path))
    mse_model.eval()
    best_model.eval()



# Function to generate data
def generate_data(model, initial_condition, shortinitial_condition, is_dynamical=True):
    # Long orbit generation
    one_step = torch.linspace(0, time_step, 2).to(device)
    long_orbit = vectorized_simulate(model, initial_condition, one_step, long_len_T, device).detach().cpu().numpy()
    short_orbit = None
    
    return long_orbit, short_orbit

# Generate true dynamics long orbit and its short orbits
trueshort_initial_condition = torch.randn(num_trajectories,dim) 
true_long, true_short= generate_data(dyn, true_initial_condition, trueshort_initial_condition, is_dynamical=True)
true_long = np.transpose(true_long, (1, 0, 2))
print("true long inside initial orbit", true_long[0, -1])
# true_short = np.transpose(true_short, (1, 0, 2))

print("orbitshape", true_long.shape) #orbitshape (100000, 3) short (500, 5000, 3)

learned_long, learned_short = generate_data(best_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)      
mse_long, mse_short = generate_data(mse_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)


learned_long = np.transpose(learned_long, (1, 0, 2))
mse_long = np.transpose(mse_long, (1, 0, 2))



# Function to plot histograms for three models in one subplot
def plot_histograms(ax, data_true, data_learned, data_mse, title, first, idx):
    bins = np.linspace(min(np.min(data_true), np.min(data_learned), np.min(data_mse)), max(np.max(data_true), np.max(data_learned), np.max(data_mse)), 500)
    selected_values = mse_long[(mse_long[:, 0] >= -23) & (mse_long[:, 0] <= 23)]
    selected_values_l = learned_long[(learned_long[:, 0] >= -23) & (learned_long[:, 0] <= 23)]
    # print("selected val", selected_values_l[-100:, 0])

    if idx == 0:
        ax.hist(data_true, bins=bins, alpha=0.6, label='True', color='black', histtype='step', linewidth=5., density=True)
    elif (idx == 1) or (idx == 2):
        ax.hist(data_true, bins=bins, alpha=0.6, label='True', color='black', histtype='step', linewidth=5., density=True)
        ax.hist(selected_values[:, 0], bins=bins, alpha=0.7, label='MSE', color='red', histtype='step', linewidth=5., density=True)
        ax.hist(selected_values_l[:, 0], bins=bins, alpha=0.6, label='JAC', color='blue', histtype='step', linewidth=5., density=True)

    ax.set_title(title, fontsize=45)
    ax.xaxis.set_tick_params(labelsize=45)
    ax.yaxis.set_tick_params(labelsize=45)
    ax.set_xlabel("X", fontsize=45)
    # ax.set_xlim([-23, 23])
    ax.legend(fontsize=45)

fig, axes = plt.subplots(1, 2, figsize=(28, 7))  # 2 rows (time, ensemble) x 3 columns (x, y, z)
dimensions = ['X', 'Y', 'Z']
title = ['TRUE', 'Empirical Density', 'JAC_MLP']

true_long = np.squeeze(true_long)
learned_long = np.squeeze(learned_long)
mse_long = np.squeeze(mse_long)

models = [true_long, mse_long, learned_long]

for j in range(2): 
    m = models[j]
    index = 0
    if j == 0:
        plot_histograms(axes[j], true_long[:, index], m[:, index], m[:, index], f'{title[j]}', True, j)
    else:
        plot_histograms(axes[j], true_long[:, index], m[:, index], m[:, index], f'{title[j]}', False, j)
    # plot_histograms(axes[1, j], true_short[:, :, j].flatten(), learned_short[:, :, j].flatten(), mse_short[:, :, j].flatten(), f'Ensemble Avg - {dimensions[j]}')

plt.tight_layout()
plt.savefig(pdf_path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
# plt.show()
