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
model = 'MLP'
num_trajectories = 5000
long_len_T = 500*int(1/time_step)
init = "outside"
# short_len_T = 50*int(1/time_step)
# true_initial_condition = torch.randn(1, dim)  # Initial condition for the true model
# true_initial_condition = torch.tensor([-9.116407, -3.381641, 33.748295]).reshape(1, dim) # after time 50

if init == "inside":
    true_initial_condition = torch.tensor([-9.116407, -3.381641, 33.748295]).reshape(1, dim)
    pdf_path = '../plot/phase_inside_'+str(model)+'.jpg'
else:
    true_initial_condition = torch.tensor([-15., -15., 5.]).reshape(1, dim)
    pdf_path = '../plot/phase_outside_all_'+str(model)+'.jpg'

model='MLP_skip'
MSE_MS_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model.pth"
JAC_MS_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
mse_ms_model = ODE_MLP_skip(y_dim=dim, n_hidden=512, n_layers=5).to(device)
best_ms_model = ODE_MLP_skip(y_dim=dim, n_hidden=1024, n_layers=5).to(device)

model='MLP'
MSE_mlp_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model.pth"
JAC_mlp_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
mse_mlp_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=7).to(device)
best_mlp_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=7).to(device)


mse_ms_model.load_state_dict(torch.load(MSE_MS_path))
best_ms_model.load_state_dict(torch.load(JAC_MS_path))
mse_mlp_model.load_state_dict(torch.load(MSE_mlp_path))
best_mlp_model.load_state_dict(torch.load(JAC_mlp_path))
mse_ms_model.eval()
best_ms_model.eval()
mse_mlp_model.eval()
best_mlp_model.eval()

# Function to generate data
def generate_data(model, initial_condition, shortinitial_condition, is_dynamical=True):
    # Long orbit generation
    one_step = torch.linspace(0, time_step, 2).to(device)
    long_orbit = vectorized_simulate(model, initial_condition, one_step, long_len_T, device).detach().cpu().numpy()
    
    # short_orbit = vectorized_simulate(model, shortinitial_condition, one_step, short_len_T, device).detach().cpu().numpy()
    # print("short orbit length", short_orbit.shape)
    short_orbit = None
    
    return long_orbit, short_orbit

# Generate true dynamics long orbit and its short orbits
trueshort_initial_condition = torch.randn(num_trajectories,dim) 
true_long, true_short= generate_data(dyn, true_initial_condition, trueshort_initial_condition, is_dynamical=True)
true_long = np.transpose(true_long, (1, 0, 2))
print("true long inside initial orbit", true_long[0, -1])
# true_short = np.transpose(true_short, (1, 0, 2))

print("orbitshape", true_long.shape) #orbitshape (100000, 3) short (500, 5000, 3)

learned_ms_long, learned_ms_short = generate_data(best_ms_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)      
mse_ms_long, mse_ms_short = generate_data(mse_ms_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)
learned_mlp_long, learned_mlp_short = generate_data(best_mlp_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)      
mse_mlp_long, mse_mlp_short = generate_data(mse_mlp_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)

learned_ms_long = np.transpose(learned_ms_long, (1, 0, 2))
mse_ms_long = np.transpose(mse_ms_long, (1, 0, 2))
learned_mlp_long = np.transpose(learned_mlp_long, (1, 0, 2))
mse_mlp_long = np.transpose(mse_mlp_long, (1, 0, 2))

# print("orbitshape", true_long.shape, learned_long.shape, mse_long.shape)



# Function to plot histograms for three models in one subplot
def plot_histograms(ax, data_true, data_learned, data_mse, title):
    bins = np.linspace(min(np.min(data_true), np.min(data_learned), np.min(data_mse)), max(np.max(data_true), np.max(data_learned), np.max(data_mse)), 500)

    ax.hist(data_mse, bins=bins, alpha=0.8, label='MSE', color='turquoise', histtype='step', linewidth=2., density=True)
    ax.hist(data_learned, bins=bins, alpha=0.8, label='JAC', color='slateblue', histtype='step', linewidth=2., density=True)
    ax.hist(data_true, bins=bins, alpha=0.8, label='True', color='salmon', histtype='step', linewidth=2., density=True)

    ax.set_title(title, fontsize=30)
    ax.xaxis.set_tick_params(labelsize=34)
    ax.yaxis.set_tick_params(labelsize=34)
    ax.legend(fontsize=30)

true_long = np.squeeze(true_long)
learned_ms_long = np.squeeze(learned_ms_long)
mse_ms_long = np.squeeze(mse_ms_long)
learned_mlp_long = np.squeeze(learned_mlp_long)
mse_mlp_long = np.squeeze(mse_mlp_long)



# create plot of attractor with initial point starting from 
fig, axs = subplots(1, 5, figsize=(56,7))
# fig, axs = subplots(ncols=6,figsize=(42,7), gridspec_kw={"width_ratios":[1,1,1,1.,1., 0.05]})
cmap = cm.hsv
traj = [true_long, mse_mlp_long, mse_ms_long, learned_mlp_long, learned_ms_long]
title = ['TRUE', 'MSE_MLP', 'MSE_Res', 'JAC_MLP', 'JAC_Res']


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create figure and subplots
fig, axs = plt.subplots(1, 5, figsize=(42, 7), constrained_layout=True)

# Compute overall min and max for the colorbar
all_orbit_values = np.concatenate([orbit[:, 2] for orbit in traj])
vmin, vmax = all_orbit_values.min(), all_orbit_values.max()

for y in range(5):
    orbit = traj[y]
    axs[y].plot(orbit[0, 0], orbit[0, 2], '+', markersize=35, color=cmap(0), linewidth=40)
    scatter = axs[y].scatter(orbit[:, 0], orbit[:, 2], c=orbit[:, 2], s=6, cmap='plasma', alpha=0.5, vmin=vmin, vmax=vmax)
    axs[y].set_xlabel("X")
    if y == 0:
        axs[y].set_ylabel("Z")
    axs[y].tick_params(labelsize=45)
    axs[y].xaxis.label.set_size(45)
    axs[y].yaxis.label.set_size(45)
    axs[y].set_title(title[y], fontsize=45)

# Create a colorbar for the scatter plots
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(scatter, cax=cax)
cbar.ax.tick_params(labelsize=45)  # Set colorbar tick label size

fig.savefig(pdf_path, format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)


# # fig, (ax1, ax2, ax3, ax4, ax5, cax) = plt.subplots(ncols=5,figsize=(42,7), gridspec_kw={"width_ratios":[1,1,1,1,1, 0.05]})
# fig.subplots_adjust(wspace=0.3)
# # im  = ax1.imshow(np.random.rand(11,8))
# # im2 = ax2.imshow(np.random.rand(11,8))
# # ax1.set_ylabel("y label")
# # ax1.set_xlabel("")
# # fig.colorbar(im, cax=cax)


# for y in range(5):
#     orbit = traj[y]
#     axs[y].plot(orbit[0, 0], orbit[0, 2], '+', markersize=35, color=cmap.colors[0], linewidth=40)
#     if y == 1:
#         scatter = axs[y].scatter(orbit[:, 0], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
#     else:
#         axs[y].scatter(orbit[:, 0], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
#     axs[y].set_xlabel("X")
#     if y == 0:
#         axs[y].set_ylabel("Z")
#     axs[y].tick_params(labelsize=45)
#     axs[y].xaxis.label.set_size(45)
#     axs[y].yaxis.label.set_size(45)
#     axs[y].set_title(title[y], fontsize=45)

# cbar = fig.colorbar(scatter, ax=axs)
# cbar.ax.tick_params(labelsize=45)  # Set colorbar tick label size
# # tight_layout()
# fig.savefig(pdf_path, format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
