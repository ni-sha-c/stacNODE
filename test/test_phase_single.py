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
import matplotlib.cm as cm

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

if init == "inside":
    true_initial_condition = torch.tensor([-9.116407, -3.381641, 33.748295]).reshape(1, dim)
    pdf_path = '../plot/phase_inside_'+str(model)+'.jpg'
else:
    true_initial_condition = torch.tensor([-15., -15., -5.]).reshape(1, dim)
    pdf_path = '../plot/phase_outside_'+str(model)+'.jpg'



mlp_path = "../plot/Vector_field/unroll2_40_Res/best_model.pth"
path = "../plot/Vector_field/unroll2_40_Res/attractor.png"
model = ODE_MLP_skip(y_dim=dim, n_hidden=256, n_layers=5).to(device)

state_dict = torch.load(mlp_path)
print(state_dict.keys())
print(model.state_dict().keys())


model.load_state_dict(torch.load(mlp_path))
model.eval()


# Function to generate data
def generate_data(model, initial_condition, shortinitial_condition, is_dynamical=True):
    # Long orbit generation
    one_step = torch.linspace(0, time_step, 2).to(device)
    long_orbit = vectorized_simulate(model, initial_condition, one_step, long_len_T, device).detach().cpu().numpy()
    
    short_orbit = None
    
    return long_orbit, short_orbit

# Generate true dynamics long orbit and its short orbits
trueshort_initial_condition = torch.randn(num_trajectories,dim) 
true_o, true_short= generate_data(dyn, true_initial_condition, trueshort_initial_condition, is_dynamical=True)
true_o = np.transpose(true_o, (1, 0, 2))
print("true long inside initial orbit", true_o[0, -1])
# true_short = np.transpose(true_short, (1, 0, 2))

print("orbitshape", true_o.shape) #orbitshape (100000, 3) short (500, 5000, 3)

learned_o, learned_short = generate_data(model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)      
# mse_long, mse_short = generate_data(mse_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)
learned_o = np.transpose(learned_o, (1, 0, 2))

print("orbitshape", true_o.shape, learned_o.shape)

print(learned_o.shape)
print("plotting attractor!", path)
# generate true orbit and learned orbit
dyn, dim, time_step = lorenz, 3, 0.01
# Define the colormap
cmap = cm.plasma

fig, axs = plt.subplots(2, 3, figsize=(24,12))
num_row, num_col = axs.shape


for x in range(num_row):
    for y in range(num_col):
        orbit = true_o.squeeze() if x == 0 else learned_o.squeeze()
        print(num_row, num_col)
        print(orbit[:,2])
        orbit_norm = orbit[:, 2]
        if y == 0:
            axs[x,y].plot(orbit[0, 0], orbit[0, 1], '+', markersize=35, color=cmap(0))
            scatter = axs[x,y].scatter(orbit[:, 0], orbit[:, 1], c=orbit_norm, s=6, cmap='plasma', alpha=0.5)
            axs[x,y].set_xlabel("X")
            axs[x,y].set_ylabel("Y")
        elif y == 1:
            axs[x,y].plot(orbit[0, 0], orbit[0, 2], '+', markersize=35, color=cmap(0))
            scatter = axs[x,y].scatter(orbit[:, 0], orbit[:, 2], c=orbit_norm, s=6, cmap='plasma', alpha=0.5)
            axs[x,y].set_xlabel("X")
            axs[x,y].set_ylabel("Z")
        else:
            axs[x,y].plot(orbit[0, 1], orbit[0, 2], '+', markersize=35, color=cmap(0))
            scatter = axs[x,y].scatter(orbit[:, 1], orbit[:, 2], c=orbit_norm, s=6, cmap='plasma', alpha=0.5)
            axs[x,y].set_xlabel("Y")
            axs[x,y].set_ylabel("Z")
        
        axs[x,y].tick_params(labelsize=42)
        axs[x,y].xaxis.label.set_size(42)
        axs[x,y].yaxis.label.set_size(42)

plt.tight_layout()
fig.savefig(path, format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)