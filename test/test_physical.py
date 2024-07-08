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
import scipy

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
long_len_T = 500*int(1/time_step)
init = "outside"
# short_len_T = 50*int(1/time_step)
# true_initial_condition = torch.randn(1, dim)  # Initial condition for the true model
# true_initial_condition = torch.tensor([-9.116407, -3.381641, 33.748295]).reshape(1, dim) # after time 50

if init == "inside":
    true_initial_condition = torch.tensor([-9.116407, -3.381641, 33.748295]).reshape(1, dim)
    pdf_path = '../plot/phase_inside_'+str(model)+'.jpg'
else:
    true_initial_condition = torch.tensor([-15., -15., -5.]).reshape(1, dim)
    pdf_path = '../plot/phase_outside_'+str(model)+'.jpg'

if model == "MLP_skip":
    MSE_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model.pth"
    JAC_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
    mse_model = ODE_MLP_skip(y_dim=dim, n_hidden=512, n_layers=5).to(device)
    best_model = ODE_MLP_skip(y_dim=dim, n_hidden=1024, n_layers=5).to(device)
elif model == "FNO":
    MSE_path = 
    JAC_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
else:
    MSE_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model.pth"
    JAC_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
    mse_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=7).to(device)
    best_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=7).to(device)
mse_model.load_state_dict(torch.load(MSE_path))
best_model.load_state_dict(torch.load(JAC_path))
mse_model.eval()
best_model.eval()

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

learned_long, learned_short = generate_data(best_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)      
mse_long, mse_short = generate_data(mse_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)
learned_long = np.transpose(learned_long, (1, 0, 2))
# learned_short = np.transpose(learned_short, (1, 0, 2))
mse_long = np.transpose(mse_long, (1, 0, 2))
# mse_short = np.transpose(mse_short, (1, 0, 2))

print("orbitshape", true_long.shape, learned_long.shape, mse_long.shape)



# Function to plot histograms for three models in one subplot
def plot_histograms(ax, data_true, data_learned, data_mse, title):
    bins = np.linspace(min(np.min(data_true), np.min(data_learned), np.min(data_mse)), max(np.max(data_true), np.max(data_learned), np.max(data_mse)), 500)

    ax.hist(data_mse, bins=bins, alpha=0.8, label='MSE', color='turquoise', histtype='step', linewidth=2., density=True)
    ax.hist(data_learned, bins=bins, alpha=0.8, label='JAC', color='slateblue', histtype='step', linewidth=2., density=True)
    ax.hist(data_true, bins=bins, alpha=0.8, label='True', color='salmon', histtype='step', linewidth=2., density=True)

    ax.set_title(title, fontsize=30)
    # Set font size for axis labels and title if needed
    # ax.set_xlabel(fontsize=30)
    # ax.set_ylabel(fontsize=30)
    ax.xaxis.set_tick_params(labelsize=34)
    ax.yaxis.set_tick_params(labelsize=34)
    ax.legend(fontsize=30)

fig, axes = plt.subplots(1, 3, figsize=(20, 10))  # 2 rows (time, ensemble) x 3 columns (x, y, z)
dimensions = ['X', 'Y', 'Z']

true_long = np.squeeze(true_long)
learned_long = np.squeeze(learned_long)
mse_long = np.squeeze(mse_long)

torch.set_printoptions(sci_mode=False, precision=5)

# Compute Wasserstein Distance
dist_x = scipy.stats.wasserstein_distance(learned_long[:, 0], true_long[:, 0])
dist_y = scipy.stats.wasserstein_distance(learned_long[:, 1], true_long[:, 1])
dist_z = scipy.stats.wasserstein_distance(learned_long[:, 2], true_long[:, 2])
print("JAC", dist_x, dist_y, dist_z)
print("JAC", torch.norm(torch.tensor([dist_x, dist_y, dist_z])))

dist_x_mse = scipy.stats.wasserstein_distance(mse_long[:, 0], true_long[:, 0])
dist_y_mse = scipy.stats.wasserstein_distance(mse_long[:, 1], true_long[:, 1])
dist_z_mse = scipy.stats.wasserstein_distance(mse_long[:, 2], true_long[:, 2])
print("MSE", dist_x_mse, dist_y_mse, dist_z_mse)
print("MSE", torch.norm(torch.tensor([dist_x_mse, dist_y_mse, dist_z_mse])))

# Compute Time avg
ta_x = np.mean(learned_long[:, 0]) - np.mean(true_long[:, 0])
ta_y = np.mean(learned_long[:, 1]) - np.mean(true_long[:, 1])
ta_z = np.mean(learned_long[:, 2]) - np.mean(true_long[:, 2])
print("Time Avg JAC", ta_x, ta_y, ta_z)
print(torch.norm(torch.tensor([ta_x, ta_y, ta_z])))

ta_x = np.mean(mse_long[:, 0]) - np.mean(true_long[:, 0])
ta_y = np.mean(mse_long[:, 1]) - np.mean(true_long[:, 1])
ta_z = np.mean(mse_long[:, 2]) - np.mean(true_long[:, 2])
print("Time Avg MSE", ta_x, ta_y, ta_z)
print(torch.norm(torch.tensor([ta_x, ta_y, ta_z])))
