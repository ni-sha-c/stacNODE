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
import random
import csv
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
len_T = 100*int(1/time_step)
ind_func = 0
s = 0.2
hidden = 256
model = 'MLP_skip'
num_trajectories = 10000
# long_len_T = 100*int(1/time_step)
short_len_T = 400*int(1/time_step)

torch.manual_seed(42)

if model == "MLP_skip":
    MSE_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model.pth"
    JAC_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
    mse_model = ODE_MLP_skip(y_dim=dim, n_hidden=512, n_layers=3).to(device)
    best_model = ODE_MLP_skip(y_dim=dim, n_hidden=256, n_layers=4).to(device)
else:
    MSE_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model.pth"
    JAC_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model.pth"
    mse_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=5).to(device)
    best_model = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=5).to(device)
mse_model.load_state_dict(torch.load(MSE_path))
best_model.load_state_dict(torch.load(JAC_path))
mse_model.eval()
best_model.eval()

# Function to generate data
def generate_data(model, initial_condition, shortinitial_condition, is_dynamical=True):
    # Long orbit generation
    one_step = torch.linspace(0, time_step, 2).to(device)
    # long_orbit = vectorized_simulate(model, initial_condition, one_step, long_len_T, device).detach().cpu().numpy()
    long_orbit = None
    short_orbit = vectorized_simulate(model, shortinitial_condition, one_step, short_len_T, device).detach().cpu().numpy()
    print("short orbit length", short_orbit.shape)
    
    return long_orbit, short_orbit

# Generate true dynamics long orbit and its short orbits
true_initial_condition = torch.randn(1, dim)  # Initial condition for the true model
# trueshort_initial_condition = torch.randn(num_trajectories,dim) 
x = -20 + 40 * torch.randn(num_trajectories, 1)
y = -20 + 40 * torch.randn(num_trajectories, 1)
z = 10 + 30 * torch.randn(num_trajectories, 1)
trueshort_initial_condition = torch.cat((x, y, z), dim=1)
print("concat shape", trueshort_initial_condition.shape)

true_long, true_short= generate_data(dyn, true_initial_condition, trueshort_initial_condition, is_dynamical=True)
# true_long = np.transpose(true_long, (1, 0, 2))
true_short = np.transpose(true_short, (1, 0, 2))
# print("orbitshape", true_long.shape, "short", true_short.shape) #orbitshape (100000, 3) short (500, 5000, 3)

learned_long, learned_short = generate_data(best_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)      
mse_long, mse_short = generate_data(mse_model, true_initial_condition, trueshort_initial_condition, is_dynamical=False)
# learned_long = np.transpose(learned_long, (1, 0, 2))
learned_short = np.transpose(learned_short, (1, 0, 2))
# mse_long = np.transpose(mse_long, (1, 0, 2))
mse_short = np.transpose(mse_short, (1, 0, 2))
# print("orbitshape", true_long.shape, learned_long.shape, mse_long.shape, "short", true_short.shape, learned_short.shape, mse_short.shape)

learned_LE = []
mse_LE = []
true_LE = []

for i in range(num_trajectories):
    print(i)
    true_LE.append(lyap_exps([dyn_sys, dyn, dim, time_step], s, torch.tensor(true_short[i]), short_len_T).detach().cpu().numpy())
    learned_LE.append(lyap_exps([dyn_sys, best_model, dim, time_step], s, torch.tensor(learned_short[i]), short_len_T).detach().cpu().numpy())
    mse_LE.append(lyap_exps([dyn_sys, mse_model, dim, time_step], s, torch.tensor(mse_short[i]), short_len_T).detach().cpu().numpy())


JAC_file = "JAC_LE_500_2.csv"
MSE_file = "MSE_LE_500_2.csv"
TRUE_file = "TRUE_LE_500_2.csv"

# Open the CSV file in write mode
with open(JAC_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(learned_LE)

with open(MSE_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(mse_LE)

with open(TRUE_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(true_LE)