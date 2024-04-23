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
import math
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d


from torch.utils.data import DataLoader, TensorDataset
from modulus.models.fno import FNO
# from modulus.launch.logging import LaunchLogger
# from modulus.launch.utils.checkpoint import save_checkpoint

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    ### Equation ###
    def lorenz(t, u, params=[10.0,28.0,8/3]):
        """ Lorenz chaotic differential equation: du/dt = f(t, u)
        t: time T to evaluate system
        u: state vector [x, y, z] 
        return: new state vector in shape of [3]"""

        du = torch.stack([
                params[0] * (u[1] - u[0]),
                u[0] * (params[1] - u[2]) - u[1],
                (u[0] * u[1]) - (params[2] * u[2])
            ])
        return du

    ### Dataset ###
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

    print("Creating Dataset")
    dataset = create_data([lorenz, 3, 0.01], n_train=10000, n_test=3000, n_val=3000, n_trans=0)
    train_list = [dataset[0], dataset[1]]
    val_list = [dataset[2], dataset[3]]
    test_list = [dataset[4], dataset[5]]

    train_data = TensorDataset(*train_list)
    val_data = TensorDataset(*val_list)
    dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False)
    print(len(dataloader), dataloader.batch_size)

    model = FNO(
        in_channels=3,
        out_channels=3,
        num_fno_modes=2,
        dimension=1
    ).to('cuda')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3)

    ### Training Loop ###
    n_store, k  = 100, 0
    jac_diff_train, jac_diff_test = torch.empty(n_store+1), torch.empty(n_store+1)
    print("Computing analytical Jacobian")
    f = lambda x: lorenz(0, x)
    true_jac_fn = torch.vmap(torch.func.jacrev(f))
    True_J = true_jac_fn(train_list[0])
    Test_J = true_jac_fn(test_list[0])
    
    print("Beginning training")
    num_epochs = 10
    for epoch in range(num_epochs):
        # with LaunchLogger(
        #         "train",
        #         epoch=epoch,
        #         num_mini_batch=len(dataloader),
        #         epoch_alert_freq=10
        #     ) as log:
        print("epoch: ", epoch)
        for data in dataloader:
            optimizer.zero_grad()
            y_true = data[1].to('cuda')
            print(data[0])
            y_pred = model(data[0].unsqueeze(dim=2).to('cuda'))

            # MSE Loss
            loss_mse = criterion(y_pred.squeeze(), y_true)
            time_step = 0.01
            loss = loss_mse * (1/time_step/time_step)
            print("loss_mse: ", loss_mse)
            
            jac = torch.func.jacrev(model)
            compute_batch_jac = torch.vmap(jac, in_dims=(0))
            x = data[0].unsqueeze(dim=1).unsqueeze(dim=3).to('cuda')
            cur_model_J = compute_batch_jac(x)
            jac_norm_diff = loss_mse(True_J, cur_model_J)
            print("loss_jac: ", jac_norm_diff)
            reg_param=500
            loss += reg_param*jac_norm_diff
            print("loss: ", loss)

            loss.backward()
            optimizer.step()
        # log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # save_checkpoint(
        #     "./checkpoints",
        #     models=[model],
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     epoch=epoch
        # )

if __name__ == "__main__":
    main()