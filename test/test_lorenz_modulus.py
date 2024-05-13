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
import csv
import math
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d

# mpirun -n 2 python test_....

from torch.utils.data import DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected
# from modulus.launch.logging import LaunchLogger
# from modulus.launch.utils.checkpoint import save_checkpoint

class Timer:
    def __init__(self):
        self.elapsed_times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.elapsed_times.append(self.elapsed_time)
        return False

def main(logger, loss_type):

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

    ### Compute Metric ###
    def rk4(x, f, dt):
        k1 = f(0, x)
        k2 = f(0, x + dt*k1/2)
        k3 = f(0, x + dt*k2/2)
        k4 = f(0, x + dt*k3)
        return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
    def lyap_exps(dyn_sys_info, ds_name, traj, iters, batch_size):
        model, dim, time_step = dyn_sys_info
        LE = torch.zeros(dim).to(device)
        traj_gpu = traj.to(device)
        if model == lorenz:
            f = lambda x: rk4(x, model, time_step)
            Jac = torch.vmap(torch.func.jacrev(f))(traj_gpu)
        else:
            f = model
            # traj_in_batch = traj_gpu.reshape(-1, 1, dim, 1)
            traj_data = TensorDataset(traj_gpu)
            traj_loader = DataLoader(traj_data, batch_size=batch_size, shuffle=False)
            Jac = torch.randn(traj_gpu.shape[0], dim, dim).cuda()
            i = 0
            for traj in traj_loader:
                # print("shape", traj)
                jac = torch.func.jacrev(f)
                x = traj[0].unsqueeze(dim=2).to('cuda')
                batchsize = x.shape[0]
                cur_model_J = jac(x)
                squeezed_J = cur_model_J[:, :, 0, :, :, 0]
                non_zero_indices = torch.nonzero(squeezed_J)
                non_zero_values = squeezed_J[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]
                learned_J = non_zero_values.reshape(batchsize, 3, 3)
                Jac[i:i+batchsize] = learned_J
                i +=batchsize
            print(Jac)

        Q = torch.rand(dim,dim).to(device)
        eye_cuda = torch.eye(dim).to(device)
        for i in range(iters):
            if i > 0 and i % 1000 == 0:
                print("Iteration: ", i, ", LE[0]: ", LE[0].detach().cpu().numpy()/i/time_step)
            Q = torch.matmul(Jac[i], Q)
            Q, R = torch.linalg.qr(Q)
            LE += torch.log(abs(torch.diag(R)))
        return LE/iters/time_step

    def model_size(model):
        # Adapted from https://discuss.pytorch.org/t/finding-model-size/130275/11
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))
        return size_all_mb

    def plot_attractor(model, dyn_info, time, path):
        # generate true orbit and learned orbit
        dyn, dim, time_step = dyn_info
        tran_orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
        true_o = torchdiffeq.odeint(dyn, tran_orbit[-1], torch.arange(0, time, time_step), method='rk4', rtol=1e-8)

        learned_o = torch.zeros(time*int(1/time_step), dim)
        x0 = tran_orbit[-1]
        for t in range(time*int(1/time_step)):
            learned_o[t] = x0
            new_x = model(x0.reshape(1, dim, 1).cuda())
            x0 = new_x.squeeze()
        learned_o = learned_o.detach().cpu().numpy()

        # create plot of attractor with initial point starting from 
        fig, axs = subplots(2, 3, figsize=(24,12))
        cmap = cm.plasma
        num_row, num_col = axs.shape

        for x in range(num_row):
            for y in range(num_col):
                orbit = true_o if x == 0 else learned_o
                if y == 0:
                    axs[x,y].plot(orbit[0, 0], orbit[0, 1], '+', markersize=35, color=cmap.colors[0])
                    axs[x,y].scatter(orbit[:, 0], orbit[:, 1], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
                    axs[x,y].set_xlabel("X")
                    axs[x,y].set_ylabel("Y")
                elif y == 1:
                    axs[x,y].plot(orbit[0, 0], orbit[0, 2], '+', markersize=35, color=cmap.colors[0])
                    axs[x,y].scatter(orbit[:, 0], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
                    axs[x,y].set_xlabel("X")
                    axs[x,y].set_ylabel("Z")
                else:
                    axs[x,y].plot(orbit[0, 1], orbit[0, 2], '+', markersize=35, color=cmap.colors[0])
                    axs[x,y].scatter(orbit[:, 1], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
                    axs[x,y].set_xlabel("Y")
                    axs[x,y].set_ylabel("Z")
            
                axs[x,y].tick_params(labelsize=42)
                axs[x,y].xaxis.label.set_size(42)
                axs[x,y].yaxis.label.set_size(42)
        tight_layout()
        fig.savefig(path, format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
        return

    print("Creating Dataset")
    n_train = 3000
    batch_size = 50
    dataset = create_data([lorenz, 3, 0.01], n_train=n_train, n_test=100, n_val=100, n_trans=0)
    train_list = [dataset[0], dataset[1]]
    val_list = [dataset[2], dataset[3]]
    test_list = [dataset[4], dataset[5]]

    train_data = TensorDataset(*train_list)
    val_data = TensorDataset(*val_list)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    print("Mini-batch: ", len(dataloader), dataloader.batch_size)

    model = FNO(
        in_channels=3,
        out_channels=3,
        num_fno_modes=3,
        padding=4,
        dimension=1,
        latent_channels=128
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
    num_epochs = 2000
    time_step = 0.01
    jac_diff_train, jac_diff_test = torch.empty(n_store+1), torch.empty(n_store+1)
    print("Computing analytical Jacobian")
    t = torch.linspace(0, time_step, 2).cuda()
    threshold = 0.001
    f = lambda x: torchdiffeq.odeint(lorenz, x, t, method="rk4")[1]
    torch.cuda.empty_cache()
    timer = Timer()
    elapsed_time_train = []

    if loss_type == "JAC":
        true_jac_fn = torch.func.jacrev(f)
        True_j = torch.zeros(n_train, 3, 3)
        for j in range(n_train):
            True_j[j] = true_jac_fn(train_list[0][j])
        True_J = True_j.reshape(len(dataloader), dataloader.batch_size, 3, 3).cuda()

        print("Sanity Check: \n", True_j[0], True_j[batch_size], True_j[2*batch_size], True_j[3*batch_size])
        print("True: ", True_J[0:4, 0])
    
    print("Beginning training")
    for epoch in range(num_epochs):
        start_time = time.time()
        full_loss = 0.0
        idx = 0
        for data in dataloader:
            optimizer.zero_grad()
            y_true = data[1].to('cuda')
            y_pred = model(data[0].unsqueeze(dim=2).to('cuda'))

            # MSE Loss
            loss_mse = criterion(y_pred.view(batch_size, -1), y_true.view(batch_size, -1))
            loss = loss_mse / torch.norm(y_true, p=2)
            
            if loss_type == "JAC":
                with timer:
                    jac = torch.func.jacrev(model)
                    x = data[0].unsqueeze(dim=2).to('cuda')
                    cur_model_J = jac(x)
                    squeezed_J = cur_model_J[:, :, 0, :, :, 0]
                    #print("non-zero element", torch.count_nonzero(squeezed_J)) -> 180 -> 20 x 3 x 3 not 3600 with would mean that it is actually computing 20 x 20 x 3 x 3
                    # print("example", squeezed_J[:, :, 0, :], squeezed_J[0, :, 0, :], squeezed_J[:, :, 0, :].shape)
                    learned_J = [squeezed_J[in_out_pair, :, in_out_pair, :] for in_out_pair in range(batch_size)]
                    learned_J = torch.stack(learned_J, dim=0).cuda()

                    jac_norm_diff = criterion(True_J[idx], learned_J)
                    reg_param = 500.0
                    loss += (jac_norm_diff / torch.norm(True_J[idx]))*reg_param
                        
            full_loss += loss
            idx += 1
            end_time = time.time()  
            elapsed_time_train.append(end_time - start_time)
            
        full_loss.backward()
        optimizer.step()
        print("epoch: ", epoch, "loss: ", full_loss)
        if full_loss < threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")
    model_size = model_size(model)
    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_{loss_type}.pth")

    if loss_type == "JAC":
        with open('../test_result/Time/Modulus_FNO_elapsed_times_Jacobian.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Elapsed Time (seconds)'])
            for epoch, elapsed_time in enumerate(timer.elapsed_times, 1):
                writer.writerow([epoch, elapsed_time])
    with open('../test_result/Time/Modulus_FNO_epoch_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Elapsed Time (seconds)'])
        for epoch, elapsed_time in enumerate(elapsed_time_train, 1):
            writer.writerow([epoch, elapsed_time])
    

    print("Creating plot...")
    phase_path = f"../plot/Phase_plot/FNO_Modulus_{loss_type}.png"
    plot_attractor(model, [lorenz, 3, 0.01], 50, phase_path)

    # compute LE
    torch.cuda.empty_cache()
    dim = 3
    init = torch.randn(dim)
    true_traj = torchdiffeq.odeint(lorenz, torch.randn(dim), torch.arange(0, 50, 0.01), method='rk4', rtol=1e-8)

    init_point = torch.randn(dim)
    learned_traj = torch.empty_like(true_traj).cuda()
    learned_traj[0] = init_point
    for i in range(1, len(learned_traj)):
        learned_traj[i] = model(learned_traj[i-1].reshape(1, dim, 1).cuda()).reshape(dim,)
    
    print("Computing LEs of NN...")
    learned_LE = lyap_exps([model, dim, 0.01], "lorenz", learned_traj, true_traj.shape[0], batch_size).detach().cpu().numpy()
    print("Computing true LEs...")
    True_LE = lyap_exps([lorenz, dim, 0.01], "lorenz", true_traj, true_traj.shape[0], batch_size).detach().cpu().numpy()

    print("Computing rest of metrics...")
    True_mean = torch.mean(true_traj, dim = 0)
    Learned_mean = torch.mean(learned_traj, dim = 0)
    True_var = torch.var(true_traj, dim = 0)
    Learned_var = torch.var(learned_traj, dim=0)

    logger.info("%s: %s", "Model Size", str(model_size))
    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss))
    logger.info("%s: %s", "Learned LE", str(learned_LE))
    logger.info("%s: %s", "True LE", str(True_LE))
    logger.info("%s: %s", "Learned mean", str(Learned_mean))
    logger.info("%s: %s", "True mean", str(True_mean))

if __name__ == "__main__":

    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_JAC_{start_time}.txt")
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()

    # call main
    main(logger, "JAC")
