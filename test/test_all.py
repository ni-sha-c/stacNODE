import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
import datetime
import numpy as np
import argparse
import logging
import os
from matplotlib.pyplot import *

import sys
sys.path.append('..')

from dyn_sys.dim1 import *
from dyn_sys.dim2 import *
from dyn_sys.dim3 import *
from dyn_sys.dim4 import *
from dyn_sys.KS import *
# from ..src import jacobian


########################
### Dynamical System ###
########################
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ODE_MLP(nn.Module):
    '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''

    def __init__(self, y_dim=3, n_hidden=512, n_layers=2):
        super(ODE_MLP, self).__init__()
        layers = [nn.Linear(y_dim, n_hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.GELU()])
        layers.append(nn.Linear(n_hidden, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        res = self.net(y)
        return res
    

class ODE_MLP_skip(nn.Module):
    def __init__(self, y_dim=3, n_hidden=512, n_layers=5):
        super(ODE_MLP_skip, self).__init__()
        layers = [nn.Linear(y_dim, n_hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.skip = nn.Sequential(
            nn.Linear(y_dim, n_hidden),
            nn.ReLU(),
        )
        self.output = nn.Linear(n_hidden, y_dim)
    
    def forward(self, t, y):
        res = self.net(y) + self.skip(y)
        return self.output(res)   


##############
## Training ##
##############

def simulate_map(model, s, dim, n_iter, x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    traj = torch.zeros(n_iter, dim).to(device)
    for i in range(n_iter):
        traj[i] = x
        if (model == tilted_tent_map) or (model == pinched_tent_map) or (model == plucked_tent_map) or (model == baker):
            cur_pred = model(x, s)
        else:
            cur_pred = model(0, x.cuda())
        x = cur_pred
    return traj

def create_data(dyn_info, s, n_train, n_test, n_val, n_trans):
    dyn, dim, time_step = dyn_info

    # Adjust total time to account for the validation set
    tot_time = time_step * (n_train + n_test + n_val + n_trans + 1)
    t_eval_point = torch.arange(0, tot_time, time_step)

    if (dim == 1) or (dim == 2):
        traj = simulate_map(dyn, s, dim, t_eval_point.shape[0], torch.randn(dim))[n_trans:].requires_grad_(True)
    else:
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

    print("created data", torch.max(X_train, axis=0), torch.min(X_train, axis=0))

    return [X_train, Y_train, X_val, Y_val, X_test, Y_test]

def calculate_relative_error(model, dyn, dim, device):
    # Simulate an orbit using the true dynamics
    time_step = 0.01  # Example timestep, adjust as needed
    orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8).to(device)
    
    # Compute vector field from model and true dynamics
    vf_nn = model(0, orbit).detach()
    vf_true = torch.stack([dyn(0, orbit[i]) for i in range(orbit.size(0))])

    # Calculate relative error
    err = torch.linalg.norm(vf_nn - vf_true, dim=1)
    mag = torch.linalg.norm(vf_true, dim=1)
    relative_error = torch.mean(err / mag).item() * 100  # As percentage
    return relative_error

def update_lr(optimizer, epoch, total_e, origin_lr):
    """ A decay factor of 0.1 raised to the power of epoch / total_epochs. Learning rate decreases gradually as the epoch number increases towards the total number of epochs. """
    new_lr = origin_lr * (0.1 ** (epoch / float(total_e)))
    for params in optimizer.param_groups:
        params['lr'] = new_lr
    return

def define_dyn_sys(dyn_sys):
    DYNSYS_MAP = {'tilted_tent_map' : [tilted_tent_map, 1],
                  'pinched_tent_map' : [pinched_tent_map, 1],
                  'plucked_tent_map' : [plucked_tent_map, 1],
                  'KS': [run_KS, 127],
                  'baker' : [baker, 2],
                  'lorenz' : [lorenz, 3],
                  'rossler' : [rossler, 3],
                  'hyperchaos': [hyperchaos, 4]}
    dyn_sys_info = DYNSYS_MAP[dyn_sys]
    dyn_sys_func, dim = dyn_sys_info
    return dyn_sys_func, dim

def find_ds_name(dyn_sys_func):
    DYNSYS_NAME_MAP = {tilted_tent_map : "tilted_tent_map",
                        pinched_tent_map : "pinched_tent_map",
                        plucked_tent_map : "plucked_tent_map",
                        run_KS : "KS",
                        baker : "baker",
                        lorenz : "lorenz",
                        rossler : "rossler",
                        hyperchaos : "hyperchaos"}
    return DYNSYS_NAME_MAP[dyn_sys_func]

def vectorized_simulate(model, X, t_eval_point, len_T, device):
    torch.cuda.empty_cache()
    integrated_model = lambda x: one_step_rk4(model, x, t_eval_point).to(device)
    compute_batch = torch.func.vmap(integrated_model, in_dims=(0), chunk_size=2000)
    
    traj = torch.zeros(len_T, X.shape[0], X.shape[1]) # len_T x num_init x dim
    traj[0] = X
    for i in range(1, len_T):
        traj[i] = compute_batch(X.double().to(device)).detach() 
        X = traj[i]
    return traj

def train(dyn_sys_info, model, device, dataset, optim_name, criterion, epochs, lr, weight_decay, reg_param, loss_type, model_type, s):

    # Initialize
    n_store, k  = 100, 0
    ep_num, loss_hist, test_loss_hist = torch.empty(n_store+1,dtype=int), torch.empty(n_store+1), torch.empty(n_store+1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = X_train.to(device), Y_train.to(device), X_val.to(device), Y_val.to(device), X_test.to(device), Y_test.to(device)
    num_train = X_train.shape[0]
    dyn_sys, dim, time_step = dyn_sys_info
    ds_name = find_ds_name(dyn_sys)
    idx = 1 if ds_name == "lorenz" else 2
    t_eval_point = torch.linspace(0, time_step, 2).to(device)
    torch.cuda.empty_cache()
    
    # Compute True Jacobian
    if loss_type == "Jacobian":
        jac_diff_train, jac_diff_test = torch.empty(n_store+1), torch.empty(n_store+1)
        print("Jacobian loss!")
        f = lambda x: dyn_sys(0, x)
        true_jac_fn = torch.vmap(torch.func.jacrev(f))
        True_J = true_jac_fn(X_train)
        Test_J = true_jac_fn(X_test)

    # Training Loop
    min_relative_error = 1000000
    for i in range(epochs):
        model.train()
        if (dim == 1) or (dim == 2):
            y_pred = model(0, X_train).to(device)
        else: 
            y_pred = torchdiffeq.odeint(model, X_train, t_eval_point, method="rk4")[-1].to(device)
        optimizer.zero_grad()
        train_loss = criterion(y_pred, Y_train)  * (1/time_step/time_step)

        if loss_type == "Jacobian":
            # Compute Jacobian
            if (dim == 1) or (dim == 2):
                jacrev = torch.func.jacrev(model)
                compute_batch_jac = torch.vmap(jacrev, chunk_size=1000)
                cur_model_J = compute_batch_jac(X).to(device)
            elif (ds_name == "KS"):
                integrated_model = lambda x: rk4(x, model, t_eval_point).to(device)
                jacrev = torch.func.jacrev(integrated_model)
                compute_batch_jac = torch.func.vmap(jacrev, in_dims=(0), chunk_size=500)
                cur_model_J = compute_batch_jac(X).to(device)
            else:
                jacrev = torch.func.jacrev(model, argnums=1)
                compute_batch_jac = torch.vmap(jacrev, in_dims=(None, 0), chunk_size=1000)
                cur_model_J = compute_batch_jac(0, X_train).to(device)
            jac_norm_diff = criterion(True_J, cur_model_J)
            train_loss += reg_param*jac_norm_diff

        train_loss.backward()
        optimizer.step()
        update_lr(optimizer, i, epochs, args.lr)

        # Save Training and Test History
        if i % (epochs//n_store) == 0 or (i == epochs-1):
            with torch.no_grad():
                model.eval()
                if (dim == 1) or (dim == 2):
                    y_pred_test = model(0, X_test).to(device)
                else: 
                    y_pred_test = torchdiffeq.odeint(model, X_test, t_eval_point, rtol=1e-9, atol=1e-9, method="rk4")[-1].to(device)
                test_loss = criterion(y_pred_test, Y_test) * (1/time_step/time_step)

                if (dim == 1) or (dim == 2):
                    current_relative_error = test_loss
                else:
                    current_relative_error = calculate_relative_error(model, dyn_sys_info[0], dim, device)
                # Check if current model has the lowest relative error so far
                if current_relative_error < min_relative_error:
                    min_relative_error = current_relative_error
                    # Save the model
                    torch.save(model.state_dict(), f"{args.train_dir}/best_model.pth")
                    logger.info(f"Epoch {i}: New minimal relative error: {min_relative_error:.2f}%, model saved.")

                # save predicted node feature for analysis            
                logger.info("Epoch: %d Train: %.5f Test: %.5f", i, train_loss.item(), test_loss.item())
                ep_num[k], loss_hist[k], test_loss_hist[k] = i, train_loss.item(), test_loss.item()

                if loss_type == "Jacobian":
                    test_model_J = compute_batch_jac(0, X_test).to(device)
                    test_jac_norm_diff = criterion(Test_J, test_model_J)
                    jac_diff_train[k], jac_diff_test[k] = jac_norm_diff, test_jac_norm_diff
                    if (dim != 1) and (dim != 2):
                        JAC_plot_path = f'{args.train_dir}JAC_'+str(i)+'.jpg'
                        plot_vector_field(model, dim, path=JAC_plot_path, idx=idx, t=0., N=100, device='cuda')
                k = k + 1

    if loss_type == "Jacobian":
        for i in [0, 1, 50, -2, -1]:
            print("Point:", X_train[i].detach().cpu().numpy(), "\n", "True:", "\n", True_J[i].detach().cpu().numpy(), "\n", "JAC:", "\n", cur_model_J[i].detach().cpu().numpy())
    else:
        MSE_plot_path = f'{args.train_dir}MSE_'+str(i)+'.jpg'
        if (dim == 1) or (dim == 2):
            plot_map(model, dim, s, 1000, torch.randn(dim), MSE_plot_path)
        else:
            plot_vector_field(model, ds_name, dim, path=MSE_plot_path, idx=idx, t=0., N=100, device='cuda')
        jac_diff_train, jac_diff_test = None, None
    # Load the best relative error model
    best_model = model
    best_model.load_state_dict(torch.load(f"{args.train_dir}/best_model.pth"))
    best_model.eval()
    RE_plot_path = f'{args.train_dir}minRE.jpg'
    if (dim == 1) or (dim == 2):
        plot_map(model, dim, s, 1000, torch.randn(dim), RE_plot_path)
    else:
        plot_vector_field(best_model, ds_name, dim, path=RE_plot_path, idx=idx, t=0., N=100, device='cuda')
    return ep_num, loss_hist, test_loss_hist, jac_diff_train, jac_diff_test, Y_test



##############
#### Plot ####
##############

def plot_loss(epochs, train, test, path):
    print("plotting loss!", path)
    fig, ax = subplots()
    ax.plot(epochs[30:].numpy(), train[30:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Train")
    ax.plot(epochs[30:].numpy(), test[30:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Test")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    savefig(path, bbox_inches ='tight', pad_inches = 0.1)

def plot_map(model, dim, s, n_iter, init_state, pdf_path):
    fig, ax = subplots(figsize=(24,13))
    colors = cm.viridis(np.linspace(0, 1, 5))
    whole_traj = simulate_map(model, s, dim, n_iter, torch.randn(dim)).detach().cpu().numpy()
    ax.scatter(whole_traj[0:-1], whole_traj[1:], color=colors[0], linewidth=6, alpha=0.8, label='s = ' + str(s))
    ax.set_xlabel(r"$x_n$", fontsize=44)
    ax.set_ylabel(r"$x_{n+1}$", fontsize=44)
    ax.tick_params(labelsize=40)
    ax.legend(loc='best', fontsize=40)
    tight_layout()
    fig.savefig(pdf_path, format='jpg', dpi=400)
    return

def plot_attractor(model, dyn_info, time, path):
    print("plotting attractor!", path)
    # generate true orbit and learned orbit
    dyn, dim, time_step = dyn_info
    tran_orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    true_o = torchdiffeq.odeint(dyn, tran_orbit[-1], torch.arange(0, time, time_step), method='rk4', rtol=1e-8)
    learned_o = torchdiffeq.odeint(model.eval().to(device), tran_orbit[-1].to(device), torch.arange(0, time, time_step), method="rk4", rtol=1e-8).detach().cpu().numpy()

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

def plot_vf_err(model, dyn_info, model_type, loss_type):
    dyn, dim, time_step = dyn_info
    dyn_sys_type = "lorenz" if dyn == lorenz else "rossler"

    orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    orbit = torchdiffeq.odeint(dyn, orbit[-1], torch.arange(0, 20, time_step), method='rk4', rtol=1e-8)
    len_o = orbit.shape[0]

    vf_nn = model(0, orbit.to('cuda')).detach().cpu()
    vf = torch.zeros(len_o, dim)
    for i in range(len_o):
        vf[i] = dyn(0,orbit[i])
    vf_nn, vf = vf_nn.T, vf.T
    ax = figure().add_subplot()
    vf_nn, vf = vf_nn.numpy(), vf.numpy()
    mag = np.linalg.norm(vf, axis=0)
    err = np.linalg.norm(vf_nn - vf, axis=0)
    t = time_step*np.arange(0, len_o)
    percentage_err = err/mag*100
    
    ax.plot(t, percentage_err, "o", label=r"$\frac{\|\hat x - x\|_2}{\|x\|_2}$", ms=3.0)
    np.savetxt(f'{args.train_dir}{args.loss_type}error_attractor.txt', np.column_stack((t, err/mag*100)), fmt='%.6f')
    ax.set_xlabel("time",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.set_ylim(0, int(max(percentage_err)) + 1)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    path = f"../plot/Relative_error/{dyn_sys_type}/{args.model_type}_{args.loss_type}_{dyn_sys_type}.png"
    savefig(path)
    return percentage_err

def plot_vf_err_test(model, y_pred_train, dyn_info, model_type, loss_type):
    dyn, dim, time_step = dyn_info
    dyn_sys_type = "lorenz" if dyn == lorenz else "rossler"
    orbit = y_pred_train
    len_o = orbit.shape[0]
    orbit_gpu = orbit.to('cuda')
    vf_nn = model(0, orbit_gpu).detach().cpu()
    vf = torch.zeros(len_o, dim)
    # for i in range(len_o):
    true_vf = lambda x: dyn(0,x)
    vf = torch.vmap(true_vf)(orbit_gpu).detach().cpu()
    vf_nn, vf = vf_nn.T, vf.T
    ax = figure().add_subplot()
    vf_nn, vf = vf_nn.numpy(), vf.numpy()
    mag = np.linalg.norm(vf, axis=0)
    # mag = abs(vf[2])
    err = np.linalg.norm(vf_nn - vf, axis=0)
    # err = abs(vf_nn[2]-vf[2])
    t = time_step*np.arange(0, len_o)
    ax.plot(t, err/mag*100, "o", label=r"$\|Error\|_2$", ms=3.0)
    np.savetxt(f'{args.train_dir}{args.loss_type}error_test.txt', np.column_stack((t, err/mag*100)), fmt='%.6f')
    ax.set_xlabel("time",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.set_ylim(0, int(max(err/mag*100)) + 1)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()

    path = f"{args.train_dir}MSE_error_Ytest.png"
    savefig(path)

def plot_vector_field(model, d_name, dim, path, idx, t, N, device='cuda'):
    # Credit: https://torchdyn.readthedocs.io/en/latest/_modules/torchdyn/utils.html

    if dim == 3:
        if d_name == "lorenz":
            x = torch.linspace(-20, 20, N)
            y = torch.linspace(-25, 25, N)
        else:
            x = torch.linspace(-10, 10, N)
            y = torch.linspace(0, 20, N)
    elif dim == 4:
        x = torch.linspace(-150, 150, N)
        y = torch.linspace(-150, 150, N)
    else:
        x = torch.linspace(-50, 50, N)
        y = torch.linspace(-50, 50, N)
    X, Y = torch.meshgrid(x,y)
    Z_random = torch.randn(1)*10
    U, V = np.zeros((N,N)), np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if dim == 3:
                if idx == 1:
                    # lorenz
                    phi = torch.stack([X[i,j], Y[i,j], torch.tensor(20.)]).to('cuda')
                else:
                    # rossler
                    phi = torch.stack([X[i,j], torch.tensor(-2.), Y[i,j]]).to('cuda')
                
                O = model(0., phi).detach().cpu().numpy()
                if O.ndim == 1:
                    U[i,j], V[i,j] = O[0], O[idx]
                else:
                    U[i,j], V[i,j] = O[0, 0], O[0, idx]
            elif dim == 4:
                # z-w plane
                phi = torch.stack([torch.tensor(20.), torch.tensor(20.), X[i,j], Y[i,j]]).to('cuda')
                O = model(0., phi).detach().cpu().numpy()
                if O.ndim == 1:
                    U[i,j], V[i,j] = O[2], O[3]
                else:
                    U[i,j], V[i,j] = O[0, 2], O[0, 3]

    fig = figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    contourf = ax.contourf(X, Y, np.sqrt(U**2 + V**2), cmap='jet')
    ax.streamplot(X.T.numpy(),Y.T.numpy(),U.T,V.T, color='k')
    ax.set_xlim([x.min(),x.max()])
    ax.set_ylim([y.min(),y.max()])
    ax.set_xlabel(r"$x$", fontsize=17)
    if idx == 1:
        ax.set_ylabel(r"$y$", fontsize=17)
    else:
        ax.set_ylabel(r"$z$", fontsize=17)
    ax.xaxis.set_tick_params(labelsize=17)
    ax.yaxis.set_tick_params(labelsize=17)
    fig.colorbar(contourf)
    tight_layout()
    savefig(path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    close()
    return

def rk4(x, f, dt):
    k1 = f(0, x)
    k2 = f(0, x + dt*k1/2)
    k3 = f(0, x + dt*k2/2)
    k4 = f(0, x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
def lyap_exps(dyn_sys_info, s, traj, iters):
    dyn_type, model, dim, time_step = dyn_sys_info
    LE = torch.zeros(dim).to(device)
    traj_gpu = traj.to(device)

    if (dim == 1) or (dim == 2):
        le = 0
        for t in range(traj_gpu.shape[0]):
            if (model == tilted_tent_map) or (model == plucked_tent_map) or (model == pinched_tent_map) or (model == baker):
                le += torch.log(abs(F.jacobian(lambda x: model(x, s), traj_gpu[t])))
            else:
                le += torch.log(abs(F.jacobian(lambda x: model(0, x), traj_gpu[t])))
        return le/traj_gpu.shape[0]
    else:
        f = lambda x: rk4(x, model, time_step)
        Jac = torch.vmap(torch.func.jacrev(f))(traj_gpu)
        if dyn_type == "KS":
            lower_dim = 15
            Q = torch.eye(*(dim, lower_dim))
        else:
            Q = torch.rand(dim,dim).to(device)
        # eye_cuda = torch.eye(dim).to(device)
        for i in range(iters):
            if i > 0 and i % 1000 == 0:
                print("Iteration: ", i, ", LE[0]: ", LE[0].detach().cpu().numpy()/i/time_step)
            Q = torch.matmul(Jac[i], Q)
            Q, R = torch.linalg.qr(Q)
            LE += torch.log(abs(torch.diag(R)))
        return LE/iters/time_step


if __name__ == '__main__':

    # Set device
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=8000)
    parser.add_argument("--num_val", type=int, default=3000)
    parser.add_argument("--num_trans", type=int, default=0)
    parser.add_argument("--loss_type", default="Jacobian", choices=["Jacobian", "MSE"])
    parser.add_argument("--dyn_sys", default="rossler", choices=["lorenz", "rossler", "baker", "tilted_tent_map", "plucked_tent_map", "pinched_tent_map", "KS", "hyperchaos"])
    parser.add_argument("--model_type", default="MLP_skip", choices=["MLP","MLP_skip"])
    parser.add_argument("--s", type=int, default=0.5)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=15)
    parser.add_argument("--reg_param", type=float, default=1000)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--train_dir", default="../plot/Vector_field/rossler/train_MLPskip_Jacobian_fullbatch/")

    # Initialize Settings
    args = parser.parse_args()
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    dyn_sys_func, dim = define_dyn_sys(args.dyn_sys)
    dyn_sys_info = [dyn_sys_func, dim, args.time_step]
    criterion = torch.nn.MSELoss()
    if args.dyn_sys == "KS":
        L = 128 # L = [128, 256, 512, 700]
        n = L-1 # num of internal node
        T = 1501
        c = 0.4
        dx = L/(n+1)
        dt = args.time_step
        x = torch.arange(0, L+dx, dx) # [0, 0+dx, ... 128] shape: L + 1
        u0 = 2.71828**(-(x-64)**2/512).to(device).double().requires_grad_(True) 

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"{start_time}_{args.model_type}_{args.loss_type}_{args.dyn_sys}.txt")
    print("saved in ", out_file)
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Create Dataset
    if args.dyn_sys == "KS":
        # boundary condition
        u0[0], u0[-1] = 0, 0 
        u0 = u0.requires_grad_(True)
        # Generate Training/Test/Multi-Step Prediction Data
        torch.cuda.empty_cache()
        u_list = run_KS(u0, c, dx, dt, T, False, device)
        u_list = u_list[:, 1:-1] # remove the last boundary node and keep the first boundary node as it is initial condition
    else:
        dataset = create_data(dyn_sys_info, args.s, n_train=args.num_train, n_test=args.num_test, n_trans=args.num_trans, n_val=args.num_val)

    # Create model
    if args.model_type == "MLP":
        m = ODE_MLP(y_dim=dim, n_hidden=args.n_hidden, n_layers=args.n_layers).to(device)
    elif args.model_type == "MLP_skip":
        m = ODE_MLP_skip(y_dim=dim, n_hidden=args.n_hidden).to(device)

    print("Training...") # Train the model, return node
    epochs, loss_hist, test_loss_hist, jac_train_hist, jac_test_hist, Y_test = train(dyn_sys_info, m, device, dataset, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.reg_param, args.loss_type, args.model_type, args.s)

    # Plot Loss
    loss_path = f"../plot/Loss/{args.dyn_sys}/Total/{args.model_type}_{args.loss_type}_Total_{start_time}.png"
    jac_loss_path = f"../plot/Loss/{args.dyn_sys}/JAC/{args.model_type}_{args.loss_type}_Jacobian_matching_{start_time}.png"
    mse_loss_path = f"../plot/Loss/{args.dyn_sys}/MSE/{args.model_type}_{args.loss_type}_MSE_part_{start_time}.png"
    true_plot_path_1 = f"../plot/Vector_field/{args.dyn_sys}/True_{args.dyn_sys}_1.png"
    true_plot_path_2 = f"../plot/Vector_field/{args.dyn_sys}/True_{args.dyn_sys}_2.png"
    phase_path = f"../plot/Phase_plot/{args.dyn_sys}_{args.model_type}_{args.loss_type}.png"

    plot_loss(epochs, loss_hist, test_loss_hist, loss_path) 
    if args.loss_type == "Jacobian":
        plot_loss(epochs, jac_train_hist, jac_test_hist, jac_loss_path) 
        plot_loss(epochs, abs(loss_hist - args.reg_param*jac_train_hist)*(args.time_step)**2, abs(test_loss_hist - args.reg_param*jac_test_hist)*(args.time_step)**2, mse_loss_path) 

    # Plot vector field & phase space
    if (dim != 1) and (dim != 2):
        percentage_err = plot_vf_err(m, dyn_sys_info, args.model_type, args.loss_type)
        plot_vf_err_test(m, Y_test, dyn_sys_info, args.model_type, args.loss_type)
        plot_vector_field(dyn_sys_func, args.dyn_sys, dim, path=true_plot_path_1, idx=1, t=0., N=100, device='cuda')
        plot_vector_field(dyn_sys_func, args.dyn_sys, dim, path=true_plot_path_2, idx=2, t=0., N=100, device='cuda')
        plot_attractor(m, dyn_sys_info, 50, phase_path)
    else:
        plot_map(m, dim, args.s, 3000, torch.randn(dim), phase_path)
        plot_map(dyn_sys_func, dim, args.s, 3000, torch.randn(dim), true_plot_path_1)

    # compute LE
    if (dim == 1) or (dim == 2):
        print("dim is 1 or 2")
        true_traj = simulate_map(dyn_sys_func, args.s, dim, 1000, torch.randn(dim))
    else:
        print("dim is bigger than 2")
        rand_x = torch.randn(dim).cuda()
        print("x", rand_x)
        true_traj = torchdiffeq.odeint(dyn_sys_func, rand_x, torch.arange(0, 300, args.time_step), method='rk4', rtol=1e-8)
    print("Computing LEs of NN...")
    learned_LE = lyap_exps([args.dyn_sys, m, dim, args.time_step], args.s, true_traj, true_traj.shape[0]).detach().cpu().numpy()
    print("Computing true LEs...")
    True_LE = lyap_exps([args.dyn_sys, dyn_sys_func, dim, args.time_step], args.s, true_traj, true_traj.shape[0]).detach().cpu().numpy()

    logger.info("%s: %s", "Training Loss", str(loss_hist[-1]))
    logger.info("%s: %s", "Test Loss", str(test_loss_hist[-1]))
    if args.loss_type == "Jacobian":
        logger.info("%s: %s", "Jacobian term Training Loss", str(jac_train_hist[-1]))
        logger.info("%s: %s", "Jacobian term Test Loss", str(jac_test_hist[-1]))
    logger.info("%s: %s", "Learned LE", str(learned_LE))
    logger.info("%s: %s", "True LE", str(True_LE))
    if (dim != 1) and (dim != 2):
        logger.info("%s: %s", "Relative Error", str(percentage_err))
    print("Learned:", learned_LE, "\n", "True:", True_LE)
