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

# Set seed
# torch.manual_seed(42)
########################
### Dynamical System ###
########################

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def rossler(t, X):
    '''Parameter values picked from: The study of Lorenz and Rössler strange attractors by means of quantum theory by Bogdanov et al.
    https://arxiv.org/ftp/arxiv/papers/1412/1412.2242.pdf
    LE:  0.07062, 0.000048, -5.3937
    '''
    x, y, z = X
    a = 0.2
    b = 0.2
    c = 5.7
    
    dx = -(y + z)
    dy = x + a * y
    dz = b + z * (x - c)
    return torch.stack([dx, dy, dz])

# class ODE_MLP(nn.Module):
#     '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''

#     def __init__(self, y_dim=3, n_hidden=512, n_layers=2):
#         super(ODE_MLP, self).__init__()
#         layers = [nn.Linear(y_dim, n_hidden), nn.GELU()]
#         for _ in range(n_layers - 1):
#             layers.extend([nn.Linear(n_hidden, n_hidden), nn.GELU()])
#         layers.append(nn.Linear(n_hidden, y_dim))
#         self.net = nn.Sequential(*layers)

#     def forward(self, t, y):
#         res = self.net(y)
#         return res
class ODE_MLPmse(nn.Module):
    def __init__(self, y_dim=3, n_hidden=256):
        super(ODE_MLPmse, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim, n_hidden),
            nn.ReLU(),
            
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            
            nn.Linear(n_hidden, n_hidden*2),
            nn.ReLU(),
            
            # nn.Linear(n_hidden*2, n_hidden*2),
            # nn.ReLU(),
            
            nn.Linear(n_hidden*2, n_hidden),
            nn.ReLU(),
           
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.skip = nn.Sequential(
            nn.Linear(y_dim, n_hidden),
            nn.ReLU(),
        )
        self.output = nn.Linear(n_hidden, y_dim)
    
    def forward(self, t, y):
        res = self.net(y) + self.skip(y)
        return self.output(res)
class ODE_MLP1(nn.Module):
    def __init__(self, y_dim=3, n_hidden=512):
        super(ODE_MLP1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),  # Apply dropout after activation
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),  # Apply dropout after activation
            nn.Linear(n_hidden, n_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Apply dropout after activation
            nn.Linear(n_hidden*2, n_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Apply dropout after activation
            nn.Linear(n_hidden*2, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),  # Apply dropout after activation
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),  # Apply dropout after activation
        )
        self.skip = nn.Sequential(
            nn.Linear(y_dim, n_hidden),
            nn.ReLU(),
            # Typically skip connections do not have dropout.
            # But if you wish to add, uncomment the next line.
            # nn.Dropout(0.2),
        )
        self.output = nn.Linear(n_hidden, y_dim)
    
    def forward(self, t, y):
        res = self.net(y) + self.skip(y)
        # print('shape of res:', self.output(res).shape)
        return self.output(res)


class ODE_MLP(nn.Module):
    def __init__(self, y_dim=3, n_hidden=256):
        super(ODE_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim, n_hidden),
            nn.ReLU(),
            
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            
            # nn.Linear(n_hidden, n_hidden*2),
            # nn.ReLU(),
            
            # nn.Linear(n_hidden*2, n_hidden*2),
            # nn.ReLU(),
            
            # nn.Linear(n_hidden*2, n_hidden),
            # nn.ReLU(),
           
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.skip = nn.Sequential(
            nn.Linear(y_dim, n_hidden),
            nn.ReLU(),
        )
        self.output = nn.Linear(n_hidden, y_dim)
    
    def forward(self, t, y):
        res = self.net(y) + self.skip(y)
        return self.output(res)
    # def forward(self, t, y):
    #     return self.net(y)

import torch
import torch.nn as nn
import torchdiffeq

class Func(nn.Module):
    def __init__(self,  y_dim=3, n_hidden=256, depth=2):
        super(Func, self).__init__()
        # Example architecture, adjust as needed
        layers = [nn.Linear(y_dim, n_hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, y_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        return self.net(y)  # Compute dy/dt given y and potentially t

class NeuralODE(nn.Module):
    def __init__(self,  y_dim=3, n_hidden=256, depth=2):
        super(NeuralODE, self).__init__()
        self.func = Func(y_dim, n_hidden, depth)

    def forward(self, ts, y0):
        solution = torchdiffeq.odeint(
            self.func,
            y0,
            torch.linspace(0, 10, 101),
            method='dopri5',  
        )
        # print('shape of sol node:', solution[-1,:,:].shape)
        if solution.dim() == 2:
            output = solution[-1, :].squeeze(0)
            print('shape of sol:', output.shape)
        elif solution.dim() == 3:
            output = solution[-1, :, :].squeeze(0)
            print('shape of sol:', output.shape)
        else:
            print('shape of sol:', solution.shape)
            raise ValueError("Invalid solution shape")
        return output

class ODE_HigherDim_CNN(nn.Module):

    def __init__(self, y_dim=3, n_hidden=512):
        super(ODE_HigherDim_CNN, self).__init__()
        self.conv2d = nn.Conv2d(1, 1, kernel_size=(5,5), padding=2, bias=False)
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, 3)
        )
        init = torch.randn(3, 3)
        self.matrix = nn.Parameter(init)

    def forward(self, t, y):
        if y.dim() == 1: # needed for vmap
            y = y.reshape(1, -1)
        
        y = torch.matmul(self.matrix, y.T)
        y = torch.unsqueeze(y, 0)
        y = self.conv2d(y)
        y = self.conv2d(y)
        y = self.conv2d(y)
        y = self.net(y.squeeze().T)
        return y

class ODE_CNN(nn.Module):

    def __init__(self, y_dim=3, n_hidden=512):
        super(ODE_CNN, self).__init__()
        self.conv1d = nn.Conv1d(3, 3, kernel_size=3, padding=1, bias=False)
        self.conv2d = nn.Conv2d(1, 1, kernel_size=(5,5), padding=2, bias=False)
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, 3)
        )

    def forward(self, t, y):
        if y.dim() == 1: # needed for vmap
            y = y.reshape(1, -1)
        
        y = torch.unsqueeze(y.T, 0)
        y = self.conv2d(y)
        y = self.conv2d(y)
        y = self.conv2d(y)
        y = self.net(y.squeeze().T)
        return y

class ODE_CNN1(nn.Module):

    def __init__(self, y_dim=3, n_hidden=512):
        super(ODE_CNN1, self).__init__()
        self.conv1d = nn.Conv1d(3, 3, kernel_size=3, padding=1, bias=False)
        self.conv2d = nn.Conv2d(1, 1, kernel_size=(5,5), padding=2, bias=False)
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.GELU(),
            nn.Dropout(0.2),  # Apply dropout after activation
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Dropout(0.2),  # Apply dropout after activation
            nn.Linear(n_hidden, 3)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5,5), padding=2, bias=False),
            nn.GELU(),
            # Optionally, add dropout to the skip layer as well
            # nn.Dropout(0.2),
        )
        self.output = nn.Conv2d(1, 1, kernel_size=(5,5), padding=2, bias=False)

    def forward(self, t, y):
        if y.dim() == 1:  # needed for vmap
            y = y.reshape(1, -1)

        # Prepare the input for convolutional layers
        y = torch.unsqueeze(y.T, 0)

        # Skip connection
        skip_out = self.skip(y)

        # Convolutional layers with dropout
        y = self.conv2d(y)
        y = self.conv2d(y)
        y = self.conv2d(y)

        # Add the skip connection output to the result before the final layers
        y = y + skip_out

        # Flatten the output for the sequential linear layers
        y = self.net(y.squeeze().T)
        
        # Final output layer, if needed
        # y = self.output(y)

        return y


##############
## Training ##
##############

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

def total_variation_loss(y_pred):
    """
    Compute the Total Variation Loss for 1D vector field.
    y_pred: Tensor of shape [batch_size, n_points, feature_dim]
            This represents the predicted vector field.
    """
    # Calculate the difference between adjacent predictions
    tv_loss = torch.sum(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]))
    return tv_loss

def update_lr(optimizer, epoch, total_e, origin_lr):
    """ A decay factor of 0.1 raised to the power of epoch / total_epochs. Learning rate decreases gradually as the epoch number increases towards the total number of epochs. """
    new_lr = origin_lr * (0.1 ** (epoch / float(total_e)))
    for params in optimizer.param_groups:
        params['lr'] = new_lr
    return 


def calculate_relative_error(model, dyn, device):
    # Simulate an orbit using the true dynamics
    time_step = 0.01  # Example timestep, adjust as needed
    orbit = torchdiffeq.odeint(dyn, torch.randn(3), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    
    # Compute vector field from model and true dynamics
    vf_nn = model(0, orbit).detach()
    vf_true = torch.stack([dyn(0, orbit[i]) for i in range(orbit.size(0))])

    # Calculate relative error
    err = torch.linalg.norm(vf_nn - vf_true, dim=1)
    mag = torch.linalg.norm(vf_true, dim=1)
    relative_error = torch.mean(err / mag).item() * 100  # As percentage
    return relative_error

def train(dyn_sys_info, model, device, dataset, optim_name, criterion, epochs, lr, weight_decay, reg_param, loss_type, model_type, batch_size):
    # Initialize
    n_store, k = 10, 0
    ep_num, loss_hist, val_loss_hist, test_loss_hist = torch.empty(n_store + 1, dtype=int), torch.empty(n_store + 1), torch.empty(n_store + 1), torch.empty(n_store + 1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = X_train, Y_train, X_val, Y_val, X_test, Y_test
    dyn_sys, dim, time_step = dyn_sys_info
    
    dyn_sys_type = "lorenz" if dyn_sys == lorenz else "rossler"
    t_eval_point = torch.linspace(0, time_step, 2)
    torch.cuda.empty_cache()
    logger = logging.getLogger()
        # Compute True Jacobian
    if loss_type == "Jacobian":
        jac_diff_train, jac_diff_test = torch.empty(n_store+1), torch.empty(n_store+1)
        print("Jacobian loss!")
        f = lambda x: dyn_sys(0, x)
        true_jac_fn = torch.vmap(torch.func.jacrev(f))
        True_J = true_jac_fn(X_train)
        Test_J = true_jac_fn(X_test)

    # Learning rate scheduler (if needed)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    min_relative_error = 1000000
    # Training Loop
    for i in range(epochs):
        model.train()
        total_loss = 0.0
        # initial_condition = torch.randn(dim, requires_grad=True)
        # orbit = torchdiffeq.odeint(dyn, initial_condition, torch.arange(0, 9, time_step), method='rk4')

        # # Compute the model's vector field along the orbit
        # vf_nn = model(0, orbit)
        #for j in range(0, X_train.size(0), batch_size):
        X_batch = X_train #[j:j+batch_size]
        Y_batch = Y_train#[j:j+batch_size]

        optimizer.zero_grad()
        y_pred_train = torchdiffeq.odeint(model, X_batch, t_eval_point, method="rk4")[-1]
        y_pred_train = y_pred_train

        vf_nn = model(0, y_pred_train)
        vf_mag = torch.sqrt(vf_nn[:, 0]**2 + vf_nn[:, 1]**2)  # Assuming vf_nn is a 2D tensor representing (x, y) components
        # print('vf_mag', vf_mag)
        # # Penalize if the magnitude of the vector field exceeds 1200
        # vf_loss = torch.mean(torch.relu(vf_mag - 400))

            
        train_loss = criterion(y_pred_train, Y_batch) * (1 / time_step / time_step)
        
        # total_loss += train_loss + vf_loss
# def train(dyn_sys_info, model, device, dataset, optim_name, criterion, epochs, lr, weight_decay,  reg_param,loss_type, model_type):
#     # Initialize
#     n_store, k = 10, 0
#     ep_num, loss_hist, val_loss_hist, test_loss_hist = torch.empty(n_store + 1, dtype=int), torch.empty(n_store + 1), torch.empty(n_store + 1), torch.empty(n_store + 1)
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset
#     X_train, Y_train, X_val, Y_val, X_test, Y_test = X_train, Y_train, X_val, Y_val, X_test, Y_test
#     dyn, dim, time_step = dyn_sys_info
#     t_eval_point = torch.linspace(0, time_step, 2)
#     torch.cuda.empty_cache()
#     logger = logging.getLogger()

#     # Learning rate scheduler (if needed)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

#     # Training Loop
#     for i in range(epochs):
#         model.train()
#         # optimizer.zero_grad()
#         y_pred_train = torchdiffeq.odeint(model, X_train, t_eval_point, method="rk4")[-1]
#         y_pred_train = y_pred_train
#         optimizer.zero_grad()
#         train_loss = criterion(y_pred_train, Y_train) * (1 / time_step / time_step)
        # Adding L1 regularization
        # l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        # train_loss += reg_param * l1_loss

        # # Adding total variation regularization (if it makes sense for your model)
        # tv_loss = total_variation_loss(y_pred_train)
        # train_loss += reg_param * tv_loss
        if loss_type == "Jacobian":
            # Compute Jacobian
            jacrev = torch.func.jacrev(model, argnums=1)
            compute_batch_jac = torch.vmap(jacrev, in_dims=(None, 0), chunk_size=1000)
            cur_model_J = compute_batch_jac(0, X_train)
            jac_norm_diff = criterion(True_J, cur_model_J)
            train_loss += reg_param*jac_norm_diff
        train_loss.backward()
        optimizer.step()
        # scheduler.step()
        update_lr(optimizer, i, epochs, args.lr)
        # Validation phase
        model.eval()
        with torch.no_grad():
            y_pred_val = torchdiffeq.odeint(model, X_val, t_eval_point, method="rk4")[-1]
            val_loss = criterion(y_pred_val, Y_val) * (1 / time_step / time_step)

            # Test phase
            y_pred_test = torchdiffeq.odeint(model, X_test, t_eval_point, method="rk4")[-1]
            test_loss = criterion(y_pred_test, Y_test) * (1 / time_step / time_step)

        # Logging
        if i % (epochs // n_store) == 0 or i == epochs - 1:
            logger.info(f"Epoch: {i} Train Loss: {train_loss.item():.5f} Validation Loss: {val_loss.item():.5f} Test Loss: {test_loss.item():.5f}")# VF Loss: {vf_loss.item():.5f}
            
            ep_num[k], loss_hist[k], val_loss_hist[k], test_loss_hist[k] = i, train_loss.item(), val_loss.item(), test_loss.item()
            # k += 1
            current_relative_error = calculate_relative_error(model, dyn_sys_info[0], device)
            # Check if current model has the lowest relative error so far
            if current_relative_error < min_relative_error:
                min_relative_error = current_relative_error
                # Save the model
                torch.save(model.state_dict(), f"{args.train_dir}/best_model.pth")
                logger.info(f"Epoch {i}: New minimal relative error: {min_relative_error:.2f}%, model saved.")
            if loss_type == "Jacobian":
                test_model_J = compute_batch_jac(0, X_test)
                test_jac_norm_diff = criterion(Test_J, test_model_J)
                jac_diff_train[k], jac_diff_test[k] = jac_norm_diff, test_jac_norm_diff
                JAC_plot_path = f'{args.train_dir}JAC_'+str(i)+'.jpg'
                # JAC_plot_path = f'{args.train_dir}train_{model_type}_{dyn_sys_type}/JAC_'+str(i)+'.jpg'
                # plot_vector_field(model, path=JAC_plot_path, idx=1, t=0., N=100, device='cuda')
            else:
                MSE_plot_path = f'{args.train_dir}MSE_'+str(i)+'.jpg'
                # plot_vector_field(model, y_pred_train,path=MSE_plot_path, idx=1, t=0., N=100, device='cuda')

            k = k + 1

    # if loss_type == "Jacobian":
    #     for i in [0, 1, -2, -1]:
    #         print("Point:", X[i].detach().cpu().numpy(), "\n", "True:", "\n", True_J[i].detach().cpu().numpy(), "\n", "JAC:", "\n", cur_model_J[i].detach().cpu().numpy())
    # else:
    #     MSE_plot_path = f'{args.train_dir}MSE_'+str(i)+'.jpg'
    #     # plot_vector_field(model, y_pred_train, path=MSE_plot_path, idx=1, t=0., N=100, device='cuda')
    #     jac_diff_train, jac_diff_test = None, None

    return ep_num, loss_hist, test_loss_hist, jac_diff_train, jac_diff_test, Y_test



##############
#### Plot ####
##############

def plot_loss(epochs, train, test, path):
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

def plot_attractor(model, dyn_info, time, path):
    # generate true orbit and learned orbit
    dyn, dim, time_step = dyn_info
    tran_orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    true_o = torchdiffeq.odeint(dyn, tran_orbit[-1], torch.arange(0, time, time_step), method='rk4', rtol=1e-8)
    learned_o = torchdiffeq.odeint(model.eval(), tran_orbit[-1], torch.arange(0, time, time_step), method="rk4", rtol=1e-8).detach().cpu().numpy()

    # create plot
    fig, axs = subplots(2, 3, figsize=(24,12))
    cmap = cm.plasma
    num_row, num_col = axs.shape

    for x in range(num_row):
        for y in range(num_col):
            orbit = true_o if x == 0 else learned_o
            if y == 0:
                axs[x,y].plot(orbit[0, 0], orbit[0, 1], '+', markersize=35, color=cmap.colors[0])
                axs[x,y].scatter(orbit[:, 0], orbit[:, 1], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.2)
                axs[x,y].set_xlabel("X")
                axs[x,y].set_ylabel("Y")
            elif y == 1:
                axs[x,y].plot(orbit[0, 0], orbit[0, 2], '+', markersize=35, color=cmap.colors[0])
                axs[x,y].scatter(orbit[:, 0], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.2)
                axs[x,y].set_xlabel("X")
                axs[x,y].set_ylabel("Z")
            else:
                axs[x,y].plot(orbit[0, 1], orbit[0, 2], '+', markersize=35, color=cmap.colors[0])
                axs[x,y].scatter(orbit[:, 1], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.2)
                axs[x,y].set_xlabel("Y")
                axs[x,y].set_ylabel("Z")
        
            axs[x,y].tick_params(labelsize=42)
            axs[x,y].xaxis.label.set_size(42)
            axs[x,y].yaxis.label.set_size(42)
    tight_layout()
    fig.savefig(path, format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    return

def plot_vf_err_st(model, dyn_info, model_type, loss_type):
    dyn, dim, time_step = dyn_info
    
    dyn_sys_type = "lorenz" if dyn == lorenz else "rossler"

    orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    orbit = torchdiffeq.odeint(dyn, orbit[-1], torch.arange(0, 10, time_step), method='rk4', rtol=1e-8)
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

    # # For debugging purpose, will remove it later
    # print("vf_nn", vf_nn.shape)
    # print("vf", vf.shape)
    # print("vf_nn-vf", vf_nn - vf)
    # print("err", err, err.shape)
    # print("mag", mag, mag.shape)
    # print(percentage_err)
    
    # ax.plot(t, percentage_err, "o", label=r"$\frac{\|\hat x - x\|_2}{\|x\|_2}$", ms=3.0)
    np.savetxt(f'{args.train_dir}{args.loss_type}error_st.txt', np.column_stack((t, err/mag*100)), fmt='%.6f')
    # ax.set_xlabel("time",fontsize=24)
    # ax.xaxis.set_tick_params(labelsize=24)
    # ax.yaxis.set_tick_params(labelsize=24)
    # ax.set_ylim(0, 50)
    # ax.legend(fontsize=24)
    # ax.grid(True)
    # tight_layout()
    # path = f"../plot/Relative_error/{args.model_type}_{args.loss_type}_{dyn_sys_type}.png"
    # savefig(path)
    return percentage_err


def plot_vf_err(model, y_pred_train, dyn_info, model_type, loss_type):
    dyn, dim, time_step = dyn_info
    dyn_sys_type = "lorenz" if dyn == lorenz else "rossler"
    # orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    # orbit = torchdiffeq.odeint(dyn, orbit[-1], torch.arange(0, 4, time_step), method='rk4', rtol=1e-8)
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
    # stop
    # mag = abs(vf[2])
    err = np.linalg.norm(vf_nn - vf, axis=0)
    # err = abs(vf_nn[2]-vf[2])
    t = time_step*np.arange(0, len_o)
    ax.plot(t, err/mag*100, "o", label=r"$\|Error\|_2$", ms=3.0)
    np.savetxt(f'{args.train_dir}{args.loss_type}error.txt', np.column_stack((t, err/mag*100)), fmt='%.6f')
    ax.set_xlabel("time",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.set_ylim(0, 2)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()

    path = f"{args.train_dir}MSE_error.png"
    savefig(path)


def plot_vector_field(model, y_pred_train,path, idx, t, N, device='cuda'):
    # Credit: https://torchdyn.readthedocs.io/en/latest/_modules/torchdyn/utils.html

    x = torch.linspace(-50, 50, N)
    y = torch.linspace(-50, 50, N)
    X, Y = torch.meshgrid(x,y)
    Z_random = torch.randn(1)*10
    U, V = np.zeros((N,N)), np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if idx == 1:
                phi = torch.stack([X[i,j], Y[i,j], torch.tensor(20.)]).to('cuda')
            else:
                phi = torch.stack([X[i,j].clone().detach(), torch.tensor(0), Y[i,j].clone().detach()]).to('cuda')
            O = model(0., phi).detach().cpu().numpy()
            if O.ndim == 1:
                U[i,j], V[i,j] = O[0], O[idx]
            else:
                U[i,j], V[i,j] = O[0, 0], O[0, idx]

    fig = figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    contourf = ax.contourf(X, Y, np.sqrt(U**2 + V**2), cmap='jet', vmin=vf_mag.min(), vmax=vf_mag.max())
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

def plot_vector_field(model, y_pred_train, path, idx, t, N, device='cuda'):
    # Adjust the range of x and y coordinates to focus on the area around y_pred_train
    x_range = torch.linspace(y_pred_train[:, 0].min().item() - 2, y_pred_train[:, 0].max().item() + 2, N)
    y_range = torch.linspace(y_pred_train[:, 1].min().item() - 2, y_pred_train[:, 1].max().item() + 2, N)
    X, Y = torch.meshgrid(x_range, y_range)
    
    # Compute the vector field components U and V
    U, V = np.zeros((N,N)), np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if idx == 1:
                phi = torch.stack([X[i,j], Y[i,j], torch.tensor(20.)])
            else:
                phi = torch.stack([X[i,j].clone().detach(), torch.tensor(0), Y[i,j].clone().detach()])
            O = model(0., phi).detach().cpu().numpy()
            if O.ndim == 1:
                U[i,j], V[i,j] = O[0], O[idx]
            else:
                U[i,j], V[i,j] = O[0, 0], O[0, idx]
    
    # Create the plot
    fig = figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    contourf = ax.contourf(X, Y, np.sqrt(U**2 + V**2), cmap='jet')
    fig.colorbar(contourf, ax=ax)  # Add a color bar to the plot

    ax.streamplot(X.T.numpy(), Y.T.numpy(), U.T, V.T, color='k')

    ax.set_xlim([x_range.min(), x_range.max()])
    ax.set_ylim([y_range.min(), y_range.max()])
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
    
def lyap_exps(dyn_sys_info, traj, iters):
    model, dim, time_step = dyn_sys_info
    LE = torch.zeros(dim)
    traj_gpu = traj
    f = lambda x: rk4(x, model, time_step)
    Jac = torch.vmap(torch.func.jacrev(f),randomness='same')(traj_gpu)
    Q = torch.rand(dim,dim)
    eye_cuda = torch.eye(dim)
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
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--num_epoch", type=int, default=10000)
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=5000)
    parser.add_argument("--num_trans", type=int, default=0)
    parser.add_argument("--num_val", type=int, default=3000)
    parser.add_argument("--loss_type", default="Jacobian", choices=["Jacobian", "MSE"])
    parser.add_argument("--dyn_sys", default="lorenz", choices=["lorenz", "rossler"])
    parser.add_argument("--model_type", default="MLP", choices=["MLP","MLP1", "NeuralODE", "CNN","CNN1", "HigherDimCNN", "GRU"])
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--reg_param", type=float, default=500)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--train_dir", default="../plot/Vector_field/train_vf_best_Jac_5k/") #"../plot/Vector_field/train_vf_best_Jac/"
        #####################3
    parser.add_argument("--mode", default="plot", choices=["train", "plot"])


    # Initialize Settings
    args = parser.parse_args()
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    
    dim = 3
    dyn_sys_func = lorenz if args.dyn_sys == "lorenz" else rossler
    dyn_sys_info = [dyn_sys_func, dim, args.time_step]
    criterion = torch.nn.MSELoss()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = f"{args.train_dir}{start_time}_{args.model_type}_{args.loss_type}_{args.dyn_sys}.txt"
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Create model
    if args.model_type == "MLP":
        m = ODE_MLP(y_dim=dim, n_hidden=args.n_hidden) #, n_layers=args.n_layers
    elif args.model_type == "MLP1":
        m = ODE_MLP1(y_dim=dim, n_hidden=args.n_hidden) 
    elif args.model_type == "NeuralODE":
        m = NeuralODE(y_dim=dim, n_hidden=args.n_hidden, depth=2) 
    elif args.model_type == "CNN":
        m = ODE_CNN(y_dim=dim, n_hidden=args.n_hidden)
    elif args.model_type == "CNN1":
        m = ODE_CNN1(y_dim=dim, n_hidden=args.n_hidden)
    elif args.model_type == "HigherDimCNN":
        m = ODE_HigherDim_CNN(y_dim=dim, n_hidden=args.n_hidden)


    if args.mode=="train":
            # Create Dataset
        dataset = create_data(dyn_sys_info, n_train=args.num_train, n_test=args.num_test, n_trans=args.num_trans, n_val=args.num_val)

        print("Training...") # Train the model, return node
        epochs, loss_hist, test_loss_hist, jac_train_hist, jac_test_hist,Y_test  = train(dyn_sys_info, m, device, dataset, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.reg_param, args.loss_type, args.model_type, args.batch_size)

        # RE_plot_path = f'{args.train_dir}minRE.jpg'
        # plot_vector_field(best_model, path=RE_plot_path, idx=1, t=0., N=100, device='cuda')


        # Plot Loss
        loss_path = f"{args.train_dir}{args.model_type}_{args.loss_type}_Total_{start_time}.png"
        jac_loss_path = f"{args.train_dir}{args.model_type}_{args.loss_type}_Jacobian_matching_{start_time}.png"
        mse_loss_path = f"{args.train_dir}{args.model_type}_{args.loss_type}_MSE_part_{start_time}.png"
        true_plot_path = f"{args.train_dir}True_{args.dyn_sys}.png"
        phase_path = f"{args.train_dir}{args.dyn_sys}_{args.model_type}_{args.loss_type}.png"

        plot_loss(epochs, loss_hist, test_loss_hist, loss_path) 
        if args.loss_type == "Jacobian":
            plot_loss(epochs, jac_train_hist, jac_test_hist, jac_loss_path) 
            plot_loss(epochs, abs(loss_hist - args.reg_param*jac_train_hist)*(args.time_step)**2, abs(test_loss_hist - args.reg_param*jac_test_hist)*(args.time_step)**2, mse_loss_path) 
    #  y_pred_train, 
        # Plot vector field & phase space
        plot_vf_err_st(m,dyn_sys_info, args.model_type, args.loss_type)
        plot_vf_err(m, Y_test, dyn_sys_info, args.model_type, args.loss_type)
        # plot_vector_field(dyn_sys_func, y_pred_train, path=true_plot_path, idx=1, t=0., N=100, device='cuda')
        plot_attractor(m, dyn_sys_info, 50, phase_path)

        # compute LE
        true_traj = torchdiffeq.odeint(dyn_sys_func, torch.randn(dim), torch.arange(0, 300, args.time_step), method='rk4', rtol=1e-8)
        print("Computing LEs of NN...")
        learned_LE = lyap_exps([m, dim, args.time_step], true_traj, 30000).detach().cpu().numpy()
        print("Computing true LEs...")
        True_LE = lyap_exps(dyn_sys_info, true_traj, 30000).detach().cpu().numpy()
        logger.info("%s: %s", "Learned LE", str(learned_LE))
        logger.info("%s: %s", "True LE", str(True_LE))
        print("Learned:", learned_LE, "\n", "True:", True_LE)

    else:

        print("Plotting...")
        import matplotlib.pyplot as plt
        best_model = m
        best_model.load_state_dict(torch.load(f"./best_model_JAC.pth", map_location=torch.device('cpu')))
        mse_model = ODE_MLPmse(y_dim=dim, n_hidden=512) #
        mse_model.load_state_dict(torch.load("./best_model_MSE.pth", map_location=torch.device('cpu')))

        dyn, dim, time_step = dyn_sys_info

        dim = 3
        device = 'cuda' 
        time_step = 0.01
        num_trajectories = 500
        long_time_steps = torch.arange(0, 1000, time_step)
        short_time_steps = torch.arange(0, 50, time_step)

        # Function to generate data
        def generate_data(model, initial_condition, shortinitial_condition, is_dynamical=True):
            # Long orbit generation
            long_orbit = torchdiffeq.odeint(model if is_dynamical else model.eval(), initial_condition, long_time_steps, method='rk4', rtol=1e-8).detach().cpu().numpy()
            
            short_orbits = []
            # if generate_short:
                # Generate multiple short orbits with different initial conditions
            for i in range(num_trajectories):
                # init_cond = torch.randn(dim)
                short_orbit = torchdiffeq.odeint(model if is_dynamical else model.eval(), shortinitial_condition[i,:], short_time_steps, method='rk4', rtol=1e-8).detach().cpu().numpy()
                short_orbits.append(short_orbit)
            
            return long_orbit, np.array(short_orbits)

        # Generate true dynamics long orbit and its short orbits
        true_initial_condition = torch.randn(dim)  # Initial condition for the true model
        trueshort_initial_condition = torch.randn(num_trajectories,dim) 
        true_long, true_short= generate_data(dyn, true_initial_condition, trueshort_initial_condition, is_dynamical=True)
    
        print("orbitshape", true_long.shape, "short", true_short.shape)
        learned_initial_condition = torch.tensor(true_long[-1], dtype=torch.float32)
        learnedshort_initial_condition = torch.tensor(true_short[:, -1, :], dtype=torch.float32)
        

        learned_long, learned_short = generate_data(best_model, learned_initial_condition, learnedshort_initial_condition, is_dynamical=False)      
        mse_long, mse_short = generate_data(mse_model, learned_initial_condition, learnedshort_initial_condition, is_dynamical=False)

        print("orbitshape", true_long.shape, learned_long.shape, mse_long.shape, "short", true_short.shape, learned_short.shape, mse_short.shape)

        import numpy as np

        # Function to plot histograms for three models in one subplot
        def plot_histograms(ax, data_true, data_learned, data_mse, title):
            bins = np.linspace(min(np.min(data_true), np.min(data_learned), np.min(data_mse)), 
                            max(np.max(data_true), np.max(data_learned), np.max(data_mse)), 30)

            ax.hist(data_true, bins=bins, alpha=0.7, label='True Model', color='royalblue')
            ax.hist(data_learned, bins=bins, alpha=0.7, label='Learned Model', color='darkorange')
            ax.hist(data_mse, bins=bins, alpha=0.7, label='MSE Model', color='limegreen')

            ax.set_title(title)
            ax.legend()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows (time, ensemble) x 3 columns (x, y, z)

        dimensions = ['X Dimension', 'Y Dimension', 'Z Dimension']
        for j in range(3): 
            plot_histograms(axes[0, j], true_long[:, j], learned_long[:, j], mse_long[:, j], f'Time Avg - {dimensions[j]}')
            plot_histograms(axes[1, j], learned_short[ :, j].flatten(), learned_short[:, j].flatten(), learned_short[:, j].flatten(), f'Ensemble Avg - {dimensions[j]}')

        plt.tight_layout()
        plt.show()
