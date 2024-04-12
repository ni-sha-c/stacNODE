import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.autograd.functional as F
import datetime
import argparse
import json
import logging
import os
import math
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Func(nn.Module):
    def __init__(self, hidden_size):
        super(Func, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
    
    def forward(self, t, y):
        return self.mlp(y)

# class Func(nn.Module):
#     def __init__(self, hidden_size, width_size, depth, scale_factor=1.0):
#         super(Func, self).__init__()
#         layers = [nn.Linear(hidden_size, width_size), nn.Softplus()]
#         for _ in range(depth - 1):
#             layers.extend([nn.Linear(width_size, width_size), nn.Softplus()])
#         layers.append(nn.Linear(width_size, hidden_size))
#         layers.append(nn.Tanh())  
#         self.mlp = nn.Sequential(*layers)
#         self.scale = nn.Parameter(torch.tensor([scale_factor], dtype=torch.float32))

#     def forward(self, t, y):
#       
#         return self.scale * self.mlp(y)



class LatentODE(nn.Module):
    def __init__(self, data_size, hidden_size, latent_size):
        super(LatentODE, self).__init__()
        # Use an MLP instead of an RNN
        self.encoder = nn.Sequential(
            nn.Linear(data_size, hidden_size),  # +1 if time is included as a feature
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * latent_size)
        )
        self.hidden_to_latent = nn.Linear(hidden_size, 2 * latent_size)
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
        )
        self.hidden_to_data = nn.Linear(hidden_size, data_size)
        self.func = Func(hidden_size)
        self.latent_size = latent_size
        self.hidden_size = hidden_size
    
    def forward(self, ts, ys):
        
        ts_expanded = ts.unsqueeze(-1) #.expand(-1, -1, data_size)
        # print('ysshape', ys.size())
        latent_params = self.encoder(ys[:,0,:]) #encode
        mean, logstd = torch.chunk(latent_params, 2, dim=-1)

        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        latent = mean + eps * std

        y0 = self.latent_to_hidden(latent)
        sol = odeint(self.func, y0, ts[0,:])  
        # print('shapesol', sol.shape)
        sol=sol.transpose(0, 1)
        pred_ys = self.hidden_to_data(sol)

        
        return pred_ys, mean, std, self.func

    def sample(self, ts):
        latent = torch.randn((self.latent_size,), device=device) #
        latent_dynamics = self.latent_to_hidden(latent)
        sol = odeint(self.func, latent_dynamics, ts)
        return self.hidden_to_data(sol)


def process_batch(ts, ys):
    ts = ts.to(device)  
    ys = ys.to(device)  
    optimizer.zero_grad()

    pred_ys, mean, std, _ = model(ts, ys)
    logstd = torch.log(std)

    pred_ys = pred_ys.to(device)
    reconstruction_loss = torch.mean((ys - pred_ys) ** 2)
    kl_loss = torch.mean(mean ** 2 + std ** 2 - 2 * logstd - 1)
    loss = reconstruction_loss + kl_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def train(model, dataloader, optimizer, num_epochs=3):
    model.train()
    model.to(device)
    # dataset = create_data(dyn_sys_info, n_train=6000, n_test=2000, n_trans=0, n_val=500)
    # ys, Y_train, X_val, Y_val, X_test, Y_test = dataset
    # # ys, Y_train, X_val, Y_val, X_test, Y_test = X_train.to(device), Y_train.to(device), X_val.to(device), Y_val.to(device), X_test.to(device), Y_test.to(device)
    # ts = torch.linspace(0, time_step, 6000) #.to(device)
    # train_data = TensorDataset(ts, ys)
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)

    # ts, ys = get_data(dataset_size)
    
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = datetime.datetime.now() 
        for batch in dataloader:
            ts, ys = batch  
            ts, ys = ts.to(device), ys.to(device)  


            optimizer.zero_grad()
            pred_ys, mean, std, _ = model(ts, ys)
            # print('model done')
            logstd = torch.log(std)
            pred_ys=pred_ys.to(device)
            reconstruction_loss = torch.mean((ys - pred_ys) ** 2)
            kl_loss = torch.mean(mean ** 2 + std ** 2 - 2 * logstd - 1)
            loss = reconstruction_loss + kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch}: Time used {datetime.datetime.now() - start_time}")
        save_every = 200
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                
                average_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch}: Average Loss {average_loss}")
                
                sample_ts = torch.linspace(0, 12, 30)
                # batch_ts, batch_ys = next(iter(dataloader))  # Use the first batch for plotting
                # print('shapes',sample_ts.shape, batch_ys[:1].squeeze(0).shape)

                sample_ts = sample_ts.to(device)
                sample_ys = model.sample(sample_ts)

                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                sample_y_np = sample_ys.detach().cpu().numpy()
                # print('shapesampley',sample_y_np.shape)
                sample_t_np = sample_ts.detach().cpu().numpy()
                # print('shapesamplet',sample_t_np.shape)

                ax.plot(sample_t_np, sample_y_np[:, 0])
                ax.plot(sample_t_np, sample_y_np[:, 1])
                ax.plot(sample_t_np, sample_y_np[:, 2])
                ax.set_xlabel("t")
                plt.savefig(f"./latent3d_ode_{epoch}.png")
                plt.close(fig)



# # Lorenz system parameters
# sigma = 10.0
# rho = 28.0
# beta = 8.0 / 3.0

def get_data(dataset_size, total_time=10, time_steps=1000):
    # Path to save or load the data
    data_path = './ode_datasetnew.pth'

    # Check if data already exists
    if os.path.exists(data_path):
        # Load the data from the file
        data = torch.load(data_path)
        ts, ys = data['ts'], data['ys']
        print("Loaded data from saved file.")
    else:
        y0 = torch.randn(dataset_size, 3, device=device)  # Move the tensor to GPU
        ts = torch.linspace(0, total_time, steps=time_steps, device=device).repeat(dataset_size, 1)

        # Lorenz system ODEs
        def func(t, y):
            sigma = 10.0  
            rho = 28.0
            beta = 8.0 / 3.0
            x, y, z = y[..., 0], y[..., 1], y[..., 2]
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return torch.stack([dx, dy, dz], axis=-1)

        # Solve the ODE system
        ys = torch.empty(dataset_size, time_steps, 3, device=device)  # Move the tensor to GPU
        for i in range(dataset_size):
            ys[i] = odeint(func, y0[i], ts[i])

        torch.save({'ts': ts, 'ys': ys}, data_path)
        print("Saved data to file.")

    return ts, ys

def create_data(dyn_info, n_train, n_test, n_val, n_trans):
    dyn, dim, time_step = dyn_info
    # Adjust total time to account for the validation set
    tot_time = time_step * (n_train + n_test + n_val + n_trans + 1)
    t_eval_point = torch.arange(0, tot_time, time_step)

    # Generate trajectory using the dynamical system
    traj = odeint(dyn, torch.randn(dim), t_eval_point, method='rk4', rtol=1e-8)
    
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




def create_dataloader(ts, ys, batch_size):
    dataset = TensorDataset(ts, ys)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

def transformed_vector_field(model, x):
    # Transform x from data to hidden space, apply the vector field, and transform back to data space.
    h = model.encoder(x) 
    v_h = model.func(0, h)  
    J_g = model.hidden_to_data.weight  
    v_d = torch.matmul(v_h, J_g.T) + model.hidden_to_data.bias  
    return v_d

def rk4(x, f, dt):
    k1 = f(0, x)
    k2 = f(0, x + dt*k1/2)
    k3 = f(0, x + dt*k2/2)
    k4 = f(0, x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def rk4_nn(x, f, dt):
    k1 = f(x)
    k2 = f(x + dt*k1/2)
    k3 = f(x + dt*k2/2)
    k4 = f(x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
def lyap_exps(dyn_sys_info, traj, iters):
    model, dim, time_step = dyn_sys_info
    LE = torch.zeros(dim).to(device)
    traj_gpu = traj.to(device)

    if model == lorenz:
        f = lambda x: rk4(x, model, time_step)
    else:
        f = lambda x: rk4_nn(x, lambda x: transformed_vector_field(model, x), time_step)
    Jac = torch.vmap(torch.func.jacrev(f))(traj_gpu)
    Q = torch.rand(dim,dim).to(device)
    eye_cuda = torch.eye(dim).to(device)
    for i in range(iters):
        if i > 0 and i % 1000 == 0:
            print("Iteration: ", i, ", LE[0]: ", LE[0].detach().cpu().numpy()/i/time_step)
        Q = torch.matmul(Jac[i], Q)
        Q, R = torch.linalg.qr(Q)
        LE += torch.log(abs(torch.diag(R)))
    return LE/iters/time_step

def plot_attractor(model, dyn_info, time, path):
    # generate true orbit and learned orbit
    dyn, dim, time_step = dyn_info
    tran_orbit = odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    true_o = odeint(dyn, tran_orbit[-1], torch.arange(0, time, time_step), method='rk4', rtol=1e-8)
    #learned_o = odeint(model.func.eval().to(device), tran_orbit[-1].to(device), torch.arange(0, time, time_step), method="rk4", rtol=1e-8).detach().cpu().numpy()

    sample_ts = torch.arange(0, 50, time_step)
    sample_ts = sample_ts.to(device)
    learned_o = model.sample(sample_ts).detach().cpu().numpy()

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

time_step = 0.01
data_size = 3  
hidden_size = 20
latent_size = 10
model = LatentODE(data_size, hidden_size, latent_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
dim = data_size
print('preparing data..')
dataset_size = 100

batch_size = 32
nn_sys_func = model
nn_sys_info = [nn_sys_func, data_size, time_step]
dyn_sys_func = lorenz 
dyn_sys_info = [dyn_sys_func, data_size, time_step]
ts, ys = get_data(dataset_size)

dataloader = create_dataloader(ts, ys, batch_size)
train(model, dataloader, optimizer, num_epochs=1000)

true_traj = odeint(dyn_sys_func, torch.randn(dim), torch.arange(0, 300, time_step), method='rk4', rtol=1e-8)
print("Computing true LEs...")
True_LE = lyap_exps(dyn_sys_info, true_traj, 3000).detach().cpu().numpy()
print("True LEs: ", True_LE)

print("Computing LEs of NN...")
learned_LE = lyap_exps(nn_sys_info , true_traj, 3000).detach().cpu().numpy()
print("Learned LEs: ", learned_LE)

phase_path = "attractor.png"
plot_attractor(nn_sys_func, dyn_sys_info, 50, phase_path)