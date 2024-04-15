import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np

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

class LatentODE(nn.Module):
    def __init__(self, data_size, hidden_size, latent_size):
        super(LatentODE, self).__init__()
        # Use an MLP instead of an RNN
        self.encoder= nn.Sequential(
            nn.Linear(data_size + 1, hidden_size),  # +1 if time is included as a feature
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
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
        print('ys_f', ys.size())
        ts_expanded = ts.unsqueeze(-1) #.expand(-1, -1, data_size)
        print('ts_expanded', ts_expanded.size())
        combined_input = torch.cat([ys[:,0,:], ts_expanded[:,0,:]], dim=-1)  # Concatenating time with data
        latent_params = self.encoder(combined_input) #encode
        context = self.hidden_to_latent(latent_params)
        # print('context',context.shape)
        mean, logstd = context[:,: self.latent_size], context[:,self.latent_size :]
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        latent = mean + eps * std

        y0 = self.latent_to_hidden(latent)
        sol = odeint(self.func, y0, ts[0,:])  
        sol=sol.transpose(0, 1)
        pred_ys = self.hidden_to_data(sol)
        
        return pred_ys, mean, std, self.func

    def sample(self, ts):
        latent = torch.randn((self.latent_size,)) #, device=device
        latent_dynamics = self.latent_to_hidden(latent)
        sol = odeint(self.func, latent_dynamics, ts)
        return self.hidden_to_data(sol)




def train(model, dataloader, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        print('start training')
        for batch in dataloader:
            ts, ys = batch 
            # for ts, ys in zip(*batch):
                # print('startbatch')
                # ts = ts.unsqueeze(0)  # Add batch dimension
                # ys = ys.unsqueeze(0)  # Add batch dimension
                # print('ys_loader', ys.size())
                # print('ts_loader', ts.size())
            optimizer.zero_grad()
            pred_ys, mean, std, _ = model(ts, ys)
            logstd = torch.log(std)
            # print('ys', ys.size())
            # print('pred_ys', pred_ys.size())
            reconstruction_loss = torch.mean((ys - pred_ys) ** 2)
            kl_loss = torch.mean(mean ** 2 + std ** 2 - 2 * logstd - 1)
            loss = reconstruction_loss + kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Average Loss {total_loss / len(dataloader.dataset)}")
        # print(f"Epoch {epoch}: Average Loss {total_loss / len(dataloader)}")

        save_every = 20
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                # For plotting, use a single batch or a representative sample
                average_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch}: Average Loss {average_loss}")

                with torch.no_grad():
                    # For plotting, use a single batch or a representative sample
                    sample_ts = torch.linspace(0, 12, 30)
                    batch_ts, batch_ys = next(iter(dataloader))  # Use the first batch for plotting
                    # print('shapes',sample_ts.shape, batch_ys[:1].squeeze(0).shape)
                    sample_ys = model.sample(sample_ts)
                    # sample_ys, _, _, _ = model.sample(sample_ts, batch_ys[:1].squeeze(0))

                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    sample_y_np = sample_ys.detach().cpu().numpy()
                    # print('shapesampley',sample_y_np.shape)
                    sample_t_np = sample_ts.numpy()
                    # print('shapesamplet',sample_t_np.shape)

                    ax.plot(sample_t_np, sample_y_np[:, 0])
                    ax.plot(sample_t_np, sample_y_np[:, 1])
                    # ax.plot(sample_t_np, sample_y_np[:, 2])
                    ax.set_xlabel("t")
                    plt.savefig(f"./latentode/latent2d_ode_{epoch}.png")
                    plt.close(fig)

# # Example usage
# data_size = 1
# hidden_size = 20
# latent_size = 10
# model = LatentODE(data_size, hidden_size, latent_size)
# optimizer = optim.Adam(model.parameters(), lr=0.01)

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
def get_data(dataset_size, total_time=2, time_steps=30):
    data_path = './latentode/ode2d.pth'

    # Check if data already exists
    if os.path.exists(data_path):
        # Load the data from the file
        data = torch.load(data_path)
        ts, ys = data['ts'], data['ys']
        print("Loaded data from saved file.")
    else:
        # Initial conditions
        y0 = torch.randn(dataset_size, 2)

        # Uniformly discretized time points
        ts = torch.linspace(0, total_time, steps=time_steps).repeat(dataset_size, 1)

        # Define the ODE system
        def func(t, y):
            A = torch.tensor([[-0.1, 1.3], [-1, -0.1]])
            return A @ y

        # Solve the ODE system
        ys = torch.empty(dataset_size, time_steps, 2)
        for i in range(dataset_size):
            ys[i] = odeint(func, y0[i], ts[i])

        torch.save({'ts': ts, 'ys': ys}, data_path)
        print("Saved data to file.")

    return ts, ys


def create_dataloader(ts, ys, batch_size):
    dataset = TensorDataset(ts, ys)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def transformed_vector_field(model, x):
    # Transform x from data to hidden space, apply the vector field, and transform back to data space.
    # print('xshape',x.shape) #3
    h = model.encoder(x) 
    # print('hshape',h.shape) #=128
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

def odefunc(t, y):
        A = torch.tensor([[-0.1, 1.3], [-1, -0.1]])
        return A @ y



def lyap_exps(dyn_sys_info, traj, iters):
    model, dim, time_step = dyn_sys_info
    LE = torch.zeros(dim) #.to(device)
    traj_gpu = traj #.to(device)

    if model == odefunc:
        f = lambda x: rk4(x, model, time_step)
    else:
        f = lambda x: rk4_nn(x, lambda x: transformed_vector_field(model, x), time_step)
    Jac = torch.vmap(torch.func.jacrev(f))(traj_gpu)
    Q = torch.rand(dim,dim) #.to(device)
    eye_cuda = torch.eye(dim) #.to(device)
    for i in range(iters):
        if i > 0 and i % 1000 == 0:
            print("Iteration: ", i, ", LE[0]: ", LE[0].detach().cpu().numpy()/i/time_step)
        Q = torch.matmul(Jac[i], Q)
        Q, R = torch.linalg.qr(Q)
        LE += torch.log(abs(torch.diag(R)))
    return LE/iters/time_step

time_step = 0.01
dim=2
data_size = 2  # Assuming 2-dimensional data
hidden_size = 20
latent_size = 10
model = LatentODE(data_size, hidden_size, latent_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Prepare the data
dataset_size = 1000
batch_size = 1000
ts, ys = get_data(dataset_size)

dataloader = create_dataloader(ts, ys, batch_size)

# Train the model

train(model, dataloader, optimizer, num_epochs=1000)

nn_sys_func = model
nn_sys_info = [nn_sys_func, data_size, time_step]
dyn_sys_func = odefunc
dyn_sys_info = [dyn_sys_func, data_size, time_step]
true_traj = odeint(dyn_sys_func, torch.randn(dim), torch.arange(0, 300, time_step), method='rk4', rtol=1e-8)

print("Computing true LEs...")
True_LE = lyap_exps(dyn_sys_info, true_traj, 3000) #.detach().cpu().numpy()
print("True LEs: ", True_LE)

# print("Computing LEs of NN...")
# learned_LE = lyap_exps(nn_sys_info , true_traj, 3000) #.detach().cpu().numpy()
# print("Learned LEs: ", learned_LE)
