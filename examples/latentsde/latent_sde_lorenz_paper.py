# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a latent SDE on data from a stochastic Lorenz attractor.

Reproduce the toy example in Section 7.2 of https://arxiv.org/pdf/2001.01328.pdf

To run this file, first run the following to install extra requirements:
pip install fire

To run, execute:
python -m examples.latent_sde_lorenz
"""

import EntropyHub as eh
import logging
import os
from typing import Sequence
from torch.func import jacrev
import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
from torch.autograd import grad
import torchsde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class StochasticLorenz(object):
    """Stochastic Lorenz attractor.

    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (0, 0, 0)):
        super(StochasticLorenz, self).__init__()
        self.a = a
        self.b = b

    def f(self, t, y):
        # print('y:', y.shape)
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        # print('x1:', x1.shape, 'x2:', x2.shape, 'x3:', x3.shape)
        a1, a2, a3 = self.a

        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)

    def g(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        b1, b2, b3 = self.b

        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3
        return torch.cat([g1, g2, g3], dim=1)

    def sample(self, x0, ts, noise_std, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        # xs = torchsde.sdeint(self, x0, ts)
        xs = rk4_solver(self, x0, ts, dt=1e-2)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
            return xs, (mean, std)
        return xs



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        # print('yshape', y.shape)
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=1e-2, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method)

        _xs = self.projector(zs)
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        # print('xs:', xs.shape, '_xs:', _xs.shape)
        mse = torch.mean((_xs - xs) ** 2) #.sum(dim=(0, 2)).mean(dim=0)
        return log_pxs, logqp0 + logqp_path, mse

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        print('z0:', z0.shape)
        # zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=None) #
        zs = euler_maruyama(self, z0, ts, dt=1e-2)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs

def euler_maruyama_h(sde, y0, ts, dt):
    """Simple Euler-Maruyama solver for SDEs.

    Args:
        sde: An SDE object with methods `f` and `g` for drift and diffusion.
        y0 (torch.Tensor): Initial state of the system.
        ts (torch.Tensor): Times at which to simulate, assumed to be evenly spaced.
        dt (float): Time step size.
    
    Returns:
        torch.Tensor: Simulated trajectory.
    """
    num_steps = len(ts)
    y = torch.zeros(num_steps, *y0.shape, device=y0.device, dtype=y0.dtype)
    y[0] = y0

    for i in range(1, num_steps):
        t = ts[i-1]
        dw = torch.randn_like(y0) * torch.sqrt(torch.tensor(dt, device=y0.device))
        y[i] = y[i-1] + sde.h(t, y[i-1]) * dt + sde.g(t, y[i-1]) * dw

    return y

class trueLorenz:
    def __init__(self, sigma=10.0, rho=28.0, beta=8/3, noise_scale=0.1):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.noise_scale = noise_scale  # Scaling factor for noise term

    def f(self, t, y):
        """Drift component of the Lorenz system."""
        # print('y:', y)
        x, y, z = y[..., 0], y[..., 1], y[..., 2]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.stack([dx, dy, dz], dim=-1)



def euler_maruyama_d(sde, y0, ts, dt):
    """Simple Euler-Maruyama solver for SDEs.

    Args:
        sde: An SDE object with methods `f` and `g` for drift and diffusion.
        y0 (torch.Tensor): Initial state of the system.
        ts (torch.Tensor): Times at which to simulate, assumed to be evenly spaced.
        dt (float): Time step size.
    
    Returns:
        torch.Tensor: Simulated trajectory.
    """
    num_steps = len(ts)
    y = torch.zeros(num_steps, *y0.shape, device=y0.device, dtype=y0.dtype)
    y[0] = y0

    for i in range(1, num_steps):
        t = ts[i-1]
        dw = torch.randn_like(y0) * torch.sqrt(torch.tensor(dt, device=y0.device))
        y_prev = y[i-1].clone()  # Clone to avoid in-place operations
        dy = sde.f(t, y_prev) * dt 
        y[i] = y_prev + dy

    return y

def euler_maruyama(sde, y0, ts, dt):
    """Simple Euler-Maruyama solver for SDEs.

    Args:
        sde: An SDE object with methods `f` and `g` for drift and diffusion.
        y0 (torch.Tensor): Initial state of the system.
        ts (torch.Tensor): Times at which to simulate, assumed to be evenly spaced.
        dt (float): Time step size.
    
    Returns:
        torch.Tensor: Simulated trajectory.
    """
    num_steps = len(ts)
    y = torch.zeros(num_steps, *y0.shape, device=y0.device, dtype=y0.dtype)
    y[0] = y0

    for i in range(1, num_steps):
        t = ts[i-1]
        dw = torch.randn_like(y0) * torch.sqrt(torch.tensor(dt, device=y0.device))
        y_prev = y[i-1].clone()  # Clone to avoid in-place operations
        dy = sde.h(t, y_prev) * dt + sde.g(t, y_prev) * dw
        y[i] = y_prev + dy

    return y



def rk4_step(sde, y, t, dt):
    """Perform a single RK4 step."""
    f = sde.f
    k1 = f(t, y) * dt
    k2 = f(t + dt / 2, y + k1 / 2) * dt
    k3 = f(t + dt / 2, y + k2 / 2) * dt
    k4 = f(t + dt, y + k3) * dt
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def rk4_step_h(sde, y, t, dt):
    """Perform a single RK4 step."""
    f = sde.h
    k1 = f(t, y) * dt
    k2 = f(t + dt / 2, y + k1 / 2) * dt
    k3 = f(t + dt / 2, y + k2 / 2) * dt
    k4 = f(t + dt, y + k3) * dt
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk4_solver(sde, y0, ts, dt):
    num_steps = len(ts)
    y = torch.zeros(num_steps, *y0.shape, device=y0.device, dtype=y0.dtype)
    y[0] = y0.clone()

    for i in range(1, num_steps):
        t = ts[i-1]
        y[i] = rk4_step(sde, y[i-1].clone(), t, dt)  # Ensure no in-place modification by cloning

    return y

def rk4_solver_h(sde, y0, ts, dt):
    num_steps = len(ts)
    y = torch.zeros(num_steps, *y0.shape, device=y0.device, dtype=y0.dtype)
    y[0] = y0.clone()

    for i in range(1, num_steps):
        t = ts[i-1]
        y[i] = rk4_step_h(sde, y[i-1].clone(), t, dt)  # Ensure no in-place modification by cloning

    return y

def make_dataset(t0, t1, batch_size, noise_std, train_dir, device):
    data_path = os.path.join(train_dir, 'lorenz_data_ode.pth')
    if os.path.exists(data_path):
        data_dict = torch.load(data_path)
        xs, ts, norms = data_dict['xs'], data_dict['ts'], data_dict['norms']
        logging.warning(f'Loaded toy data at: {data_path}')
        if xs.shape[1] != batch_size:
            raise ValueError("Batch size has changed; please delete and regenerate the data.")
        if ts[0] != t0 or ts[-1] != t1:
            raise ValueError("Times interval [t0, t1] has changed; please delete and regenerate the data.")
    else:
        _y0 = torch.randn(batch_size, 3, device=device)
        ts0 = torch.linspace(t0, 16, steps=8000, device=device)
        xs0 = StochasticLorenz().sample(_y0, ts0, noise_std, normalize=False)
        ts = torch.linspace(t0, t1, steps=1000, device=device)
        xs, norms = StochasticLorenz().sample(xs0[-1,:], ts, noise_std, normalize=True)

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save({'xs': xs, 'ts': ts, 'norms': norms}, data_path)
        logging.warning(f'Stored toy data at: {data_path}')
    return xs, ts, norms

def make_vis_dataset(t0, t1, batch_size, noise_std, train_dir, device):
    data_path = os.path.join(train_dir, 'lorenz_data_ode_vis.pth')
    if os.path.exists(data_path):
        data_dict = torch.load(data_path)
        xs, ts, norms = data_dict['xs'], data_dict['ts'], data_dict['norms']
        logging.warning(f'Loaded toy data at: {data_path}')
        if xs.shape[1] != batch_size:
            raise ValueError("Batch size has changed; please delete and regenerate the data.")
        if ts[0] != t0 or ts[-1] != t1:
            raise ValueError("Times interval [t0, t1] has changed; please delete and regenerate the data.")
    else:
        _y0 = torch.randn(batch_size, 3, device=device)
        ts0 = torch.linspace(t0, 16, steps=8000, device=device)
        xs0 = StochasticLorenz().sample(_y0, ts0, noise_std, normalize=False)
        ts = torch.linspace(t0, t1, steps=5000, device=device)
        xs, norms = StochasticLorenz().sample(xs0[-1,:], ts, noise_std, normalize=True)

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save({'xs': xs, 'ts': ts, 'norms': norms}, data_path)
        logging.warning(f'Stored toy data at: {data_path}')
    return xs, ts, norms

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

def compute_dimension_wise_wasserstein(z1, z2):
    """
    Compute pairwise Wasserstein distances separately for each dimension between two sets of time series data using PyTorch.

    Parameters:
    z1 (torch.Tensor): First set of sample trajectories with shape (time step, batch size, 3).
    z2 (torch.Tensor): Second set of sample trajectories with shape (time step, batch size, 3).

    Returns:
    dict: Dictionary containing matrices of Wasserstein distances for each dimension (x, y, z).
    dict: Dictionary containing means of the Wasserstein distances for each dimension.
    dict: Dictionary containing medians of the Wasserstein distances for each dimension.
    """
    # Initialize dictionaries to hold results
    distances = {'x': None, 'y': None, 'z': None}
    mean_distances = {}
    median_distances = {}

    # Loop over each dimension
    for dim, label in enumerate(['x', 'y', 'z']):
        # Extract the time series for the current dimension
        z1_dim = z1[:, :, dim]
        z2_dim = z2[:, :, dim]

        # Since it's a 2D slice, no need to transpose dimensions, just convert to numpy if using scipy.stats
        z1_flat = z1_dim#.cpu().numpy() 
        z2_flat = z2_dim#.cpu().numpy()
        

        # Create a matrix to store distances for the current dimension
        dim_distances = np.zeros((z1_flat.shape[0], z2_flat.shape[0]))

        # Compute pairwise Wasserstein distances
        for i in range(z1_flat.shape[0]):
            for j in range(z2_flat.shape[0]):
                # print('z1flat type', type(z1_flat[i]), z1_flat[i].shape)
                # print('z2flat type', type(z2_flat[j]), z2_flat[j].shape)
                u = z1_flat[i].cpu().numpy()
                v = z2_flat[j] #.cpu().numpy()
                dim_distances[i, j] = wasserstein_distance(u, v)
                # dim_distances[i, j] = wasserstein_distance(z1_flat[i], z2_flat[j])

        # Store distances and calculate summary statistics
        distances[label] = dim_distances
        mean_distances[label] = np.mean(dim_distances)
        median_distances[label] = np.median(dim_distances)

    return distances, mean_distances, median_distances

def plot_combined_empirical_measures(x_vals1, y_vals1, z_vals1, x_vals2, y_vals2, z_vals2, norms1, norms2, axes, bins=50):
    """
    Plot the empirical measure for x, y, and z from two datasets on given axes, with specific legend labels.
    Unnormalize the data using provided normalization constants.

    norms1 and norms2 are tuples where:
    - norms[0] contains the means for x, y, z
    - norms[1] contains the standard deviations for x, y, z
    """
    labels = ['x', 'y', 'z']
    datasets1 = [x_vals1, y_vals1, z_vals1]
    datasets2 = [x_vals2, y_vals2, z_vals2]
    colors = [['blue', 'orange'], ['blue', 'orange'], ['blue', 'orange']]  # Use nested lists to ensure proper cycling
    legend_labels = [['Density for true x', 'Density for learned x'],
                     ['Density for true y', 'Density for learned y'],
                     ['Density for true z', 'Density for learned z']]

    for i, ((dataset1, dataset2), ax, label, color_pair, legend_pair) in enumerate(zip(zip(datasets1, datasets2), axes, labels, colors, legend_labels)):
        
        # Plot histograms
        for data, col, leg_label in zip([dataset1, dataset2], color_pair, legend_pair):
            vals_flat = data.flatten()
            hist, edges = np.histogram(vals_flat, bins=bins, density=True)
            widths = np.diff(edges)
            ax.bar(edges[:-1], hist, width=widths, align='edge', alpha=0.7, label=leg_label, color=col)

        ax.set_xlabel(f'${label}$', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title(f'Empirical Measure of ${label}$', fontsize=20)
        ax.grid(True)
        ax.legend()


def plot_empirical_measure(x_vals, z_vals, ax, bins=50):
    """
    Plot the empirical measure as a 2D histogram on the given axes.

    Parameters:
    x_vals (np.ndarray): X-values of the data points.
    z_vals (np.ndarray): Z-values of the data points.
    ax (matplotlib.axes.Axes): The axis to plot the empirical measure on.
    bins (int): The number of bins for the 2D histogram.
    """
    # Flatten the arrays to consider all batches and time steps
    x_vals_flat = x_vals.flatten()
    z_vals_flat = z_vals.flatten()

    # Create a 2D histogram as the empirical measure (projection on x-z plane)
    hist, x_edges, z_edges = np.histogram2d(x_vals_flat, z_vals_flat, bins=bins, density=True)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    x_grid, z_grid = np.meshgrid(x_centers, z_centers)
    pcm = ax.pcolormesh(x_grid, z_grid, hist.T, shading='auto', cmap='plasma')
    plt.colorbar(pcm, ax=ax, label='Density')

    ax.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax.set_ylabel('$z_3$', labelpad=.5, fontsize=16)
    ax.set_title('Empirical Measure', fontsize=20)




def vis1(xs, ts, lorenz, latent_sde, bm_vis, norms_vis, norms_data, train_dir, num_samples=30):
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 2)
    ax00 = fig.add_subplot(gs[0, 0], projection='3d')
    ax01 = fig.add_subplot(gs[0, 1], projection='3d')
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])



    # left plot: samples from learned model.
    xs_nn = latent_sde.sample(batch_size=xs.size(1), ts=ts).cpu().numpy() #, bm=bm_vis


    if isinstance(norms_vis[0], torch.Tensor):
        mean_vis = norms_vis[0].cpu().numpy()
        std_vis = norms_vis[1].cpu().numpy()
        # mean_data = norms_data[0].cpu().numpy()
        # std_data = norms_data[1].cpu().numpy()
    else:
        mean_vis, std_vis = norms_vis
    mean_data, std_data = norms_data
    xs_nn = xs_nn * std_vis + mean_vis  # Unnormalize the samples
    xs = xs * std_data + mean_data  # Unnormalize the data

    z1_sample, z2_sample, z3_sample = np.split(xs_nn, indices_or_sections=3, axis=-1)

    [ax01.plot(z1_sample[:, i, 0], z2_sample[:, i, 0], z3_sample[:, i, 0]) for i in range(num_samples)]
    ax01.scatter(z1_sample[0, :num_samples, 0], z2_sample[0, :num_samples, 0], z3_sample[0, :num_samples, 0], marker='x')
    ax01.set_yticklabels([])
    ax01.set_xticklabels([])
    ax01.set_zticklabels([])
    ax01.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax01.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax01.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    ax01.set_title('Samples', fontsize=20)

    xlim = ax01.get_xlim()
    ylim = ax01.get_ylim()
    zlim = ax01.get_zlim()

    _y0 = torch.tensor(np.stack([z1_sample[0, :num_samples, 0], z2_sample[0, :num_samples, 0], z3_sample[0, :num_samples, 0]], axis=-1), dtype=torch.float32, device=ts.device)
    noise_std=0.01
    # xs = StochasticLorenz().sample(_y0, ts, noise_std, normalize=False)
    # right plot: data.

    xs = rk4_solver(lorenz, _y0, ts, dt=1e-2)
    z1, z2, z3 = np.split(xs.cpu().numpy(), indices_or_sections=3, axis=-1)
    [ax00.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
    ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], z3[0, :num_samples, 0], marker='x')
    ax00.set_yticklabels([])
    ax00.set_xticklabels([])
    ax00.set_zticklabels([])
    ax00.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax00.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax00.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    ax00.set_title('Data', fontsize=20)
    ax00.set_xlim(xlim)
    ax00.set_ylim(ylim)
    ax00.set_zlim(zlim)

    # Plot the empirical measure (data)
    plot_empirical_measure(z1[:, :, 0], z3[:, :, 0], ax10)
    ax10.set_xlim(xlim)
    ax10.set_ylim(zlim)

    # Plot the empirical measure (samples)
    plot_empirical_measure(z1_sample[:, :, 0], z3_sample[:, :, 0], ax11)
    ax11.set_xlim(xlim)
    ax11.set_ylim(zlim)
    

    plt.savefig(train_dir+'attractor.pdf') #img_path
    plt.close()

    # fig_3d, axs_3d = plt.subplots(2, 3, figsize=(28, 12))
    # plot_individual_empirical_measures(z1[:, :, 0], z2[:, :, 0], z3[:, :, 0], axs_3d[0,:])
    # plot_individual_empirical_measures(z1_sample[:, :, 0], z2_sample[:, :, 0], z3_sample[:, :, 0], axs_3d[1,:])
    # plt.savefig(img_path) #'./dump/lorenz_ode_may8/measure.pdf'
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    plot_combined_empirical_measures(z1[:, :, 0], z2[:, :, 0], z3[:, :, 0], 
                                  z1_sample[:, :, 0], z2_sample[:, :, 0], z3_sample[:, :, 0], norms_vis, norms_data, axs)

    plt.savefig(train_dir+'e_measure.pdf') #
    plt.close()

    return xs, xs_nn


def vis(xs, ts, latent_sde, bm_vis, norms_vis, norms_data, train_dir, num_samples=30):
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 2)
    ax00 = fig.add_subplot(gs[0, 0], projection='3d')
    ax01 = fig.add_subplot(gs[0, 1], projection='3d')
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])



    # left plot: samples from learned model.
    xs_nn = latent_sde.sample(batch_size=xs.size(1), ts=ts).cpu().numpy() #, bm=bm_vis


    if isinstance(norms_vis[0], torch.Tensor):
        mean_vis = norms_vis[0].cpu().numpy()
        std_vis = norms_vis[1].cpu().numpy()
        # mean_data = norms_data[0].cpu().numpy()
        # std_data = norms_data[1].cpu().numpy()
    else:
        mean_vis, std_vis = norms_vis
    mean_data, std_data = norms_data
    xs_nn = xs_nn * std_vis + mean_vis  # Unnormalize the samples
    xs = xs * std_data + mean_data  # Unnormalize the data

    z1_sample, z2_sample, z3_sample = np.split(xs_nn, indices_or_sections=3, axis=-1)

    [ax01.plot(z1_sample[:, i, 0], z2_sample[:, i, 0], z3_sample[:, i, 0]) for i in range(num_samples)]
    ax01.scatter(z1_sample[0, :num_samples, 0], z2_sample[0, :num_samples, 0], z3_sample[0, :num_samples, 0], marker='x')
    ax01.set_yticklabels([])
    ax01.set_xticklabels([])
    ax01.set_zticklabels([])
    ax01.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax01.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax01.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    ax01.set_title('Samples', fontsize=20)

    xlim = ax01.get_xlim()
    ylim = ax01.get_ylim()
    zlim = ax01.get_zlim()

    _y0 = torch.tensor(np.stack([z1_sample[0, :num_samples, 0], z2_sample[0, :num_samples, 0], z3_sample[0, :num_samples, 0]], axis=-1), dtype=torch.float32, device=ts.device)
    noise_std=0.01
    # xs = StochasticLorenz().sample(_y0, ts, noise_std, normalize=False)
    # right plot: data.
    z1, z2, z3 = np.split(xs.cpu().numpy(), indices_or_sections=3, axis=-1)
    [ax00.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
    ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], z3[0, :num_samples, 0], marker='x')
    ax00.set_yticklabels([])
    ax00.set_xticklabels([])
    ax00.set_zticklabels([])
    ax00.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax00.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax00.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    ax00.set_title('Data', fontsize=20)
    ax00.set_xlim(xlim)
    ax00.set_ylim(ylim)
    ax00.set_zlim(zlim)

    # Plot the empirical measure (data)
    plot_empirical_measure(z1[:, :, 0], z3[:, :, 0], ax10)
    ax10.set_xlim(xlim)
    ax10.set_ylim(zlim)

    # Plot the empirical measure (samples)
    plot_empirical_measure(z1_sample[:, :, 0], z3_sample[:, :, 0], ax11)
    ax11.set_xlim(xlim)
    ax11.set_ylim(zlim)
    

    plt.savefig(train_dir+'attractor.pdf') #img_path
    plt.close()

    # fig_3d, axs_3d = plt.subplots(2, 3, figsize=(28, 12))
    # plot_individual_empirical_measures(z1[:, :, 0], z2[:, :, 0], z3[:, :, 0], axs_3d[0,:])
    # plot_individual_empirical_measures(z1_sample[:, :, 0], z2_sample[:, :, 0], z3_sample[:, :, 0], axs_3d[1,:])
    # plt.savefig(img_path) #'./dump/lorenz_ode_may8/measure.pdf'
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    plot_combined_empirical_measures(z1[:, :, 0], z2[:, :, 0], z3[:, :, 0], 
                                  z1_sample[:, :, 0], z2_sample[:, :, 0], z3_sample[:, :, 0], norms_vis, norms_data, axs)

    plt.savefig(train_dir+'e_measure.pdf') #
    plt.close()

    return xs, xs_nn
   
def rk4(x, f, dt):
    
    # x=model.projector(x)
    # print(f"x shape: {x.shape}")
    k1 = f(0, x)
    increm=dt*k1/2
    # print(f"x + dt*k1/2 shape: {x.shape}-{increm.shape}") 
    k2 = f(0, x + dt*k1/2)
    k3 = f(0, x + dt*k2/2)
    k4 = f(0, x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def inverse_projector(xs, projector):
    """
    Calculate A^{-1}(x - b), where A and b are the weight and bias of a given linear layer.

    Args:
    xs (torch.Tensor): Input tensor in the data space.
    projector (torch.nn.Linear): Linear layer.

    Returns:
    torch.Tensor: The corresponding latent space tensor.
    """
    
    A = projector.weight  #(3, latent_size)
    b = projector.bias    #(3,)
    A_inv = torch.linalg.pinv(A)  #(latent_size, 3)
    xs_centered = xs - b
    zs = torch.matmul(xs_centered, A_inv.T)  #(num_samples, latent_size)

    return zs

def transform_vector_field(model, t, xs):
    """
    Transforms data space vector to latent space, computes vector field, and projects back to the data space.

    Args:
    model: The model containing the encoder RNN, latent vector field, and projector.
    t: Scalar time value (e.g., torch.Tensor([0.0])).
    xs: Tensor containing input data samples in the data space.

    Returns:
    Tensor: The vector field transformed back to the data space.
    """
    t = torch.as_tensor(t)

    # Encode the input data
    ctx = model.encoder(xs)

    # Contextualize with time and context
    model.contextualize((t, ctx))
    ts, ctx0 = model._ctx

    # Obtain the mean of the latent distribution
    qz0_mean, _ = model.qz0_net(ctx).chunk(chunks=2, dim=1)

    # Compute the inverse projector to transform the data space vectors to latent space
    zs = inverse_projector(xs, model.projector)

    # Compute the vector field in the latent space using the model's f_net
    vh_latent = model.f_net(torch.cat((zs, ctx), dim=1))

    # Project the latent vector field back to the data space using the projector
    projector_weights = model.projector.weight  # Shape: (data_size, latent_size)
    projector_bias = model.projector.bias

    v_data = torch.matmul(vh_latent, projector_weights.T)



    return v_data

def transform_vector_field_mean(model, t, xs):
    """
    Transforms data space vector to latent space, computes vector field, and projects back to data space.
    
    Args:
    x_data (Tensor): Input tensor in the data space.

    Returns:
    Tensor: The vector field transformed back to the data space.
    """

    t = torch.as_tensor(t)
    
    ctx = model.encoder(xs) #x_t
    # ctx = torch.flip(ctx, dims=(0,))
    # print('ctx shape:', ctx[0,:].shape)
    model.contextualize((t, ctx))
    ts, ctx0 = model._ctx

    qz0_mean, qz0_log_var = model.qz0_net(ctx).chunk(chunks=2, dim=1) #m_phi, Sigma_{\phi}
    qz0_std = torch.exp(0.5 * qz0_log_var)

    num_samples = 100

    z0_samples = qz0_mean.unsqueeze(0) + qz0_std.unsqueeze(0) * torch.randn(num_samples, *qz0_mean.shape, device=xs.device)

    ts = torch.linspace(0, 0.01, 2)    
    zs = torchsde.sdeint(model, z0_samples[:,0,:], ts, names={'drift': 'h'}, dt=1e-3)#

    
    _xs = model.projector(zs)
    vf = (_xs[-1]-xs[-1])/0.01
    # print('shape', vf.shape)
    vf = vf.mean(dim=0)
    return vf

def plot_vector_field(model, path, idx, t, N, device='cuda'):
    # Credit: https://torchdyn.readthedocs.io/en/latest/_modules/torchdyn/utils.html

    x = torch.linspace(-50, 50, N)
    y = torch.linspace(-50, 50, N)
    X, Y = torch.meshgrid(x,y)
    Z_random = torch.randn(1)*10
    U, V = np.zeros((N,N)), np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if idx == 1:
                phi = torch.stack([X[i,j], Y[i,j], torch.tensor(1.)]).to('cuda')
            else:
                phi = torch.stack([X[i,j], torch.tensor(0), Y[i,j]]).to('cuda')
            phi = phi.unsqueeze(0)
            O = model(0., phi).detach().cpu().numpy()
            if O.ndim == 1:
                U[i,j], V[i,j] = O[0], O[idx]
            else:
                U[i,j], V[i,j] = O[0, 0], O[0, idx]

    fig = plt.figure(figsize=(5,4))
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
    # tight_layout()
    plt.savefig(path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.close()
    return


def latentsde_map(x, model, dt):

    ts = torch.linspace(0, dt, 2, device=device)  
    z0 = inverse_projector(x, model.projector)

    # zs = rk4_solver_h(model, z0, ts, dt=1e-2)[-1] 
    zs = euler_maruyama(model, z0, ts, dt=1e-2)[-1]  # Simulating
    y = model.projector(zs)
    return y

def calculate_batched_lyapunov_exponents(xs, g, dt=0.01):
    num_batches, num_dims = xs.size()
    exponents = torch.zeros(num_dims, device=xs.device)
    num_steps=4000
    
    q = torch.rand(num_dims, num_dims, device=xs.device)

    for i in range(1, num_steps):
        
        xs = g(xs)
        J0 = torch.autograd.functional.jacobian(g, xs)

        J = [J0[in_out_pair, :, in_out_pair, :] for in_out_pair in range(num_batches)]
        J = torch.stack(J, dim=0).cuda()
        q_j, r_j = torch.linalg.qr(J @ q)
        q = q_j
        f = lambda A: torch.log(torch.abs(torch.diag(A)))
        exponents += torch.vmap(f)(r_j).sum(dim=0) #
        if i % 100 == 0:
            print('i:', i)
            print(exponents / (i * num_batches)/dt, 'ex')
    
    return exponents / (num_steps * num_batches)/dt

def calculate_batched_lyapunov_exponents_lorenz(xs, g, dt=0.01):
    num_batches, num_dims = xs.size()
    exponents = torch.zeros(num_dims, device=xs.device)
    num_steps=400
    
    # J = torch.vmap(torch.func.jacrev(g)) #batch, dim, dim
    
    # print('J', J.shape)
    # J = torch.vmap()
    q = torch.rand(num_dims, num_dims, device=xs.device)
    # print('J', J(xs))
    # Iterate over time and batches
    for i in range(1, num_steps):
        print('i:', i)
        xs = g(xs)
        # print('Jshape', J.shape)
        q_j, r_j = torch.linalg.qr(J(xs) @ q)
        q = q_j
        f = lambda A: torch.log(torch.abs(torch.diag(A)))
        exponents += torch.vmap(f)(r_j).sum(dim=0) #
        print(exponents.shape, 'ex')
    
    # Normalizing exponents
    return exponents / (num_steps * num_batches)/dt
    
def main(
        batch_size=1024,
        latent_size=4,
        context_size=64,
        hidden_size=128,
        lr_init=1e-2,
        t0=0.,
        t1=6., 
        lr_gamma=0.997,
        num_iters=5000,
        kl_anneal_iters=1000,
        pause_every=500,
        noise_std=0.01,
        adjoint=False,
        train_dir='./dump/lorenz_ode_may11_stable5/',
        method="euler",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss()
    xs, ts, norms_data = make_dataset(t0=t0, t1=t1, batch_size=batch_size, noise_std=noise_std, train_dir=train_dir, device=device)
    latent_sde = LatentSDE(
        data_size=3,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)

    # dyn_sys_func = lorenz_batch #if args.dyn_sys == "lorenz" else rossler
    # dyn_sys_info = [dyn_sys_func, 3, 1e-2]
    # dataset = create_data(dyn_sys_info, n_train=4000, n_test=1000, n_trans=0, n_val=1000)
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset
    # X_train = X_train.to(device)
    
    ################todo
        # Fix the same Brownian motion for visualization.
    # bm_vis = torchsde.BrownianInterval(
        # t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")
    bm_vis=None
    ###########################

    if os.path.exists(os.path.join(train_dir, 'model5000_lr1e-2.pth')): #model8k+500_lr1e-4
        print('load pretrained model')
        latent_sde.load_state_dict(torch.load(os.path.join(train_dir, 'model5000_lr1e-2.pth')))
    else:
        optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)


        for global_step in tqdm.tqdm(range(1, num_iters + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio = latent_sde(xs, ts, noise_std, adjoint, method)
            
            # print('=====Shape of log_pxs:', log_pxs.shape, 'Shape of log_ratio:', log_ratio.shape)
            loss = -log_pxs + log_ratio * kl_scheduler.val #+ reg_param * jac_norm_diff

            loss.backward()
            clip_grad_norm_(latent_sde.parameters(), max_norm=1.0, norm_type=2)  # Clip gradients to prevent explosion

            optimizer.step()
            # scheduler.step()
            kl_scheduler.step()

            if global_step % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.warning(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
                )
                # print('jac_norm_diff:', jac_norm_diff, '-log_pxs + log_ratio * kl_scheduler.val ', -log_pxs + log_ratio * kl_scheduler.val )
                img_path = os.path.join(train_dir, f'global_step_{global_step:06d}.pdf')
                ts0 = torch.linspace(t0, 10, steps=1000, device=device)
                traj = vis(xs, ts, latent_sde, bm_vis, norms_data, norms_data, img_path)
                print('---')
                torch.save(latent_sde.state_dict(), os.path.join(train_dir, f'model{global_step}_lr1e-2.pth'))


    print('plotting:')
    from torchdiffeq import odeint
    img_path = os.path.join(train_dir, f'test.pdf')
    # xs, ts, norms_vis = make_vis_dataset(t0=t0, t1=10, batch_size=batch_size*4, noise_std=noise_std, train_dir=train_dir, device=device)
    # ts0 = torch.linspace(t0, 10, steps=1000, device=device)
    dim=3
    time_step = 0.01
    torch.cuda.empty_cache()
    # with torch.no_grad():
    _y0 = torch.randn(1, 3, device=device)
    ts0 = torch.linspace(0, 2, steps=1000, device=device)
    true_plot_path_1 = train_dir+"True_vf.png"
    nn_plot_path_1 = train_dir+"nn_vf.png"
    
    nn_func = lambda t,x: transform_vector_field(latent_sde, t, x)
    plot_vector_field(nn_func, path=nn_plot_path_1, idx=1, t=0., N=100, device='cuda')


    print('LE of NN')
    # xs = traj[0]
    xs1 = latent_sde.sample(batch_size=10, ts=ts, bm=bm_vis)#.cpu() #.numpy()
    lorenz = trueLorenz()
    # x_t = torch.rand([2,3], device=xs.device)
    dt=0.01
    time1_lorenz = lambda x: rk4_step(lorenz,x, 0,dt)
    # 
    time1_latentsde = lambda x: latentsde_map(x, latent_sde, dt)

    print(calculate_batched_lyapunov_exponents(xs1[0], time1_latentsde, dt))

    ts = torch.linspace(t1, 300, steps=20000, device=device)
    # xs_nn = latent_sde.sample(batch_size=xs.size(1)*16, ts=ts).cpu().numpy() #, bm=bm_vis
    bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(batch_size*4, latent_size,), device=device, levy_area_approximation="space-time")
    vis(xs, ts, latent_sde, bm_vis, norms_data, norms_data, train_dir+'testensemble', num_samples=30)


    ts = torch.linspace(t0, 600, steps=60000, device=device)
    bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(1, latent_size,), device=device, levy_area_approximation="space-time")
    vis1(xs[:,0,:].unsqueeze(1), ts, lorenz, latent_sde, bm_vis, norms_data, norms_data, train_dir+'testtime', num_samples=1)
if __name__ == "__main__":
    fire.Fire(main)
