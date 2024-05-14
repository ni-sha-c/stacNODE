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

import logging
import os
from typing import Sequence

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal

import torchsde


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
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
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
        xs = torchsde.sdeint(self, x0, ts)
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
        zs = euler_maruyama(self, z0, ts, dt=1e-3)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs

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
        y[i] = y[i-1] + sde.h(t, y[i-1]) * dt + sde.g(t, y[i-1]) * dw

    return y


def make_dataset(t0, t1, batch_size, noise_std, train_dir, device):
    data_path = os.path.join(train_dir, 'lorenz_data.pth')
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
        ts = torch.linspace(t0, t1, steps=100, device=device)
        xs, norms = StochasticLorenz().sample(_y0, ts, noise_std, normalize=True)

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        # torch.save({'xs': xs, 'ts': ts}, data_path)
        torch.save({'xs': xs, 'ts': ts, 'norms': norms}, data_path)
        logging.warning(f'Stored toy data at: {data_path}')
    return xs, ts, norms

def make_testdataset(t0, t1, batch_size, noise_std, train_dir, device):
    data_path = os.path.join(train_dir, 'lorenz_testdata.pth')
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
        ts = torch.linspace(t0, t1, steps=297, device=device)
        xs, norms = StochasticLorenz().sample(_y0, ts, noise_std, normalize=True)

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        # torch.save({'xs': xs, 'ts': ts}, data_path)
        torch.save({'xs': xs, 'ts': ts, 'norms': norms}, data_path)
        logging.warning(f'Stored toy data at: {data_path}')
    return xs, ts, norms

def make_fulltestdataset(t0, t1, batch_size, noise_std, train_dir, device):
    data_path = os.path.join(train_dir, 'lorenz_fulltestdata.pth')
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
        ts = torch.linspace(t0, t1, steps=297, device=device)
        xs, norms = StochasticLorenz().sample(_y0, ts, noise_std, normalize=True)

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        # torch.save({'xs': xs, 'ts': ts}, data_path)
        torch.save({'xs': xs, 'ts': ts, 'norms': norms}, data_path)
        logging.warning(f'Stored toy data at: {data_path}')
    return xs, ts, norms


# def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=10):
#     fig = plt.figure(figsize=(20, 9))
#     gs = gridspec.GridSpec(1, 2)
#     ax00 = fig.add_subplot(gs[0, 0], projection='3d')
#     ax01 = fig.add_subplot(gs[0, 1], projection='3d')

#     # Left plot: data.
#     z1, z2, z3 = np.split(xs.cpu().numpy(), indices_or_sections=3, axis=-1)
#     [ax00.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
#     ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], z3[0, :10, 0], marker='x')
#     ax00.set_yticklabels([])
#     ax00.set_xticklabels([])
#     ax00.set_zticklabels([])
#     ax00.set_xlabel('$z_1$', labelpad=0., fontsize=16)
#     ax00.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
#     ax00.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
#     ax00.set_title('Data', fontsize=20)
#     xlim = ax00.get_xlim()
#     ylim = ax00.get_ylim()
#     zlim = ax00.get_zlim()

#     # Right plot: samples from learned model.
#     xs = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).cpu().numpy()
#     z1, z2, z3 = np.split(xs, indices_or_sections=3, axis=-1)

#     [ax01.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
#     ax01.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], z3[0, :10, 0], marker='x')
#     ax01.set_yticklabels([])
#     ax01.set_xticklabels([])
#     ax01.set_zticklabels([])
#     ax01.set_xlabel('$z_1$', labelpad=0., fontsize=16)
#     ax01.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
#     ax01.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
#     ax01.set_title('Samples', fontsize=20)
#     ax01.set_xlim(xlim)
#     ax01.set_ylim(ylim)
#     ax01.set_zlim(zlim)

#     plt.savefig(img_path)
#     plt.close()

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

def plot_individual_empirical_measures(x_vals, y_vals, z_vals, axes, bins=50):
    """
    Plot the empirical measure separately for x, y, and z on given axes.

    Parameters:
    x_vals (np.ndarray): X-values of the data points.
    y_vals (np.ndarray): Y-values of the data points.
    z_vals (np.ndarray): Z-values of the data points.
    axes (list[matplotlib.axes.Axes]): List containing three subplots for x, y, and z.
    bins (int): The number of bins for the histogram.
    """
    # for vals, ax, label in zip([x_vals, y_vals, z_vals], axes, ['x', 'y', 'z']):
    #     vals_flat = vals.flatten()
    #     hist, edges = np.histogram(vals_flat, bins=bins, density=True)
    #     centers = (edges[:-1] + edges[1:]) / 2

    #     ax.plot(centers, hist, label=f'Density of ${label}$', color='blue')
    #     ax.set_xlabel(f'${label}$', fontsize=16)
    #     ax.set_ylabel('Density', fontsize=16)
    #     ax.set_title(f'Empirical Measure of ${label}$', fontsize=20)
    #     ax.grid(True)
    #     ax.legend()
    print('x', x_vals.shape, 'y', y_vals.shape, 'z', z_vals.shape)
    for vals, ax, label in zip([x_vals, y_vals, z_vals], axes, ['x', 'y', 'z']):
        vals_flat = vals.flatten()
        hist, edges = np.histogram(vals_flat, bins=bins)#, density=True
        
        # Width of each bin calculated from bin edges
        widths = np.diff(edges)
        
        # Using align='edge' to align bars with the bin edges
        ax.bar(edges[:-1], hist, width=widths, align='edge', color='blue', alpha=0.7, label=f'Density of ${label}$')
        ax.set_xlabel(f'${label}$', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title(f'Empirical Measure of ${label}$', fontsize=20)
        ax.grid(True)
        ax.legend()

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

def vis(xs, ts, latent_sde, bm_vis, norms_vis, norms_data, train_dir, num_samples=30):
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 2)
    ax01 = fig.add_subplot(gs[0, 1], projection='3d')
    ax00 = fig.add_subplot(gs[0, 0], projection='3d')
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])



    # left plot: samples from learned model.
    xs_nn = latent_sde.sample(batch_size=xs.size(1)*16, ts=ts).cpu().numpy() #, bm=bm_vis


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

def calculate_mse(test_predictions, true_data):
    mse = torch.mean((test_predictions - true_data) ** 2).item()
    return mse

# def test_model(model, test_data, device):
#     model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():  # No gradients needed for testing
#         predictions = []
#         for data_point in test_data:
#             data_point = data_point.to(device)
#             predicted = model(data_point)  # Predict using the model
#             predictions.append(predicted)
#         predictions = torch.stack(predictions)
#     return predictions
def test_model(model, test_xs, t0, t1, device, bm_vis, batchsize, train_dir):
    ts = torch.linspace(t0, t1, steps=297, device=device)  # Ensure this matches your time step resolution
    model.eval()  # Set the model to evaluation mode
    mse_errors = []
    print('test_xs:', test_xs.shape)
    initial_state = test_xs[0]  # This should be your actual initial conditions
    with torch.no_grad():  # No gradients needed for testing

        # You should ensure that 'bm_vis' is appropriately defined for testing
        predicted = model.sample(batch_size=batchsize, ts=ts, bm=bm_vis) #.cpu() #
        # true_data, _, _ = make_dataset(t1, t1+2.97, 1, 0.01, './dump/lorenz/', device)

        # print('predicted:', predicted.shape, 'test_xs:', test_xs.shape)
        mse = torch.mean((predicted - test_xs) ** 2).item()
        print(f"Average Test MSE: {mse}")

        # vis(test_xs, ts, model, bm_vis, norms_test, norms_data, train_dir+'testdata', num_samples=30)

    return mse

def main(
        batch_size=1024,
        latent_size=4,
        context_size=64,
        hidden_size=128,
        lr_init=1e-2,
        t0=0.,
        t1=2.,
        lr_gamma=0.997,
        num_iters=5000,
        kl_anneal_iters=1000,
        pause_every=500,
        noise_std=0.01,
        adjoint=False,
        train_dir='./dump/lorenz1/',
        method="euler",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xs, ts, norms_data = make_dataset(t0=t0, t1=t1, batch_size=batch_size, noise_std=noise_std, train_dir=train_dir, device=device)
    latent_sde = LatentSDE(
        data_size=3,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)
    t0_test=2
    t1_test=4.97
    test_xs, test_ts, _ = make_testdataset(t0_test, t1_test, 1, 0.01, './dump/lorenz/', device)

    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    log_pxs_values = []
    log_ratio_values = []
    mse_values = []
    loss_values = []
    if os.path.exists(os.path.join(train_dir, 'model5000_lr1e-2.pth')): #2800+600+2200
        print('load pretrained model')
        latent_sde.load_state_dict(torch.load(os.path.join(train_dir, 'model5000_lr1e-2.pth')))
    else:


        # Fix the same Brownian motion for visualization.
        bm_vis = torchsde.BrownianInterval(
            t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

        for global_step in tqdm.tqdm(range(1, num_iters + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio, mse = latent_sde(xs, ts, noise_std, adjoint, method)
            loss = -log_pxs + log_ratio * kl_scheduler.val
            loss.backward()
            optimizer.step()
            scheduler.step()
            kl_scheduler.step()

            log_pxs_values.append(log_pxs.item())
            log_ratio_values.append(log_ratio.item())
            mse_values.append(mse.item())
            loss_values.append(loss.item())

            if global_step % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.warning(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}, MSE: {mse:.4f}'
                )                # print('jac_norm_diff:', jac_norm_diff, '-log_pxs + log_ratio * kl_scheduler.val ', -log_pxs + log_ratio * kl_scheduler.val )
                img_path = os.path.join(train_dir, f'global_step_{global_step:06d}.pdf')
                traj = vis(xs, ts, latent_sde, bm_vis, norms_data, norms_data, img_path)
                torch.save(latent_sde.state_dict(), os.path.join(train_dir, f'model{global_step}_lr1e-2.pth'))
        bm_vis = torchsde.BrownianInterval(
            t0=t0_test, t1=t1_test, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")
        # test_model(latent_sde, test_xs, t1, t1 + 2.97, device, bm_vis, 1024, train_dir)

        testxs, ts, norms_test = make_fulltestdataset(t0=t0_test, t1=t1_test, batch_size=batch_size, noise_std=noise_std, train_dir=train_dir, device=device)
        vis(testxs, ts, latent_sde, bm_vis, norms_test, norms_test, train_dir+'testdata', num_samples=30)
        
            # Plot the MSE values
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_iters + 1), mse_values, label='MSE')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('MSE over Training Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(train_dir, 'mse_plot.pdf'))
        plt.show()

        # Save the values to a text file
        log_file_path = os.path.join(train_dir, 'training_log.txt')
        with open(log_file_path, 'w') as f:
            f.write('global_step\tlog_pxs\tlog_ratio\tmse\tloss\n')
            for i in range(num_iters):
                f.write(f'{i+1}\t{log_pxs_values[i]:.4f}\t{log_ratio_values[i]:.4f}\t{mse_values[i]:.4f}\t{loss_values[i]:.4f}\n')


    # def sample(self, batch_size, ts, bm=None):
    #     eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
    #     z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
    #     zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=bm)
    #     # Most of the times in ML, we don't sample the observation noise for visualization purposes.
    #     _xs = self.projector(zs)
    #     return _xs
    ts = torch.linspace(t0, 200, steps=20000, device=device)
    # xs_nn = latent_sde.sample(batch_size=xs.size(1)*16, ts=ts).cpu().numpy() #, bm=bm_vis
    bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(batch_size*4, latent_size,), device=device, levy_area_approximation="space-time")
    vis(xs, ts, latent_sde, bm_vis, norms_data, norms_data, train_dir+'testlongdata3', num_samples=30)



if __name__ == "__main__":
    fire.Fire(main)
