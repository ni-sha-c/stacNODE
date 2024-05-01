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
# device = 'cuda'
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
from scipy.stats import wasserstein_distance
import torchsde
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class DeterministicLorenz(object):
    """Stochastic Lorenz attractor.

    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (0, 0, 0)):
        super(DeterministicLorenz, self).__init__()
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

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
        return xs


class StochasticLorenz(object):
    """Stochastic Lorenz attractor.

    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (0.2, 0.2, 0.3)):
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

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
        return xs

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
        #Shape of ts: torch.Size([100]) Shape of ctx: torch.Size([100, 1024, 64]) Shape of y: torch.Size([1024, 4]) Shape of t: torch.Size([])
        # print('Shape of ts:', ts.shape, 'Shape of ctx:', ctx.shape, 'Shape of y:', y.shape, 'Shape of t:', t.shape)
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        # print('ctx[i].shape', ctx[i].shape, 'y.shape:', y.shape) 1024 64, 1024 4
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        # print('ctx===Shape of xs:', xs.shape, 'Shape of ts:', ts.shape, 'shape flip:', torch.flip(xs, dims=(0,)).shape)
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))
        # print('forwardctx shape:', ctx.shape, 'ts shape:', ts.shape)
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
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method) #z0 is latent size. zs
        # print('zs shape:', zs.shape)
        _xs = self.projector(zs)
        # print('xs shape:', xs.shape, '_xs shape:', _xs.shape)
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device) #bacthsize*latent
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=bm)#
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.

        
        _xs = self.projector(zs)
        return _xs
    # def transform_vector_field(self, t, xs):
    #     """
    #     Transforms data space vector to latent space, computes vector field, and projects back to data space.
        
    #     Args:
    #     x_data (Tensor): Input tensor in the data space.

    #     Returns:
    #     Tensor: The vector field transformed back to the data space.
    #     """
    #     # Step 1: Transform data to latent space using the encoder
    #     # if isinstance(xs, torch._vmap_internals._TensorWrapper):
    #     #     print('tensorwrapper')



    #     t = torch.as_tensor(t)
    #     # else:
    #     #     print('not tensorwrapper')
    #     print('-tvf-xs shape:', xs.shape, 't shape:', t.shape)
    #     # ctx = self.encoder(torch.flip(xs, dims=(0,))) for xs 100,batch,3
    #     ctx = self.encoder(xs)
    #     # print('ctx shape_vf:', ctx.shape)
    #     ctx = torch.flip(ctx, dims=(0,))
    #     self.contextualize((t, ctx))
    #     # ctx = self.encoder(x_data)  # Ensure that the encoder output matches the expected input for f_net

    #     # We need some way to provide a 't' and 'y' for f_net, assuming 't' can be a dummy since not used directly in f
    #     # Assuming there is a mechanism to get 't' or it's irrelevant (using last known 't' from context)
    #       # This needs to be defined appropriately if time dependency is critical
    #     ts, ctx = self._ctx
    #     # print('vf_ts :', ts, 'ctx shape:', ctx[0].shape, 'xs shape:', xs.shape, 't:', t)
    #      # Get the appropriate context index

    #     # Since f_net requires both y and context, and context is not fully prepared in encoder output
    #     # Create a dummy y to get through f_net structure; adjust as necessary based on actual requirements
    #     latent_y = ctx  # Normally, you would use a true latent representation, but ctx is being used for simplicity
    #     xs = torch.cat((xs, xs.new_zeros(size=(xs.size(0), 1))), dim=1)
    #     # Step 2: Compute the vector field in latent space using f_net
    #     vh_latent = self.f_net(torch.cat((xs, ctx), dim=1))  # Context may need adjustment based on actual model structure

    #     # Step 3: Project the latent vector field back to the data space
    #     v_data = self.projector(vh_latent)

    #     return v_data

def make_dataset(t0, t1, batch_size, noise_std, train_dir, device):
    data_path = os.path.join(train_dir, 'lorenz_data_ode.pth')
    if os.path.exists(data_path):
        data_dict = torch.load(data_path)
        xs, ts = data_dict['xs'], data_dict['ts']
        logging.warning(f'Loaded toy data at: {data_path}')
        if xs.shape[1] != batch_size:
            raise ValueError("Batch size has changed; please delete and regenerate the data.")
        if ts[0] != t0 or ts[-1] != t1:
            raise ValueError("Times interval [t0, t1] has changed; please delete and regenerate the data.")
    else:
        _y0 = torch.randn(batch_size, 3, device=device)
        ts = torch.linspace(t0, t1, steps=100, device=device)
        xs = StochasticLorenz().sample(_y0, ts, noise_std, normalize=True)

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save({'xs': xs, 'ts': ts}, data_path)
        logging.warning(f'Stored toy data at: {data_path}')
    return xs, ts


def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=10):
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2)
    ax00 = fig.add_subplot(gs[0, 0], projection='3d')
    ax01 = fig.add_subplot(gs[0, 1], projection='3d')

    # Left plot: data.
    z1, z2, z3 = np.split(xs.cpu().numpy(), indices_or_sections=3, axis=-1)
    [ax00.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
    ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], z3[0, :10, 0], marker='x')
    ax00.set_yticklabels([])
    ax00.set_xticklabels([])
    ax00.set_zticklabels([])
    ax00.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax00.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax00.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    ax00.set_title('Data', fontsize=20)
    xlim = ax00.get_xlim()
    ylim = ax00.get_ylim()
    zlim = ax00.get_zlim()

    # Right plot: samples from learned model.
    xs_nn = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).cpu().numpy()
    z1_sample, z2_sample, z3_sample = np.split(xs_nn, indices_or_sections=3, axis=-1)
    # Compute Wasserstein distance for each dimension
    wasserstein_z1 = wasserstein_distance(z1.flatten(), z1_sample.flatten())
    wasserstein_z2 = wasserstein_distance(z2.flatten(), z2_sample.flatten())
    wasserstein_z3 = wasserstein_distance(z3.flatten(), z3_sample.flatten())

    # Average Wasserstein distance
    avg_wasserstein_distance = np.mean([wasserstein_z1, wasserstein_z2, wasserstein_z3])
    print(f'Wasserstein distance for z1: {wasserstein_z1}')
    print(f'Wasserstein distance for z2: {wasserstein_z2}')
    print(f'Wasserstein distance for z3: {wasserstein_z3}')
    print(f'Average Wasserstein distance: {avg_wasserstein_distance}\n')

    # [ax01.plot(z1_sample[i,:, 0], z2_sample[i, :, 0], z3_sample[i,:, 0]) for i in range(num_samples)]
    # ax01.scatter(z1_sample[:num_samples,:, 0], z2_sample[:num_samples,:, 0], z3_sample[:10,:, 0], marker='x')
    # ax01.set_yticklabels([])
    # ax01.set_xticklabels([])
    # ax01.set_zticklabels([])
    # ax01.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    # ax01.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    # ax01.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    # ax01.set_title('Samples', fontsize=20)
    # ax01.set_xlim(xlim)
    # ax01.set_ylim(ylim)
    # ax01.set_zlim(zlim)

    [ax01.plot(z1_sample[:, i, 0], z2_sample[:, i, 0], z3_sample[:, i, 0]) for i in range(num_samples)]
    ax01.scatter(z1_sample[0, :num_samples, 0], z2_sample[0, :num_samples, 0], z3_sample[0, :10, 0], marker='x')
    ax01.set_yticklabels([])
    ax01.set_xticklabels([])
    ax01.set_zticklabels([])
    ax01.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax01.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax01.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    ax01.set_title('Samples', fontsize=20)
    ax01.set_xlim(xlim)
    ax01.set_ylim(ylim)
    ax01.set_zlim(zlim)

    plt.savefig(img_path)
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

def transform_vector_field(model, t, xs):
        """
        Transforms data space vector to latent space, computes vector field, and projects back to data space.
        
        Args:
        x_data (Tensor): Input tensor in the data space.

        Returns:
        Tensor: The vector field transformed back to the data space.
        """

        t = torch.as_tensor(t)
        ctx =model.encoder(xs)
        ctx = torch.flip(ctx, dims=(0,))
        model.contextualize((t, ctx))
        ts, ctx = model._ctx
        latent_y = ctx  # Normally, you would use a true latent representation, but ctx is being used for simplicity
        xs = torch.cat((xs, xs.new_zeros(size=(xs.size(0), 1))), dim=1)
        # vh_latent = model.f_net(torch.cat((xs, ctx), dim=1))  # Context may need adjustment based on actual model structure
        vh_latent = model.h_net(xs)
        v_data = model.projector(vh_latent)

        return v_data

def lyap_exps(xs, ts, latent_sde, bm_vis):
    xs=xs.to(device)
    # device = xs.device
    time_step = 0.01  # Assuming time_step is supposed to be a small value like 0.01 or so for numerical integration
    dim = 3
    iters = 1000
    if latent_sde==lorenz:

        f = lambda x: rk4(x, latent_sde, time_step)
    else:
        # print('Shape of xs:', xs.shape, 'Shape of ts:', ts.shape)
        ctx = latent_sde.encoder(xs)
        # print('ctx shape:', ctx.shape)
        ctx = torch.flip(ctx, dims=(0,))
        latent_sde.contextualize((ts, ctx))

        last_time = ts[-1]
        qz0_mean, qz0_logstd = latent_sde.qz0_net(ctx).chunk(chunks=2, dim=-1)

        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        x0=latent_sde.projector(z0)
        # print('x0 shape:', x0.shape) #timesteps 100,1024,3

        f = lambda x: rk4(x, lambda t,x: transform_vector_field(latent_sde, t, x), time_step) #

    def compute_jacobian(f, xs):
        # This function will calculate the Jacobian of f for each input in xs
        jacobs = None
        # for x in xs:

        if latent_sde!=lorenz:
            xs=xs[:,0,:]   
            # print('xs shape:', xs.shape)
            for i in range(xs.shape[0]):
                x = xs[i]
                
                x = x.unsqueeze(0) #for gru cell, latentsde
                x.requires_grad_(True)
                # print('x shape:', x.shape)
                # Compute the Jacobian for this specific input
                jacob = torch.autograd.functional.jacobian(f, x)
                # print('jacob shape:', jacob.shape)
                jacob = jacob.squeeze().detach()
                # Append the computed Jacobian to the list
                # jacobs.append(jacob)
                jacob=jacob.unsqueeze(0)
                # Detach x to prevent further graph building
                x.requires_grad_(False)
                
                if jacobs is None:
                    jacobs = jacob
                else:
                    jacobs = torch.cat((jacobs, jacob), dim=0)  
        else:
            xs=xs[:,0,:]  #not for odeint
            for i in range(xs.shape[0]):
                x = xs[i]
                
                
                x.requires_grad_(True)
                # print('x shape:', x.shape)
                # Compute the Jacobian for this specific input
                jacob = torch.autograd.functional.jacobian(f, x)
                # print('jacob shape:', jacob.shape)
                jacob = jacob.squeeze().detach()
                # Append the computed Jacobian to the list
                # jacobs.append(jacob)
                jacob=jacob.unsqueeze(0)
                # Detach x to prevent further graph building
                x.requires_grad_(False)
                
                if jacobs is None:
                    jacobs = jacob
                else:
                    jacobs = torch.cat((jacobs, jacob), dim=0)  
        # print('jacobss shape:', jacobs.shape)
        return jacobs

    # Example use case
    # Define your function f, ensuring it can handle a single input x
    # xs should be a tensor containing multiple inputs

    Jac = compute_jacobian(f, xs)
    # print('Jac shape:', Jac.shape) #100,3,3
    LE = torch.zeros(dim).to(device)
    # traj_gpu = xs.to(device)
    
        # Use the RK4 method with the model's drift function
    # f = lambda x: rk4(x, model, time_step, last_time)
    # Jac = torch.vmap(torch.func.jacrev(f), randomness='same')(last_state)

    # Use the RK4 method with the model's drift function
    # f = lambda x: rk4(x, lambda t, y: model(t, y), time_step)
    # Jac = torch.vmap(torch.func.jacrev(f),randomness='same')(traj_gpu)

    Q = torch.rand(dim, dim).to(device)
    eye_cuda = torch.eye(dim).to(device)
    for i in range(iters):
        if i > 0 and i % 1000 == 0:
            print("Iteration: ", i, ", LE[0]: ", LE[0].detach().cpu().numpy() / i / time_step)
        Q = torch.matmul(Jac[i], Q)
        Q, R = torch.linalg.qr(Q)
        LE += torch.log(torch.abs(torch.diag(R)))

    return LE / iters / time_step

def main(
        batch_size=1024,
        latent_size=4,
        context_size=64,
        hidden_size=128,
        lr_init=1e-2,
        t0=0.,
        t1=2.,
        lr_gamma=0.997,
        num_iters=8000,
        kl_anneal_iters=1000,
        pause_every=500,
        noise_std=0.01,
        adjoint=False,
        train_dir='./dump/lorenz_ode_fnoth/',
        method="euler",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xs, ts = make_dataset(t0=t0, t1=t1, batch_size=batch_size, noise_std=noise_std, train_dir=train_dir, device=device)
    latent_sde = LatentSDE(
        data_size=3,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)
        # Fix the same Brownian motion for visualization.
    bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

    if os.path.exists(os.path.join(train_dir, 'model8k+500_lr1e-4.pth')):
        latent_sde.load_state_dict(torch.load(os.path.join(train_dir, 'model8k+500_lr1e-4.pth')))
    else:
        optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)


        for global_step in tqdm.tqdm(range(1, num_iters + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio = latent_sde(xs, ts, noise_std, adjoint, method)
            # print('=====Shape of log_pxs:', log_pxs.shape, 'Shape of log_ratio:', log_ratio.shape)
            loss = -log_pxs + log_ratio * kl_scheduler.val
            loss.backward()
            optimizer.step()
            # scheduler.step()
            kl_scheduler.step()

            if global_step % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.warning(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
                )
                img_path = os.path.join(train_dir, f'global_step_{global_step:06d}.pdf')
                traj = vis(xs, ts, latent_sde, bm_vis, img_path)
                print('---')
                torch.save(latent_sde.state_dict(), os.path.join(train_dir, f'model{global_step}_lr1e-2.pth'))

        from torchdiffeq import odeint

        torch.cuda.empty_cache()
        with torch.no_grad():
            _y0 = torch.randn(1, 3, device=device)
            ts0 = torch.linspace(0, 2, steps=1000, device=device)
            print('LE of NN')
            # xs = traj[0]
            xs0 = latent_sde.sample(batch_size=xs.size(1), ts=ts0, bm=bm_vis).cpu() #.numpy()
            # xs1 = odeint(latent_sde.h_net, torch.randn(3), ts,method='rk4',rtol=1e-3)
            print(lyap_exps(xs0, ts0,latent_sde, bm_vis))
            print('LE of true lorenz')
            # xs = traj[1]
            xs0 = DeterministicLorenz().sample(_y0, ts0, noise_std, normalize=False).cpu() #.numpy()
            #s=odeint(lorenz, torch.randn(3), ts,method='rk4',rtol=1e-3)
            print(lyap_exps(xs0, ts0,lorenz, bm_vis))
            print('LE of true perturbed lorenz')
            # xs = traj[1]
            xs0 = StochasticLorenz().sample(_y0, ts0, noise_std, normalize=False).cpu() #.numpy()
            #s=odeint(lorenz, torch.randn(3), ts,method='rk4',rtol=1e-3)
            print(lyap_exps(xs0, ts0,lorenz, bm_vis))

            
            xs0 = xs0[0,:,:].to(device)
            xs0 = torch.cat((xs0, xs0.new_zeros(size=(xs0.size(0), 1))), dim=1)
            print('g', latent_sde.g(0, xs0)) #torch.randn(1, 4, device=device)

if __name__ == "__main__":
    fire.Fire(main)
