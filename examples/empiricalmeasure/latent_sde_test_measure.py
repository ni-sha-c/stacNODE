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
import torchdiffeq
from torch.nn.utils import clip_grad_norm_
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
def assert_no_grad(names, maybe_tensors):
    for name, maybe_tensor in zip(names, maybe_tensors):
        if torch.is_tensor(maybe_tensor) and maybe_tensor.requires_grad:
            raise ValueError(f"Argument {name} must not require gradient.")


def handle_unused_kwargs(unused_kwargs, msg=None):
    if len(unused_kwargs) > 0:
        if msg is not None:
            warnings.warn(f"{msg}: Unexpected arguments {unused_kwargs}")
        else:
            warnings.warn(f"Unexpected arguments {unused_kwargs}")


def flatten(sequence):
    return torch.cat([p.reshape(-1) for p in sequence]) if len(sequence) > 0 else torch.tensor([])


def convert_none_to_zeros(sequence, like_sequence):
    return [torch.zeros_like(q) if p is None else p for p, q in zip(sequence, like_sequence)]


def make_seq_requires_grad(sequence):
    return [p if p.requires_grad else p.detach().requires_grad_(True) for p in sequence]


def is_strictly_increasing(ts):
    return all(x < y for x, y in zip(ts[:-1], ts[1:]))


def is_nan(t):
    return torch.any(torch.isnan(t))


def seq_add(*seqs):
    return [sum(seq) for seq in zip(*seqs)]


def seq_sub(xs, ys):
    return [x - y for x, y in zip(xs, ys)]


def batch_mvp(m, v):
    return torch.bmm(m, v.unsqueeze(-1)).squeeze(dim=-1)


def stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


def vjp(outputs, inputs, **kwargs):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.  # noqa: 74

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    _vjp = torch.autograd.grad(outputs, inputs, **kwargs)
    return convert_none_to_zeros(_vjp, inputs)


def jvp(outputs, inputs, grad_inputs=None, **kwargs):
    # Unlike `torch.autograd.functional.jvp`, this function avoids repeating forward computation.
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.  # noqa: 88

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    dummy_outputs = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    _vjp = torch.autograd.grad(outputs, inputs, grad_outputs=dummy_outputs, create_graph=True, allow_unused=True)
    _vjp = make_seq_requires_grad(convert_none_to_zeros(_vjp, inputs))

    _jvp = torch.autograd.grad(_vjp, dummy_outputs, grad_outputs=grad_inputs, **kwargs)
    return convert_none_to_zeros(_jvp, dummy_outputs)


def flat_to_shape(flat_tensor, shapes):
    """Convert a flat tensor to a list of tensors with specified shapes.

    `flat_tensor` must have exactly the number of elements as stated in `shapes`.
    """
    numels = [shape.numel() for shape in shapes]
    return [flat.reshape(shape) for flat, shape in zip(flat_tensor.split(split_size=numels), shapes)]


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

# def sdeint_euler(sde, y0, ts, dt, names=None, logqp=False):
#     if names is None:
#         names = {'drift': 'f', 'diffusion': 'g', 'prior_drift': 'h'}

#     drift_method = getattr(sde, names.get('drift', 'f'))
#     diffusion_method = getattr(sde, names.get('diffusion', 'g'))
#     prior_drift_method = getattr(sde, names.get('prior_drift', 'h'), None) if logqp else None

#     ys = [y0]
#     log_ratio_increments = []

#     for i in range(1, len(ts)):
#         t = ts[i-1]
#         y = ys[-1]

#         f = drift_method(t, y)
#         g = diffusion_method(t, y)

#         dw = torch.randn_like(y) * torch.sqrt(torch.tensor(dt))  # Weiner increment
#         y_next = y + f * dt + g * dw
#         ys.append(y_next)

#         if logqp and prior_drift_method:
#             h = prior_drift_method(t, y)
#             # u is the difference between the true drift and prior drift normalized by the diffusion
#             u = (f - h) / g
#             log_ratio_increment = 0.5 * torch.sum(u ** 2, dim=1) * dt
#             log_ratio_increments.append(log_ratio_increment)

#     ys = torch.stack(ys)

#     if logqp:
#         log_ratio_increments = torch.stack(log_ratio_increments)
#         return ys, log_ratio_increments
#     else:
#         return ys

def sdeint_euler(sde, y0, ts, dt, names=None, logqp=False):
    if names is None:
        names = {'drift': 'f', 'diffusion': 'g', 'prior_drift': 'h'}

    drift_method = getattr(sde, names.get('drift', 'f'))
    diffusion_method = getattr(sde, names.get('diffusion', 'g'))
    prior_drift_method = getattr(sde, names.get('prior_drift', 'h'), None) if logqp else None
    if drift_method is None:
        raise AttributeError(f"Drift method '{names.get('drift', 'f')}' not found in the SDE object.")
    if diffusion_method is None:
        raise AttributeError(f"Diffusion method '{names.get('diffusion', 'g')}' not found in the SDE object.")
    if logqp and prior_drift_method is None:
        raise AttributeError(f"Prior drift method '{names.get('prior_drift', 'h')}' required for logqp but not found in the SDE object.")
    ys = [y0]
    log_ratio_increments = []

    for i in range(1, len(ts)):
        t = ts[i-1]
        y = ys[-1]

        f = drift_method(t, y)
        g = diffusion_method(t, y)
        torch.manual_seed(i)
        dw = torch.randn_like(y) * torch.sqrt(torch.tensor(dt))  # Weiner increment
        y_next = y + f * dt + g * dw
        

        if logqp and prior_drift_method:
            h = prior_drift_method(t, y)
            # u is the difference between the true drift and prior drift normalized by the diffusion
            u = (f - h) / g
            log_ratio_increment = 0.5 * torch.sum(u ** 2, dim=1, keepdim=True) * dt
            log_ratio_increments.append(log_ratio_increment)

        ys.append(y_next)
    ys = torch.stack(ys)

    if logqp:
        log_ratio_increments = torch.stack(log_ratio_increments)
        return ys, log_ratio_increments
    else:
        return ys
    
# def sdeint_euler(sde, y0, ts, dt, names=None, logqp=False):
#     if names is None:
#         names = {'drift': 'f', 'diffusion': 'g', 'prior_drift': 'h'}
    
#     drift_method = getattr(sde, names.get('drift', 'f'))
#     diffusion_method = getattr(sde, names.get('diffusion', 'g'))
#     prior_drift_method = getattr(sde, names.get('prior_drift', 'h'), None) if logqp else None

#     if drift_method is None:
#         raise AttributeError(f"Drift method '{names.get('drift', 'f')}' not found in the SDE object.")
#     if diffusion_method is None:
#         raise AttributeError(f"Diffusion method '{names.get('diffusion', 'g')}' not found in the SDE object.")
#     if logqp and prior_drift_method is None:
#         raise AttributeError(f"Prior drift method '{names.get('prior_drift', 'h')}' required for logqp but not found in the SDE object.")
    
#     ys = [y0]
#     w_paths = [torch.randn_like(y0)]
#     log_ratio_increments = []

#     for i in range(1, len(ts)):
#         t = ts[i-1]
#         y = ys[-1]

#         f = drift_method(t, y)
#         g = diffusion_method(t, y)
        

#         torch.manual_seed(i)  # Set seed for each iteration
#         w=torch.randn_like(y)

#         # if names={'prior_drift': 'h'}:
#         #     h = prior_drift_method(t, y)
            
#         # else:
#         y_next = y + f * dt + g * torch.sqrt(torch.tensor(dt)) * w
#         w_paths.append(w)

#         if logqp:
#             h = prior_drift_method(t, y)
#             log_ratio_increment = 0.5 * torch.sum(((f - h) / g) ** 2, dim=1, keepdim=True) * dt

#             # print('ginverse', g.pinverse().unsqueeze(0).permute(0, 2, 1).shape)
#             # print('f-h', (f - h).unsqueeze(0).shape)
#             # u = batch_mvp(g.pinverse().unsqueeze(-1).permute(1,0,2), (f - h))  # (batch_size, brownian_size)..unsqueeze(-1)
#             # f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
            
            
#             # log_ratio_increment = 0.5 * torch.sum(((f - h) / g) ** 2, dim=1, keepdim=True) * dt
#             # log_ratio_increment = f_logqp
#             log_ratio_increments.append(log_ratio_increment)

#         # else:
#             # y_next = y + f * dt + g * torch.sqrt(torch.tensor(dt)) * w
#         # if hdrift:
#         #     y_next = y + h * dt + g * torch.sqrt(torch.tensor(dt)) * w
#         ys.append(y_next)

#     ys = torch.stack(ys)

#     if logqp:
#         log_ratio_increments = torch.cat(log_ratio_increments)
#         return ys, log_ratio_increments #, w_paths
#     else:
#         return ys #, w_paths

def eulerint(sde, y0, ts, dt=1e-3, names=None):
    """
    Numerically integrates an SDE using the Euler-Maruyama method with flexible function names.
    
    Args:
        sde (object): An object that has methods representing the drift and diffusion of the SDE.
        y0 (Tensor): Initial state tensor of shape (batch_size, d), where d is the dimensionality of the state.
        ts (Tensor): 1-D tensor of times at which to sample the path, must be strictly increasing.
        dt (float, optional): Step size for the Euler-Maruyama method.
        names (dict, optional): Dictionary to specify custom names for the drift and diffusion methods.
                                Expected keys are 'drift' and 'diffusion'.

    Returns:
        Tensor: A tensor containing the simulated states at each time in `ts`.
    """
    # Handling custom names for drift and diffusion methods
    drift_name = names.get('drift', 'f') if names else 'f'
    diffusion_name = names.get('diffusion', 'g') if names else 'g'

    # Ensure y0 and ts are tensors
    y0 = torch.as_tensor(y0, dtype=torch.float32)
    ts = torch.as_tensor(ts, dtype=torch.float32)
    
    # Pre-allocate the output
    ys = torch.empty(len(ts), *y0.shape, dtype=y0.dtype, device=y0.device)
    ys[0] = y0

    current_y = y0
    for i in range(1, len(ts)):
        t0 = ts[i - 1]
        t1 = ts[i]
        dt = t1 - t0

        # Drift and diffusion calculations
        drift = getattr(sde, drift_name)(t0, current_y)
        diffusion = getattr(sde, diffusion_name)(t0, current_y)

        # Brownian increment (assuming scalar Brownian motion for simplicity)
        dW = torch.randn_like(current_y) * torch.sqrt(dt)

        # Euler-Maruyama update
        current_y += drift * dt + diffusion * dW
        ys[i] = current_y

    return ys
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

        # print('x0.is_cuda', x0.is_cuda)
        # print('ts.is_cuda', ts.is_cuda)


        # dW = torch.randn(size=(len(ts)-1,), device= x0.device) * torch.sqrt((ts[1] - ts[0])).to(x0.device)
        # print('dW.is_cuda', dW.is_cuda)
        # xs = torchsde.sdeint(self, x0, ts, dW)
        # xs = sdeint_euler(self, x0, ts, dt=1e-2)
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
            return xs, (mean, std)
        return xs


class StochasticLorenz(object):
    """Stochastic Lorenz attractor.

    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (0.4, 0.6, 1.3)):
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
        # xs = sdeint_euler(self, x0, ts, dt=1e-2)
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

def lorenz_batch(t, u, params=[10.0, 28.0, 8/3]):
    """ Lorenz chaotic differential equation: du/dt = f(t, u)
        t: time T to evaluate system
        u: state vector [x, y, z] in batch form (batch_size, 3)
        params: list of parameters for the Lorenz system
        return: new state vector in shape of (batch_size, 3)"""

    x, y, z = u[:, 0], u[:, 1], u[:, 2]  # Extract x, y, z components for the whole batch

    # Compute derivatives
    dx = params[0] * (y - x)
    dy = x * (params[1] - z) - y
    dz = (x * y) - (params[2] * z)

    # Stack them into a single tensor
    du = torch.stack([dx, dy, dz], dim=1)
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
            # zs, log_ratio = sdeint_euler(self, z0, ts, dt=1e-2, logqp=True)
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
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-2)
        # dW = torch.randn(size=(len(ts)-1,)) * torch.sqrt((ts[1] - ts[0])).to(z0.device)
        # zs = sdeint_euler(self, z0, ts, dt=1e-3)
        # xs = torchsde.sdeint(self, x0, ts, dW)
        # zs = torchsde.sdeint(self, z0, ts, dW, names={'drift': 'h'}, dt=1e-3)#, bm=bm
        #zs = eulerint(self, z0, ts, dt=1e-3, names={'drift': 'h', 'diffusion': 'g'}) #, names={'drift': 'h'}
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
        xs, ts, norms = data_dict['xs'], data_dict['ts'], data_dict['norms']
        logging.warning(f'Loaded toy data at: {data_path}')
        if xs.shape[1] != batch_size:
            raise ValueError("Batch size has changed; please delete and regenerate the data.")
        if ts[0] != t0 or ts[-1] != t1:
            raise ValueError("Times interval [t0, t1] has changed; please delete and regenerate the data.")
    else:
        _y0 = torch.randn(batch_size, 3, device=device)
        ts0 = torch.linspace(t0, 16, steps=8000, device=device)
        xs0 = DeterministicLorenz().sample(_y0, ts0, noise_std, normalize=False)
        ts = torch.linspace(t0, t1, steps=1000, device=device)
        xs, norms = DeterministicLorenz().sample(xs0[-1,:], ts, noise_std, normalize=True)

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
        # _y0 = torch.randn(batch_size, 3, device=device)
        # ts = torch.linspace(t0, 10, steps=1000, device=device)
        # xs = torchdiffeq.odeint(dyn, _y0, ts, method='rk4', rtol=1e-8)#StochasticLorenz().sample(_y0, ts, noise_std, normalize=True)

        # os.makedirs(os.path.dirname(data_path), exist_ok=True)
        # torch.save({'xs': xs, 'ts': ts}, data_path)
        # logging.warning(f'Stored toy data at: {data_path}')
        _y0 = torch.randn(batch_size, 3, device=device)
        ts0 = torch.linspace(t0, 16, steps=8000, device=device)
        xs0 = DeterministicLorenz().sample(_y0, ts0, noise_std, normalize=False)
        ts = torch.linspace(t0, t1, steps=5000, device=device)
        xs, norms = DeterministicLorenz().sample(xs0[-1,:], ts, noise_std, normalize=True)

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
    ax01 = fig.add_subplot(gs[0, 0], projection='3d')
    ax00 = fig.add_subplot(gs[0, 1], projection='3d')
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

    # distances, mean_dist, median_dist = compute_dimension_wise_wasserstein(xs, xs_nn)
    # print("Mean Wasserstein Distance (by dimension):", mean_dist)
    # print("Median Wasserstein Distance (by dimension):", median_dist)

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

    fig_3d, axs_3d = plt.subplots(1, 3, figsize=(28, 12))
    plot_individual_empirical_measures(z1[:, :, 0], z2[:, :, 0], z3[:, :, 0], axs_3d[0,:])
    # plot_individual_empirical_measures(z1_sample[:, :, 0], z2_sample[:, :, 0], z3_sample[:, :, 0], axs_3d[1,:])
    # plt.savefig(img_path) #'./dump/lorenz_ode_may8/measure.pdf'


    # fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    # plot_combined_empirical_measures(z1[:, :, 0], z2[:, :, 0], z3[:, :, 0], 
    #                               z1_sample[:, :, 0], z2_sample[:, :, 0], z3_sample[:, :, 0], norms_vis, norms_data, axs)

    plt.savefig(train_dir+'e_measure.pdf') #
    plt.close()

    return xs, xs_nn


def vis(xs, ts, latent_sde, bm_vis, norms_vis, norms_data, train_dir, num_samples=30):
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 2)
    ax01 = fig.add_subplot(gs[0, 0], projection='3d')
    ax00 = fig.add_subplot(gs[0, 1], projection='3d')
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

    # distances, mean_dist, median_dist = compute_dimension_wise_wasserstein(xs, xs_nn)
    # print("Mean Wasserstein Distance (by dimension):", mean_dist)
    # print("Median Wasserstein Distance (by dimension):", median_dist)

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


def vis_4(xs, ts, latent_sde, bm_vis, norms_vis, norms_data, train_dir, best_model, mse_model):
    num_samples=30
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 2)
    # ax01 = fig.add_subplot(gs[0, 0], projection='3d')
    # ax00 = fig.add_subplot(gs[0, 1], projection='3d')
    # ax10 = fig.add_subplot(gs[1, 0])
    # ax11 = fig.add_subplot(gs[1, 1])

    time_step = 1e-2
    learned_o = torchdiffeq.odeint(best_model.eval().to(device), xs[-1].to(device), torch.arange(0, 200, time_step), method="rk4", rtol=1e-8).detach().cpu().numpy()
    mse_o = torchdiffeq.odeint(mse_model.eval().to(device), xs[-1].to(device), torch.arange(0, 200, time_step), method="rk4", rtol=1e-8).detach().cpu().numpy()
    z1_jac, z2_jac, z3_jac = np.split(learned_o, indices_or_sections=3, axis=-1)
    z1_mse, z2_mse, z3_mse = np.split(mse_o, indices_or_sections=3, axis=-1)


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

    # distances, mean_dist, median_dist = compute_dimension_wise_wasserstein(xs, xs_nn)
    # print("Mean Wasserstein Distance (by dimension):", mean_dist)
    # print("Median Wasserstein Distance (by dimension):", median_dist)

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

    # [ax01.plot(z1_sample[:, i, 0], z2_sample[:, i, 0], z3_sample[:, i, 0]) for i in range(num_samples)]
    # ax01.scatter(z1_sample[0, :num_samples, 0], z2_sample[0, :num_samples, 0], z3_sample[0, :num_samples, 0], marker='x')
    # ax01.set_yticklabels([])
    # ax01.set_xticklabels([])
    # ax01.set_zticklabels([])
    # ax01.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    # ax01.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    # ax01.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    # ax01.set_title('Samples', fontsize=20)

    # xlim = ax01.get_xlim()
    # ylim = ax01.get_ylim()
    # zlim = ax01.get_zlim()

    _y0 = torch.tensor(np.stack([z1_sample[0, :num_samples, 0], z2_sample[0, :num_samples, 0], z3_sample[0, :num_samples, 0]], axis=-1), dtype=torch.float32, device=ts.device)
    noise_std=0.01
    # xs = StochasticLorenz().sample(_y0, ts, noise_std, normalize=False)
    # right plot: data.
    z1, z2, z3 = np.split(xs.cpu().numpy(), indices_or_sections=3, axis=-1)
    # [ax00.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
    # ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], z3[0, :num_samples, 0], marker='x')
    # ax00.set_yticklabels([])
    # ax00.set_xticklabels([])
    # ax00.set_zticklabels([])
    # ax00.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    # ax00.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    # ax00.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    # ax00.set_title('Data', fontsize=20)
    # ax00.set_xlim(xlim)
    # ax00.set_ylim(ylim)
    # ax00.set_zlim(zlim)

    # # Plot the empirical measure (data)
    # plot_empirical_measure(z1[:, :, 0], z3[:, :, 0], ax10)
    # ax10.set_xlim(xlim)
    # ax10.set_ylim(zlim)

    # # Plot the empirical measure (samples)
    # plot_empirical_measure(z1_sample[:, :, 0], z3_sample[:, :, 0], ax11)
    # ax11.set_xlim(xlim)
    # ax11.set_ylim(zlim)
    
    # plt.savefig(train_dir+'attractor.pdf') #img_path
    # plt.close()

    # fig_3d, axs_3d = plt.subplots(2, 3, figsize=(28, 12))
    # plot_individual_empirical_measures(z1[:, :, 0], z2[:, :, 0], z3[:, :, 0], axs_3d[0,:])
    # plot_individual_empirical_measures(z1_sample[:, :, 0], z2_sample[:, :, 0], z3_sample[:, :, 0], axs_3d[1,:])
    # plt.savefig(img_path) #'./dump/lorenz_ode_may8/measure.pdf'
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    plot_combined_empirical_measures(z1[:, :, 0], z2[:, :, 0], z3[:, :, 0], 
                                  z1_sample[:, :, 0], z2_sample[:, :, 0], z3_sample[:, :, 0],z1_jac[:, :, 0], z2_jac[:, :, 0], z3_jac[:, :, 0],z1_mse[:, :, 0], z2_mse[:, :, 0], z3_mse[:, :, 0], norms_vis, norms_data, axs)

    plt.savefig(train_dir+'e_measure.pdf') #
    plt.close()

    return xs, xs_nn


# def create_data(dyn_info, n_train, n_test, n_val, n_trans):
#     dyn, dim, time_step = dyn_info
#     # Adjust total time to account for the validation set
#     tot_time = time_step * (n_train + n_test + n_val + n_trans + 1)
#     t_eval_point = torch.arange(0, tot_time, time_step)

#     # Generate trajectory using the dynamical system
#     traj = torchdiffeq.odeint(dyn, torch.randn(dim), t_eval_point, method='rk4', rtol=1e-8)
#     traj = traj[n_trans:]  # Discard transient part

#     # Create training dataset
#     X_train = traj[:n_train]
#     Y_train = traj[1:n_train + 1]

#     # Shift trajectory for validation dataset
#     traj = traj[n_train:]
#     X_val = traj[:n_val]
#     Y_val = traj[1:n_val + 1]

#     # Shift trajectory for test dataset
#     traj = traj[n_val:]
#     X_test = traj[:n_test]
#     Y_test = traj[1:n_test + 1]

#     return [X_train, Y_train, X_val, Y_val, X_test, Y_test]

# def plot_empirical_measure(x_vals, z_vals, ax, bins=50):
#     """
#     Plot the empirical measure as a 2D histogram on the given axes.

#     Parameters:
#     x_vals (np.ndarray): X-values of the data points.
#     z_vals (np.ndarray): Z-values of the data points.
#     ax (matplotlib.axes.Axes): The axis to plot the empirical measure on.
#     bins (int): The number of bins for the 2D histogram.
#     """
#     # Flatten the arrays to consider all batches and time steps
#     x_vals_flat = x_vals.flatten()
#     z_vals_flat = z_vals.flatten()

#     # Create a 2D histogram as the empirical measure (projection on x-z plane)
#     hist, x_edges, z_edges = np.histogram2d(x_vals_flat, z_vals_flat, bins=bins, density=True)

#     x_centers = (x_edges[:-1] + x_edges[1:]) / 2
#     z_centers = (z_edges[:-1] + z_edges[1:]) / 2
#     x_grid, z_grid = np.meshgrid(x_centers, z_centers)
#     pcm = ax.pcolormesh(x_grid, z_grid, hist.T, shading='auto', cmap='plasma')
#     plt.colorbar(pcm, ax=ax, label='Density')

#     ax.set_xlabel('$z_1$', labelpad=0., fontsize=16)
#     ax.set_ylabel('$z_3$', labelpad=.5, fontsize=16)
#     ax.set_title('Empirical Measure', fontsize=20)

# def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=100):
#     fig = plt.figure(figsize=(20, 9))
#     gs = gridspec.GridSpec(2, 2)
#     ax00 = fig.add_subplot(gs[0, 0], projection='3d')
#     ax01 = fig.add_subplot(gs[0, 1], projection='3d')
#     ax10 = fig.add_subplot(gs[1, 0])
#     ax11 = fig.add_subplot(gs[1, 1])

#     # Left plot: data.
#     z1, z2, z3 = np.split(xs.cpu().numpy(), indices_or_sections=3, axis=-1)
#     [ax00.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
#     ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], z3[0, :100, 0], marker='x')
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
#     xs_nn = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).cpu().numpy()
#     z1_sample, z2_sample, z3_sample = np.split(xs_nn, indices_or_sections=3, axis=-1)

#     def compute_dimension_wise_wasserstein(z1, z2):
#         """
#         Compute pairwise Wasserstein distances separately for each dimension between two sets of time series data using PyTorch.

#         Parameters:
#         z1 (torch.Tensor): First set of sample trajectories with shape (time step, batch size, 3).
#         z2 (torch.Tensor): Second set of sample trajectories with shape (time step, batch size, 3).

#         Returns:
#         dict: Dictionary containing matrices of Wasserstein distances for each dimension (x, y, z).
#         dict: Dictionary containing means of the Wasserstein distances for each dimension.
#         dict: Dictionary containing medians of the Wasserstein distances for each dimension.
#         """
#         # Initialize dictionaries to hold results
#         distances = {'x': None, 'y': None, 'z': None}
#         mean_distances = {}
#         median_distances = {}

#         # Loop over each dimension
#         for dim, label in enumerate(['x', 'y', 'z']):
#             # Extract the time series for the current dimension
#             z1_dim = z1[:, :, dim]
#             z2_dim = z2[:, :, dim]

#             # Since it's a 2D slice, no need to transpose dimensions, just convert to numpy if using scipy.stats
#             z1_flat = z1_dim#.cpu().numpy() 
#             z2_flat = z2_dim#.cpu().numpy()
            

#             # Create a matrix to store distances for the current dimension
#             dim_distances = np.zeros((z1_flat.shape[0], z2_flat.shape[0]))

#             # Compute pairwise Wasserstein distances
#             for i in range(z1_flat.shape[0]):
#                 for j in range(z2_flat.shape[0]):
#                     # print('z1flat type', type(z1_flat[i]), z1_flat[i].shape)
#                     # print('z2flat type', type(z2_flat[j]), z2_flat[j].shape)
#                     u = z1_flat[i].cpu().numpy()
#                     v = z2_flat[j] #.cpu().numpy()
#                     dim_distances[i, j] = wasserstein_distance(u, v)
#                     # dim_distances[i, j] = wasserstein_distance(z1_flat[i], z2_flat[j])

#             # Store distances and calculate summary statistics
#             distances[label] = dim_distances
#             mean_distances[label] = np.mean(dim_distances)
#             median_distances[label] = np.median(dim_distances)

#         return distances, mean_distances, median_distances

#     # distances, mean_dist, median_dist = compute_dimension_wise_wasserstein(xs, xs_nn)
#     # print("Mean Wasserstein Distance (by dimension):", mean_dist)
#     # print("Median Wasserstein Distance (by dimension):", median_dist)

#     # [ax01.plot(z1_sample[i,:, 0], z2_sample[i, :, 0], z3_sample[i,:, 0]) for i in range(num_samples)]
#     # ax01.scatter(z1_sample[:num_samples,:, 0], z2_sample[:num_samples,:, 0], z3_sample[:10,:, 0], marker='x')
#     # ax01.set_yticklabels([])
#     # ax01.set_xticklabels([])
#     # ax01.set_zticklabels([])
#     # ax01.set_xlabel('$z_1$', labelpad=0., fontsize=16)
#     # ax01.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
#     # ax01.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
#     # ax01.set_title('Samples', fontsize=20)
#     # ax01.set_xlim(xlim)
#     # ax01.set_ylim(ylim)
#     # ax01.set_zlim(zlim)

#     [ax01.plot(z1_sample[:, i, 0], z2_sample[:, i, 0], z3_sample[:, i, 0]) for i in range(num_samples)]
#     ax01.scatter(z1_sample[0, :num_samples, 0], z2_sample[0, :num_samples, 0], z3_sample[0, :100, 0], marker='x')
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
#     # Plot the empirical measure (data)
#     plot_empirical_measure(z1[:, :, 0], z3[:, :, 0], ax10)
#     ax10.set_xlim(xlim)
#     ax10.set_ylim(zlim)

#     # Plot the empirical measure (samples)
#     plot_empirical_measure(z1_sample[:, :, 0], z3_sample[:, :, 0], ax11)
#     ax11.set_xlim(xlim)
#     ax11.set_ylim(zlim)


#     plt.savefig(img_path)
#     plt.close()

#     return xs, xs_nn
    
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

# def transform_vector_field(model, t, xs):
#         """
#         Transforms data space vector to latent space, computes vector field, and projects back to data space.
        
#         Args:
#         x_data (Tensor): Input tensor in the data space.

#         Returns:
#         Tensor: The vector field transformed back to the data space.
#         """

#         t = torch.as_tensor(t)
#         ctx =model.encoder(xs)
#         model.contextualize((t, ctx))
#         ts, ctx = model._ctx
#         latent_y = ctx  
#         xs = torch.cat((xs, xs.new_zeros(size=(xs.size(0), 1))), dim=1)
#         vh_latent = model.f_net(torch.cat((xs, ctx), dim=1))  
#         v_data = model.projector(vh_latent)

#         return v_data

def inverse_projector(xs, projector):
    """
    Calculate A^{-1}(x - b), where A and b are the weight and bias of a given linear layer.

    Args:
    xs (torch.Tensor): Input tensor in the data space.
    projector (torch.nn.Linear): Linear layer.

    Returns:
    torch.Tensor: The corresponding latent space tensor.
    """
    # Extract weight matrix (A) and bias vector (b)
    A = projector.weight  # Shape: (data_size, latent_size)
    b = projector.bias    # Shape: (data_size,)
    A_inv = torch.linalg.pinv(A)  # Shape: (latent_size, data_size)
    xs_centered = xs - b
    zs = torch.matmul(xs_centered, A_inv.T)  # Shape: (num_samples, latent_size)

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

def compute_jacobian(f, xs, latent_sde):
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

            jacob = torch.autograd.functional.jacobian(f, x)
            # print('jacob shape:', jacob.shape)
            jacob = jacob.squeeze().detach()

            jacob=jacob.unsqueeze(0)
            # Detach x to prevent further graph building
            x.requires_grad_(False)
            
            if jacobs is None:
                jacobs = jacob
            else:
                jacobs = torch.cat((jacobs, jacob), dim=0)  
    # print('jacobss shape:', jacobs.shape)
    return jacobs


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

    Jac = compute_jacobian(f, xs, latent_sde)
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


def compute_jacobian(x, model):

    # Ensure x requires a gradient
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
    t = torch.as_tensor(0)
    
    ctx = model.encoder(x) #x_t
    # ctx = torch.flip(ctx, dims=(0,))
    # print('ctx shape:', ctx[0,:].shape)
    model.contextualize((t, ctx))
    ts, ctx0 = model._ctx

    qz0_mean, qz0_log_var = model.qz0_net(ctx).chunk(chunks=2, dim=1) #m_phi, Sigma_{\phi}
    qz0_std = torch.exp(0.5 * qz0_log_var)

    num_samples = 100

    z0_samples = qz0_mean + qz0_std * torch.randn(qz0_std.shape, device=x.device)

    ts = torch.linspace(0, 0.01, 2)    
    # print('z0_samples', z0_samples.shape)
    zs = torchsde.sdeint(model, z0_samples, ts, names={'drift': 'h'}, dt=1e-3)#


    y = model.projector(zs)[-1]
    if not y.requires_grad:
        raise ValueError("Output of the function does not require gradients")

    # Prepare for Jacobian computation
    batch_size, n_features = x.shape
    jacobian = torch.zeros(batch_size, y.size(1), x.size(1), device=x.device)
    # print('y shape:', y.shape)
    # print('jac shape:', jacobian.shape)
    # # Compute Jacobians for each batch element
    # for i in range(y.size(1)):
    #     grad = torch.autograd.grad(y[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
    #     jacobian[:, i, :] = grad

    # for i in range(y.size(1)):
    #     if x.grad is not None:
    #         x.grad.zero_()
    #     y[:, i].sum().backward(retain_graph=True)
    #     jacobian[:, i, :] = x.grad.data.clone()

    for i in range(y.size(1)):
        if x.grad is not None:
            x.grad.zero_()
        
        # Compute the gradient with respect to the i-th output
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, i] = 1
        grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)
        grad_tensor = grads[0]
        jacobian[:, i, :] = grad_tensor

    return jacobian



def calculate_batched_lyapunov_exponents(xs, func, dt=0.1):
    num_steps, num_batches, num_dims = xs.size()
    exponents = torch.zeros(num_dims, device=xs.device)
    
    for i in range(1, num_steps):
        x_t = xs[i-1]
        x_next = xs[i]
        
        # Compute Jacobian for the current transition for each batch
        jacobian = compute_jacobian(x_t, func)
        
        # Initialize Q as an identity matrix for each batch
        q = torch.eye(num_dims, num_dims, device=xs.device).expand(num_batches, -1, -1)
        print('q shape:', q.shape)
        print('jacobian shape:', jacobian.shape)

        for j in range(num_batches):
            # Update the orthogonal basis using the Jacobian of each batch
            q_j, r_j = torch.linalg.qr(jacobian[j] @ q[j])
            q[j] = q_j
            # Accumulate the log of the diagonal elements of R
            exponents += torch.log(torch.abs(torch.diag(r_j)))

    # Normalize by the total time and number of batches to get average exponent
    return exponents / (num_batches - 1)/num_steps

def main(
        batch_size=32,
        latent_size=4,
        context_size=64,
        hidden_size=128,
        lr_init=1e-2,
        t0=0.,
        t1=6., #8 is too long.
        lr_gamma=0.997,
        num_iters=5000,
        kl_anneal_iters=1000,
        pause_every=500,
        noise_std=0.01,
        adjoint=False,
        train_dir='.',
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

    dyn_sys_func = lorenz_batch #if args.dyn_sys == "lorenz" else rossler
    dyn_sys_info = [dyn_sys_func, 3, 1e-2]
    # dataset = create_data(dyn_sys_info, n_train=4000, n_test=1000, n_trans=0, n_val=1000)
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset
    # X_train = X_train.to(device)
    
    ################todo
        # Fix the same Brownian motion for visualization.
    # bm_vis = torchsde.BrownianInterval(
        # t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")
    bm_vis=None
    ###########################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    if os.path.exists(os.path.join(train_dir, 'model5000_lr1e-2.pth')): #model8k+500_lr1e-4
        print('load pretrained model')
        latent_sde.load_state_dict(torch.load(os.path.join(train_dir, 'model5000_lr1e-2.pth')))
    else:
        optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

        for global_step in tqdm.tqdm(range(1, num_iters + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio = latent_sde(xs, ts, noise_std, adjoint, method)
            loss = -log_pxs + log_ratio * kl_scheduler.val #+ reg_param * jac_norm_diff
            loss.backward()
            clip_grad_norm_(latent_sde.parameters(), max_norm=1.0, norm_type=2)  # Clip gradients to prevent explosion

            optimizer.step()
            kl_scheduler.step()

            if global_step % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.warning(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
                )
                img_path = os.path.join(train_dir, f'global_step_{global_step:06d}.pdf')
                ts0 = torch.linspace(t0, 10, steps=1000, device=device)
                traj = vis(xs, ts, latent_sde, bm_vis, norms_data, norms_data, img_path)
                print('---')
                torch.save(latent_sde.state_dict(), os.path.join(train_dir, f'model{global_step}_lr1e-2.pth'))

    print('plotting:')
    from torchdiffeq import odeint
    img_path = os.path.join(train_dir, f'test.png')
    dim=3
    time_step = 0.01

    torch.cuda.empty_cache()

    _y0 = torch.randn(1, 3, device=device)
    ts0 = torch.linspace(0, 2, steps=1000, device=device)
    true_plot_path_1 = train_dir+"True_vf.png"
    nn_plot_path_1 = train_dir+"nn_vf.png"
    
    nn_func = lambda t,x: transform_vector_field(latent_sde, t, x)
    # plot_vector_field(nn_func, path=nn_plot_path_1, idx=1, t=0., N=100, device='cuda')

    print('LE of NN')
    print(calculate_batched_lyapunov_exponents(xs,latent_sde))
    print('LE of true lorenz')
    print('LE of true perturbed lorenz')
    
    xs0 = xs0[0,:,:].to(device)
    xs0 = torch.cat((xs0, xs0.new_zeros(size=(xs0.size(0), 1))), dim=1)
    print('g', latent_sde.g(0, xs0)) #torch.randn(1, 4, device=device)

if __name__ == "__main__":
    fire.Fire(main)
