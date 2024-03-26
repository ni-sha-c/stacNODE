import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax


matplotlib.rcParams.update({"font.size": 30})
class Func(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * self.mlp(y)

class LatentODE(eqx.Module):
    func: Func
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int

    def __init__(
        self, *, data_size, hidden_size, latent_size, width_size, depth, key, **kwargs
    ):
        super().__init__(**kwargs)

        mkey, gkey, hlkey, lhkey, hdkey = jr.split(key, 5)

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        self.func = Func(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)

        self.hidden_to_latent = eqx.nn.Linear(hidden_size, 2 * latent_size, key=hlkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lhkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)
        # self.hidden_to_data = nn.Linear(hidden_size, data_size)

        self.hidden_size = hidden_size
        self.latent_size = latent_size

    # Encoder of the VAE
    def _latent(self, ts, ys, key):
        data = jnp.concatenate([ts[:, None], ys], axis=1) 

        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size :]
        std = jnp.exp(logstd)
        latent = mean + jr.normal(key, (self.latent_size,)) * std
        return latent, mean, std

    # Decoder of the VAE
    def _sample(self, ts, latent):
        dt0 = 0.4  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jax.vmap(self.hidden_to_data)(sol.ys)

    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    # Run both encoder and decoder during training.
    def train(self, ts, ys, *, key):
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        return self._loss(ys, pred_ys, mean, std)

    # Run just the decoder during inference.
    def sample(self, ts, *, key):
        latent = jr.normal(key, (self.latent_size,))
        return self._sample(ts, latent)



def get_data(dataset_size, *, key):
    ykey, tkey1, tkey2 = jr.split(key, 3)

    y0 = jr.normal(ykey, (dataset_size, 3))

    t0 = 0
    t1 = 2 + jr.uniform(tkey1, (dataset_size,))
    ts = jr.uniform(tkey2, (dataset_size, 20)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1

    def func(t, y, args):
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0 
        x, y, z = y  # Unpacking the state vector
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return jnp.array([dx_dt, dy_dt, dz_dt])

    # def func(t, y, args):
    #     return jnp.array([[-0.1, 1.3], [-1, -0.1]]) @ y

    def solve(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    ys = jax.vmap(solve)(ts, y0)

    return ts, ys


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

# def model_dynamics(t, y, model):
#     return model.func(t, y, None)


import jax.numpy as jnp
from jax import vmap, jacfwd, jacrev
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
import jax
from jax import random



# device = 'cuda' if torch.cuda.is_available() else 'cpu'
def lorenz_system(t, y, args):
    sigma, rho, beta = args
    x, y, z = y
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return jnp.array([dx, dy, dz])

# def lorenz_system(state, t, params=(10.0, 28.0, 8.0 / 3.0)):
#     x, y, z = state
#     sigma, rho, beta = params
#     dx = sigma * (y - x)
#     dy = x * (rho - z) - y
#     dz = x * y - beta * z
#     return jnp.array([dx, dy, dz])

params = (10.0, 28.0, 8/3)
key = jr.PRNGKey(0)
y0 = jr.normal(key, (3,)) # Initial condition
t0, t1, dt0 = 0.0, 300.0, 0.01  # Time span and initial step size
saveat = SaveAt(ts=jnp.arange(t0, t1, dt0))  # Save at these time points

solver = Dopri5()
term = ODETerm(lambda t, y, args: lorenz_system(t, y, params))

sol = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=saveat,max_steps=1000000 )

traj = sol.ys

###

import jax.numpy as jnp
from jax import jit

# Define the RK4 step function
def rk4(x, f, dt):
    k1 = f(0, x)
    k2 = f(0, x + dt * k1 / 2)
    k3 = f(0, x + dt * k2 / 2)
    k4 = f(0, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


import jax.numpy as jnp
from jax import jit, jacrev, vmap
def model_l(t, x, params=(10.0, 28.0, 8.0 / 3.0)):
    return lorenz_system(t, x, params)

@jit
def rk4_jit(x, dt, params):
    return rk4(x, lambda t, x: model_l(t, x, params), dt)

def rk4_jit_nn(x, dt, model):
    return rk4(x, lambda t, x: model.func(t, x, None), dt)


###
def lyapunov_exponent_jax(traj, dt, iters, dim, params):
    key = jax.random.PRNGKey(0)
    Q = jax.random.uniform(key, shape=(dim, dim))
    LE = jnp.zeros(dim)

    f = lambda x: rk4_jit(x, dt, params)

    for i in range(min(iters, len(traj))):
        state = traj[i]
        
        J = jacrev(f)(state)
        QJ = jnp.dot(J, Q)
        Q, R = jnp.linalg.qr(QJ)
        LE += jnp.log(jnp.abs(jnp.diag(R)))

        # if i % 1000 == 0 and i > 0:
        #     print(f"Iteration {i}: Current LE = {LE / i / dt}")
    
    return LE / iters / dt

def lyapunov_exponent_nn(traj, dt, iters, dim, model, key):
    key = jax.random.PRNGKey(0)
    Q = jax.random.uniform(key, shape=(dim, dim))
    LE = jnp.zeros(dim)

    f = lambda x: rk4_jit_nn(model,x, dt) #model, key

    # for i in range(min(iters, len(traj))):
        # state = traj[i][None, :]
        # print('state shape:', state.shape)
    ts = jnp.linspace(0, 300, 30000)
    latent, mean, std = model._latent(ts, traj, key)
    print('latent shape:', latent.shape)
    # latent_state = model.latent_to_hidden(state)latent_
    J = jacrev(f)(latent)
    # J = jacrev(f)(state)
    QJ = jnp.dot(J, Q)
    Q, R = jnp.linalg.qr(QJ)
    LE += jnp.log(jnp.abs(jnp.diag(R)))

        # if i % 1000 == 0 and i > 0:
        #     print(f"Iteration {i}: Current LE = {LE / i / dt}")
    
    return LE / iters / dt




def main(
    dataset_size=10000,
    batch_size=256,
    lr=5e-3,
    steps=20000,
    save_every=200,
    hidden_size=128,
    latent_size=128,
    width_size=16,
    depth=2,
    seed=5678,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, train_key, sample_key = jr.split(key, 5)

    ts, ys = get_data(dataset_size, key=data_key)

    model = LatentODE(
        data_size=ys.shape[-1],
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
    )

    @eqx.filter_value_and_grad
    def loss(model, ts_i, ys_i, key_i):
        batch_size, _ = ts_i.shape
        key_i = jr.split(key_i, batch_size)
        loss = jax.vmap(model.train)(ts_i, ys_i, key=key_i)
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(model, opt_state, ts_i, ys_i, key_i):
        value, grads = loss(model, ts_i, ys_i, key_i)
        key_i = jr.split(key_i, 1)[0]
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state, key_i

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    iters=30000
    print('trajectory shape:', traj.shape)
    LEs = lyapunov_exponent_jax(traj, dt0, iters, 3, params)
    print("True Lyapunov Exponents:", LEs)


    # plt.show()
    for step, (ts_i, ys_i) in zip(
        range(steps), dataloader((ts, ys), batch_size, key=loader_key)
    ):
        start = time.time()
        value, model, opt_state, train_key = make_step(
            model, opt_state, ts_i, ys_i, train_key
        )
        end = time.time()
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")
        
        if (step % save_every) == 0 or step == steps - 1:
            # print('shape of ts_i', ts_i.shape)
            # print('shape of ys_i', ys_i.shape)
            # print('shape of ts', ts.shape)
            # print('shape of ys', ys.shape)

            # sample_ts = jnp.linspace(0, 300, 30000)  
            # sample_key,_= jr.split(sample_key)
            # traj_nn = model.sample(sample_ts, key=sample_key)
            # # traj_nn = generate_sample_trajectory(model, sample_ts, sample_key)
            # print("Shape of traj_nn:", traj_nn.shape)
            # latent, mean, std = model._latent(ts_i, ys_i, key=train_key)
            # pred_ys = model_nn.sample(ts, latent)
            # 






            # Create a new figure for each save step
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            sample_t = jnp.linspace(0, 300, 30000)
            sample_y = model.sample(sample_t, key=sample_key)
            sample_t = np.asarray(sample_t)
            sample_y = np.asarray(sample_y)
  ############### Lyapunov Exponents ############################
            # LE_nn = lyapunov_exponent_nn(sample_y, dt0, iters, 3, model, key=sample_key)
            # print("Learned Lyapunov Exponents:", LE_nn)          
            ax.plot(sample_t, sample_y[:, 0])
            ax.plot(sample_t, sample_y[:, 1])
            ax.plot(sample_t, sample_y[:, 2])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("t")
            
            plt.savefig(f"latent_ode_{step}.png")
            plt.close(fig)  # Close the figure to start fresh next time




if __name__ == '__main__':

    main()