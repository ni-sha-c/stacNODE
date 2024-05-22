import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
import scipy

import sys
sys.path.append('..')

from src.util import *

from torch.utils.data import DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected

### Compute Metric ###
def rk4(x, f, dt):
    k1 = f(0, x)
    k2 = f(0, x + dt*k1/2)
    k3 = f(0, x + dt*k2/2)
    k4 = f(0, x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def lyap_exps_fno(dyn_sys_info, ds_name, traj, iters, batch_size):
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
        Jac = torch.randn(traj_gpu.shape[0], dim, dim)
        i = 0

        for traj in traj_loader:
            print("i", i)
            jac = torch.func.jacrev(model)
            x = traj[0].unsqueeze(dim=2).to('cuda')
            cur_model_J = jac(x)
            squeezed_J = cur_model_J[:, :, 0, :, :, 0]
            learned_J = [squeezed_J[in_out_pair, :, in_out_pair, :] for in_out_pair in range(batch_size)]
            learned_J = torch.stack(learned_J, dim=0).cuda()
            Jac[i:i+batch_size] = learned_J.detach().cpu()
            i +=batch_size
        print(Jac)

    Q = torch.rand(dim,dim).to(device)
    eye_cuda = torch.eye(dim).to(device)
    for i in range(iters):
        if i > 0 and i % 1000 == 0:
            print("Iteration: ", i, ", LE[0]: ", LE[0].detach().cpu().numpy()/i/time_step)
        Q = torch.matmul(Jac[i].to('cuda'), Q)
        Q, R = torch.linalg.qr(Q)
        LE += torch.log(abs(torch.diag(R)))
    return LE/iters/time_step

# Function to plot histograms for three models in one subplot
def plot_histograms(ax, data_true, data_learned, data_mse, title, first, idx):
    bins = np.linspace(min(np.min(data_true), np.min(data_learned), np.min(data_mse)), max(np.max(data_true), np.max(data_learned), np.max(data_mse)), 500)

    # ax.hist(data_mse, bins=bins, alpha=0.8, label='MSE', color='turquoise', histtype='step', linewidth=2., density=True)
    if idx == 0:
        ax.hist(data_true, bins=bins, alpha=0.6, label='True', color='black', histtype='step', linewidth=5., density=True)
    elif (idx == 1) or (idx == 2):
        ax.hist(data_true, bins=bins, alpha=0.6, label='True', color='black', histtype='step', linewidth=5., density=True)
        ax.hist(data_learned, bins=bins, alpha=0.7, label='Model', color='red', histtype='step', linewidth=5., density=True)
    else:
        ax.hist(data_true, bins=bins, alpha=0.6, label='True', color='black', histtype='step', linewidth=5., density=True)
        ax.hist(data_learned, bins=bins, alpha=0.7, label='Model', color='blue', histtype='step', linewidth=5., density=True)

    ax.set_title(title, fontsize=45)
    ax.xaxis.set_tick_params(labelsize=45)
    ax.yaxis.set_tick_params(labelsize=45)
    ax.legend(fontsize=45)
    return

# generate traj, save it on csv
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. define system
dyn_sys= "lorenz"
dyn, dim = define_dyn_sys(dyn_sys)
time_step= 0.01
ind_func = 0
s = 0.2
hidden = 256
# model = 'MLP'
num_trajectories = 5000
long_len_T = 5*int(1/time_step)
init = "outside"
n_train = 4000
n_test = 3000
batch_size = 5
model = "MSE"
statistics = False

# call FNO
model_mse = FNO(
        in_channels=3,
        out_channels=3,
        num_fno_modes=4,
        padding=5,
        dimension=1,
        latent_channels=128
    ).to('cuda')
model_mse.load_state_dict(torch.load('../plot/Vector_field/lorenz/best_model_FNO_MSE.pth'))
model = FNO(
    in_channels=3,
    out_channels=3,
    num_fno_modes=3,
    padding=4,
    dimension=1,
    latent_channels=128
).to('cuda')
model.load_state_dict(torch.load('../plot/Vector_field/lorenz/best_model_FNO_JAC.pth'))

model.eval()
model_mse.eval()
# compute LE
init = torch.randn(dim)
true_traj = torchdiffeq.odeint(lorenz, torch.randn(dim), torch.arange(0, 500, 0.01), method='rk4', rtol=1e-8)

init_point = torch.randn(dim)
learned_traj = torch.empty_like(true_traj).cuda()
learned_traj_mse = torch.empty_like(true_traj).cuda()
learned_traj[0] = init_point
learned_traj_mse[0] = init_point
for i in range(1, len(learned_traj)):
    learned_traj[i] = model(learned_traj[i-1].reshape(1, dim, 1).cuda()).reshape(dim,)
for i in range(1, len(learned_traj_mse)):
    learned_traj_mse[i] = model_mse(learned_traj_mse[i-1].reshape(1, dim, 1).cuda()).reshape(dim,)

if statistics == True:
    print("Computing LEs of NN...")
    learned_LE = lyap_exps_fno([model, dim, 0.01], "lorenz", learned_traj, true_traj.shape[0], batch_size).detach().cpu().numpy()
    print("learned_LE", learned_LE)
    print("Computing true LEs...")
    True_LE = lyap_exps_fno([lorenz, dim, 0.01], "lorenz", true_traj, true_traj.shape[0], batch_size).detach().cpu().numpy()
    print("True LE", True_LE)
    norm = torch.norm(torch.tensor(learned_LE) - torch.tensor(True_LE))
    print("Norm Diff:", norm)

    true_long = true_traj.detach().cpu().numpy()
    learned_long = learned_traj.detach().cpu().numpy()

    torch.set_printoptions(sci_mode=False, precision=5)

    # Compute Wasserstein Distance
    dist_x = scipy.stats.wasserstein_distance(learned_long[:, 0], true_long[:, 0])
    dist_y = scipy.stats.wasserstein_distance(learned_long[:, 1], true_long[:, 1])
    dist_z = scipy.stats.wasserstein_distance(learned_long[:, 2], true_long[:, 2])
    print("JAC", dist_x, dist_y, dist_z)
    print("JAC", torch.norm(torch.tensor([dist_x, dist_y, dist_z])))

    # Compute Time avg
    ta_x = np.mean(learned_long[:, 0]) - np.mean(true_long[:, 0])
    ta_y = np.mean(learned_long[:, 1]) - np.mean(true_long[:, 1])
    ta_z = np.mean(learned_long[:, 2]) - np.mean(true_long[:, 2])
    print("Time Avg JAC", ta_x, ta_y, ta_z)
    print(torch.norm(torch.tensor([ta_x, ta_y, ta_z])))
else:

    true_long = true_traj.detach().cpu().numpy()
    learned_long = learned_traj.detach().cpu().numpy()
    learned_long_mse = learned_traj_mse.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(28, 7))  # 2 rows (time, ensemble) x 3 columns (x, y, z)
    dimensions = ['X', 'Y', 'Z']
    title = ['TRUE', 'MSE_FNO', 'JAC_FNO']

    models = [true_long, learned_long, learned_long_mse]

for j in range(3): 
    m = models[j]
    index = 0
    if j == 0:
        plot_histograms(axes[j], true_long[:, index], m[:, index], m[:, index], f'{title[j]}', True, j)
    else:
        plot_histograms(axes[j], true_long[:, index], m[:, index], m[:, index], f'{title[j]}', False, j)
    # plot_histograms(axes[1, j], true_short[:, :, j].flatten(), learned_short[:, :, j].flatten(), mse_short[:, :, j].flatten(), f'Ensemble Avg - {dimensions[j]}')

plt.tight_layout()
pdf_path = "../plot/FNO_dist.jpg"
plt.savefig(pdf_path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
# plt.show()

