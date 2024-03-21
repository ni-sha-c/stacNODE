import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
import datetime
import numpy as np
import argparse
import json
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
# rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })

########################
### Dynamical System ###
########################

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

class ODE_MLP(nn.Module):
    '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''

    def __init__(self, y_dim=3, n_hidden=512):
        super(ODE_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, 3)
        )

    def forward(self, t, y):
        res = self.net(y)
        return res

class ODE_HigherDim_CNN(nn.Module):

    def __init__(self, y_dim=3, n_hidden=512):
        super(ODE_HigherDim_CNN, self).__init__()
        self.emb = nn.Sequential(
                nn.Linear(3, 9),
                nn.Linear(9, 9))
        self.conv2d = nn.Conv2d(1, 1, kernel_size=(3,3), padding=1, bias=False)
        self.net = nn.Sequential(
            nn.Linear(9, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, 3)
        )

    def forward(self, t, y):
        matrix = self.emb(y)
        matrix = matrix.reshape(-1, 1, 3, 3)

        # y = torch.matmul(matrix.reshape(-1, 3, 3), y.T)
        # y = self.conv2d(y)
        y = self.conv2d(matrix)
        y = self.conv2d(y)
        y = self.conv2d(y)
        res = self.net(y.reshape(-1, 9))
        return res

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



##############
## Training ##
##############

def create_data(dyn_info, n_train, n_test, n_trans):
    dyn, dim, time_step = dyn_info
    tot_time = time_step*(n_train+n_test+n_trans+1) 
    t_eval_point = torch.arange(0,tot_time,time_step)
    traj = torchdiffeq.odeint(dyn, torch.randn(dim), t_eval_point, method='rk4', rtol=1e-8) 
    traj = traj[n_trans:]
    ##### create training dataset #####
    X = traj[:n_train]
    Y = traj[1:n_train+1]
    ##### create test dataset #####
    traj = traj[n_train:]
    X_test = traj[:n_test]
    Y_test = traj[1:n_test+1]
    return [X, Y, X_test, Y_test]

def update_lr(optimizer, epoch, total_e, origin_lr):
    """ A decay factor of 0.1 raised to the power of epoch / total_epochs. Learning rate decreases gradually as the epoch number increases towards the total number of epochs. """
    new_lr = origin_lr * (0.1 ** (epoch / float(total_e)))
    for params in optimizer.param_groups:
        params['lr'] = new_lr
    return

def train(dyn_sys_info, model, device, dataset, optim_name, criterion, epochs, lr, weight_decay, reg_param, loss_type):

    # Initialize
    n_store, k  = 100, 0
    ep_num, loss_hist, test_loss_hist = torch.empty(n_store+1,dtype=int), torch.empty(n_store+1), torch.empty(n_store+1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    num_train = X.shape[0]
    dyn_sys, dim, time_step = dyn_sys_info
    t_eval_point = torch.linspace(0, time_step, 2).to(device)

    torch.cuda.empty_cache()
    
    # Compute True Jacobian
    if loss_type == "Jacobian":
        print("Jacobian loss!")
        f = lambda x: lorenz(0, x)
        true_jac_fn = torch.vmap(torch.func.jacrev(f))
        True_J = true_jac_fn(X)

    # Training Loop
    for i in range(epochs):
        model.train()
        y_pred = torchdiffeq.odeint(model, X, t_eval_point, method="rk4")[-1]
        y_pred = y_pred.to(device)
        optimizer.zero_grad()
        train_loss = criterion(y_pred, Y)  * (1/time_step/time_step)

        if loss_type == "Jacobian":
            # # Compute Jacobian
            jacrev = torch.func.jacrev(model, argnums=1)
            compute_batch_jac = torch.vmap(jacrev, in_dims=(None, 0))
            cur_model_J = compute_batch_jac(0, X).to(device)
            train_loss += reg_param*criterion(True_J, cur_model_J)

        train_loss.backward()
        optimizer.step()
        update_lr(optimizer, i, epochs, args.lr)

        # Save Training and Test History
        if i % (epochs//n_store) == 0 or (i == epochs-1):
            if loss_type == "Jacobian":
                JAC_plot_path = '../plot/Vector_field/train_cnn/JAC_'+str(i)+'.jpg'
                plot_vector_field(model, path=JAC_plot_path, idx=1, t=0., N=100, device='cuda')

            with torch.no_grad():
                model.eval()
                y_pred_test = torchdiffeq.odeint(model, X_test, t_eval_point, rtol=1e-9, atol=1e-9, method="rk4")[-1]
                y_pred_test = y_pred_test.to(device)
                # save predicted node feature for analysis            
                test_loss = criterion(y_pred_test, Y_test) * (1/time_step/time_step)
                print("Epoch: ", i, " Train: {:.5f}".format(train_loss.item()), " Test: {:.5f}".format(test_loss.item()))

                ep_num[k], loss_hist[k], test_loss_hist[k] = i, train_loss.item(), test_loss.item()
                k = k + 1

    if loss_type == "Jacobian":
        for i in [0, 1, -2, -1]:
            print("Point:", X[i].detach().cpu().numpy(), "\n", "True:", "\n", True_J[i].detach().cpu().numpy(), "\n", "JAC:", "\n", cur_model_J[i].detach().cpu().numpy())

    return ep_num, loss_hist, test_loss_hist



##############
#### Plot ####
##############

def plot_loss(epochs, train, test):
    fig, ax = subplots()
    ax.plot(epochs[30:].numpy(), train[30:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Train")
    ax.plot(epochs[30:].numpy(), test[30:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Test")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    savefig('../plot/loss.png', bbox_inches ='tight', pad_inches = 0.1)

def plot_attractor(model, dyn_info, time):
    # generate true orbit and learned orbit
    dyn, dim, time_step = dyn_info
    tran_orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    true_o = torchdiffeq.odeint(dyn, tran_orbit[-1], torch.arange(0, time, time_step), method='rk4', rtol=1e-8)
    learned_o = torchdiffeq.odeint(model.eval().to(device), tran_orbit[-1].to(device), torch.arange(0, time, time_step), method="rk4", rtol=1e-8).detach().cpu().numpy()

    # create plot
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
    fig.savefig("../plot/Phase_plot/phase_plot.png", format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    return

def plot_vf(model, dyn_info):
    dyn, dim, time_step = dyn_info
    orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    orbit = torchdiffeq.odeint(dyn, orbit[-1], torch.arange(0, 4, time_step), method='rk4', rtol=1e-8)
    len_o = orbit.shape[0]
    vf_nn = model(0, orbit.to('cuda')).detach().cpu()
    vf = torch.zeros(len_o, dim)
    for i in range(len_o):
        vf[i] = dyn(0,orbit[i])
    vf_nn, vf = vf_nn.T, vf.T
    ax = figure().add_subplot()
    vf_nn, vf = vf_nn.numpy(), vf.numpy()
    #mag = np.linalg.norm(vf, axis=0)
    mag = abs(vf[2])
    #err = np.linalg.norm(vf_nn - vf, axis=0)
    err = abs(vf_nn[2]-vf[2])
    t = time_step*np.arange(0, len_o)
    ax.plot(t, err/mag*100, "o", label="err vec x-comp", ms=3.0)
    ax.set_xlabel("time",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    savefig('../plot/errx.png')


def plot_vector_field(model, path, idx, t, N, device='cuda'):
    # Credit: https://torchdyn.readthedocs.io/en/latest/_modules/torchdyn/utils.html

    x = torch.linspace(-50, 50, N)
    y = torch.linspace(-50, 50, N)
    X, Y = torch.meshgrid(x,y)
    U, V = np.zeros((N,N)), np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if idx == 1:
                phi = torch.stack([torch.tensor(X[i,j]), torch.tensor(Y[i,j]), torch.tensor(20.)]).to('cuda')
            else:
                phi = torch.stack([X[i,j].clone().detach(), torch.tensor(0), Y[i,j].clone().detach()]).to('cuda')
            O = model(0., phi).detach().cpu().numpy()
            if O.ndim == 1:
                U[i,j], V[i,j] = O[0], O[idx]
            else:
                U[i,j], V[i,j] = O[0, 0], O[0, idx]

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


def lyap_exps(dyn_sys_info, true_traj, iters, method, model):
    ''' Compute Lyapunov Exponents 
        args: path = path to model '''

    # Initialize parameter
    dyn_sys_func, dim, time_step = dyn_sys_info
    # QR Method where U = tangent vector, V = regular system
    U = torch.eye(dim)
    lyap_exp = [] #empty list to store the lengths of the orthogonal axes
    real_time = iters * time_step
    t_eval_point = torch.linspace(0, time_step, 2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_eval_point = t_eval_point.to(device)
    model.eval()

    for i in range(iters):
        if i % 1000 == 0:
            print(i)

        #update x0
        x0 = true_traj[i].to(device)
        if method == "NODE":
            cur_J = F.jacobian(lambda x: torchdiffeq.odeint(model, x, t_eval_point, method="rk4"), x0)[1]
        else:
            cur_J = F.jacobian(lambda x: torchdiffeq.odeint(dyn_sys_func, x, t_eval_point, method="rk4"), x0)[1]

        J = torch.matmul(cur_J, U.to(device))
        Q, R = torch.linalg.qr(J)
        lyap_exp.append(torch.log(abs(R.diagonal())).detach().cpu().numpy())
        U = Q #new axes after iteration

    LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]

    return LE


if __name__ == '__main__':

    # Set device
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=2000)
    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--num_test", type=int, default=6000)
    parser.add_argument("--num_trans", type=int, default=0)
    parser.add_argument("--loss_type", default="MSE", choices=["Jacobian", "MSE"])
    parser.add_argument("--model_type", default="MLP", choices=["MLP", "CNN", "HigherDimCNN", "GRU"])
    parser.add_argument("--n_hidden", type=int, default=512)
    parser.add_argument("--reg_param", type=float, default=800)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])


    # Initialize Settings
    args = parser.parse_args()
    dim = 3
    dyn_sys_info = [lorenz, dim, args.time_step]
    criterion = torch.nn.MSELoss()

    # Create Dataset
    dataset = create_data(dyn_sys_info, n_train=args.num_train, n_test=args.num_test, n_trans=args.num_trans)

    # Create model
    if args.model_type == "MLP":
        m = ODE_MLP(y_dim=dim, n_hidden=args.n_hidden).to(device)
    elif args.model_type == "CNN":
        m = ODE_CNN(y_dim=dim, n_hidden=args.n_hidden).to(device)
    elif args.model_type == "HigherDimCNN":
        m = ODE_HigherDim_CNN(y_dim=dim, n_hidden=args.n_hidden).to(device)

    print("Training...") # Train the model, return node
    epochs, loss_hist, test_loss_hist= train(dyn_sys_info, m, device, dataset, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.reg_param, args.loss_type)

    # plot things
    plot_loss(epochs, loss_hist, test_loss_hist) 
    plot_vf(m, dyn_sys_info)
    JAC_plot_path = '../plot/Vector_field/JAC.jpg'
    True_plot_path = '../plot/Vector_field/True.jpg'
    plot_vector_field(m, path=JAC_plot_path, idx=1, t=0., N=100, device='cuda')
    plot_vector_field(lorenz, path=True_plot_path, idx=1, t=0., N=100, device='cuda')
    plot_attractor(m, dyn_sys_info, 50)

    # compute LE
    true_traj = torchdiffeq.odeint(lorenz, torch.randn(dim), torch.arange(0, 300, args.time_step), method='rk4', rtol=1e-8)
    learned_LE = lyap_exps(dyn_sys_info, true_traj, 30000, "NODE", m)
    True_LE = lyap_exps(dyn_sys_info, true_traj, 30000, "rk4", m)
    print("Learned:", learned_LE, "True:", True_LE)
