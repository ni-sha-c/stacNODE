import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def lorenz(t, u, rho=28.0):
    """ Lorenz chaotic differential equation: du/dt = f(t, u)
    t: time T to evaluate system
    u: state vector [x, y, z] 
    return: new state vector in shape of [3]"""

    sigma = 10.0
    beta = 8/3

    du = torch.stack([
            sigma * (u[1] - u[0]),
            u[0] * (rho - u[2]) - u[1],
            (u[0] * u[1]) - (beta * u[2])
        ])
    return du


class ODE_Lorenz(nn.Module):
    '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''

    def __init__(self, y_dim=3, n_hidden=32*9):
        super(ODE_Lorenz, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32 * 9),
            nn.GELU(),
            nn.Linear(32 * 9, 64 * 9),
            nn.GELU(),
            nn.Linear(64 * 9, 3)
        )

    def forward(self, t, y):
        res = self.net(y)
        return res


def plot_3d_vector_field(model, path, t, N, device='cuda'):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 

    x, y, z = np.meshgrid(np.linspace(-20, 20, N),
                        np.linspace(-20, 20, N),
                        np.linspace(-20, 20, N))

    U, V, W = torch.zeros(N,N,N), torch.zeros(N,N,N), torch.zeros(N,N,N)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                phi = torch.stack([torch.tensor(x[i,j,k]), torch.tensor(y[i,j,k]), torch.tensor(z[i,j,k])]).to('cuda').double()
                O = model(0., phi)
                U[i,j,k], V[i,j,k], W[i,j,k] = O[0], O[1], O[2]

                if (i % 10 == 0) and (j % 10 == 0):
                    print(i, O)

    ax.quiver(x, y, z, U.detach().numpy(), V.detach().numpy(), W.detach().numpy(), normalize=True, arrow_length_ratio=0.4, linewidths=0.7)
    plt.savefig(path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    return


def plot_vector_field(model, path, idx, t, N, device='cuda'):
    # Credit: https://torchdyn.readthedocs.io/en/latest/_modules/torchdyn/utils.html

    "Plots vector field and trajectories on it."
    
    # x = torch.arange(traj[:,:,0].min(), traj[:,:,0].max(), N)
    # y = torch.arange(traj[:,:,idx].min(), traj[:,:,idx].max(), N)

    if idx == 1:
        x = torch.linspace(-50, 50, N)
        y = torch.linspace(-50, 50, N)
    else:
        x = torch.linspace(-20, 20, N)
        y = torch.linspace(0, 40, N)
        
    X, Y = torch.meshgrid(x,y)
    U, V = torch.zeros(N,N), torch.zeros(N,N)
    print("shape", X.shape, Y.shape)

    for i in range(N):
        for j in range(N):
            if idx == 1:
                phi = torch.stack([torch.tensor(X[i,j]), torch.tensor(Y[i,j]), torch.tensor(0.)]).to('cuda').double()
            else:
                phi = torch.stack([X[i,j].clone().detach(), torch.tensor(0), Y[i,j].clone().detach()]).to('cuda').double()
            O = model(0., phi)
            if (i % 10 == 0) and (j % 10 == 0):
                print(i, O)
            U[i,j], V[i,j] = O[0], O[idx]

    print(X.requires_grad, Y.requires_grad, U.requires_grad, V.requires_grad)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    if U.requires_grad:
        contourf = ax.contourf(X, Y, torch.sqrt(U**2 + V**2).detach().numpy(), cmap='jet')
        ax.streamplot(X.T.numpy(),Y.T.numpy(),U.T.detach().numpy(),V.T.detach().numpy(), color='k')
    else:
        contourf = ax.contourf(X, Y, torch.sqrt(U**2 + V**2), cmap='jet')
        ax.streamplot(X.T.numpy(),Y.T.numpy(),U.T.numpy(),V.T.numpy(), color='k')

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
    plt.tight_layout()
    plt.savefig(path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)

    return


if __name__ == '__main__':
    # 1. initialize
    dyn_sys = "lorenz"
    JAC_path = 'JAC_Lorenz_model.pt'
    JAC = ODE_Lorenz(y_dim=3, n_hidden=128).to('cuda').double()
    JAC.load_state_dict(torch.load(JAC_path))
    JAC.eval()

    # 2. plot
    JAC_plot_path = '../plot/Vector_field/'+str(dyn_sys)+'_JAC.jpg'
    # plot_vector_field(JAC, path=JAC_plot_path, idx=1, t=0., N=100, device='cuda')
    plot_3d_vector_field(JAC, '../plot/Vector_field/'+str(dyn_sys)+'_3d_JAC.jpg', 0, 10, device='cuda')

