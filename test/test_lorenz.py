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


########################
### Dynamical System ###
########################

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

# Time Integrator
def solve_odefunc(odefunc, t, y0):

    solution = torchdiffeq.odeint(odefunc, y0, t, rtol=1e-9, atol=1e-9, method="rk4")
    final_state = solution[-1]
    return final_state

##############
## Training ##
##############

def create_data(traj, n_train, n_test, n_nodes, n_trans):

    ##### create training dataset #####
    X = np.zeros((n_train, n_nodes))
    Y = np.zeros((n_train, n_nodes))

    if torch.is_tensor(traj):
        traj = traj.detach().cpu().numpy()
    for i in torch.arange(0, n_train, 1):
        i = int(i)
        X[i] = traj[n_trans+i]
        Y[i] = traj[n_trans+1+i]

    X = torch.tensor(X).reshape(n_train,n_nodes)
    Y = torch.tensor(Y).reshape(n_train,n_nodes)

    ##### create test dataset #####
    X_test = np.zeros((n_test, n_nodes))
    Y_test = np.zeros((n_test, n_nodes))

    for i in torch.arange(0, n_test, 1):
        i = int(i)
        X_test[i] = traj[n_trans+n_train+i]
        Y_test[i] = traj[n_trans+1+n_train+i]

    X_test = torch.tensor(X_test).reshape(n_test, n_nodes)
    Y_test = torch.tensor(Y_test).reshape(n_test, n_nodes)

    return [X, Y, X_test, Y_test]

def reg_jacobian_loss(time_step, True_J, cur_model_J, output_loss, reg_param):
    #reg_param: 1e-5 #5e-4 was working well #0.11

    diff_jac = True_J - cur_model_J
    norm_diff_jac = torch.norm(diff_jac)

    total_loss = reg_param * norm_diff_jac**2 + output_loss

    return total_loss

def Jacobian_Matrix(input, sigma, r, b):
    ''' Jacobian Matrix of Lorenz '''

    x, y, z = input
    return torch.stack([torch.tensor([-sigma, sigma, 0]), torch.tensor([r - z, -1, -x]), torch.tensor([y, x, -b])])

def train(dyn_sys_info, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, rho, reg_param, loss_type, new_loss=True, multi_step = False,minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    num_train = X.shape[0]
    dyn_sys, dyn_sys_name, dim = dyn_sys_info
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
    
    # Compute True Jacobian
    if loss_type == "Jacobian":
        True_J = torch.ones(num_train, dim, dim).to(device)
        if dyn_sys_name == "lorenz":
            for i in range(num_train):
                True_J[i] = Jacobian_Matrix(X[i, :], sigma=10.0, r=rho, b=8/3)
        print("Finished Computing True Jacobian")

    # Training Loop
    for i in range(epochs):
        model.train()
        model.double()

        y_pred = solve_odefunc(model, t_eval_point, X).to(device)
        optimizer.zero_grad()
        MSE_loss = criterion(y_pred, Y)  * (1/time_step)

        if loss_type == "Jacobian":
            # Compute Jacobian
            jacrev = torch.func.jacrev(model, argnums=1)
            compute_batch_jac = torch.vmap(jacrev, in_dims=(None, 0), chunk_size=1000)
            cur_model_J = compute_batch_jac(t_eval_point, X).to(device)
            # Compute Jacobian Matching Loss
            train_loss = reg_jacobian_loss(time_step, True_J, cur_model_J, MSE_loss, reg_param)
        else:
            # Compute MSE Loss
            train_loss = MSE_loss.item()

        # Plot Vector Field
        if (i % 10 == 0) and (i < 500):
            idx=1
            JAC_path = '../plot/Vector_field/train/'+str(dyn_sys_name)+'_JAC'+'_i='+str(i)+'_'+str(idx)+'.jpg'
            plot_vector_field(model, dyn_sys_name, idx=idx, traj=None, path=JAC_path, t=0., N=50, device='cuda')
            # print(i, cur_model_J)

        train_loss.backward()
        optimizer.step()

        # Save Training History
        loss_hist.append(train_loss)
        if i % 1000 == 0:
            print(i, MSE_loss.item(), train_loss.item())

        ##### test one_step #####
        pred_test, test_loss = evaluate(dyn_sys_name, model, time_step, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)

    return loss_hist, test_loss_hist


def evaluate(dyn_sys, model, time_step, X_test, Y_test, device, criterion, iter, optimizer_name):

  with torch.no_grad():
    model.eval()

    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
    y_pred_test = solve_odefunc(model, t_eval_point, X_test).to(device)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y_test = Y_test.detach().cpu()

    test_loss = criterion(pred_test, Y_test).item()
    if iter % 1000 == 0:
        print("test loss:", test_loss)

  return pred_test, test_loss


##############
#### Plot ####
##############

def plot_vector_field(model, dyn_sys, idx, traj, path, t=0., N=50, device='cuda'):
    # credit: https://torchdyn.readthedocs.io/en/latest/_modules/torchdyn/utils.html

    if dyn_sys == "lorenz": 
        if idx == 1:
            x = torch.linspace(-100, 100, N)
            y = torch.linspace(-100, 100, N)
        else:
            x = torch.linspace(-20, 20, N)
            y = torch.linspace(0, 40, N)
        
    X, Y = torch.meshgrid(x,y)
    U, V = torch.zeros(N,N), torch.zeros(N,N)
    print("shape", X.shape, Y.shape)

    for i in range(N):
        for j in range(N):
            if idx == 1:
                phi = torch.stack([torch.tensor(X[i,j]), torch.tensor(Y[i,j]), torch.tensor(13.5)]).to('cuda').double()
            else:
                phi = torch.stack([X[i,j].clone().detach(), torch.tensor(0), Y[i,j].clone().detach()]).to('cuda').double()

            O = model(0., phi)
            U[i,j], V[i,j] = O[0], O[idx]

    print(X.requires_grad, Y.requires_grad, U.requires_grad, V.requires_grad)

    fig = figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    if U.requires_grad:
        contourf = ax.contourf(X, Y, torch.sqrt(U**2 + V**2).detach().numpy(), cmap='jet') #'RdYlBu'
        ax.streamplot(X.T.numpy(),Y.T.numpy(),U.T.detach().numpy(),V.T.detach().numpy(), color='k')
    else:
        contourf = ax.contourf(X, Y, torch.sqrt(U**2 + V**2), cmap='jet') #'RdYlBu'
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
    tight_layout()
    savefig(path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)

    return


if __name__ == '__main__':

    # Set device
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--integration_time", type=int, default=200)
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=8000)
    parser.add_argument("--num_trans", type=int, default=0)
    parser.add_argument("--iters", type=int, default=5*(10**4))
    parser.add_argument("--minibatch", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--new_loss", type=bool, default=True)
    parser.add_argument("--loss_type", default="Jacobian", choices=["Jacobian", "MSE", "Auto_corr"])
    parser.add_argument("--reg_param", type=float, default=1e-1)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--dyn_sys", default="lorenz")

    # Save args
    args = parser.parse_args()
    timestamp = datetime.datetime.now()
    path = '../test_result/expt_'+str(args.dyn_sys)+'/'+str(timestamp)+'.txt'
    # Convert the argparse.Namespace object to a dictionary
    args_dict = vars(args)
    with open(path, 'w') as f:
        json.dump(args_dict, f, indent=2)

    # 1. initialize
    dim = 3
    x0 = torch.randn(dim)
    x_multi_0 = torch.randn(dim)
    dyn_sys_info = [lorenz, args.dyn_sys, dim]

    # Initialize Model and Dataset Parameters
    criterion = torch.nn.MSELoss()
    real_time = args.iters * args.time_step

    # Create Dataset
    t_eval_point = torch.arange(0,args.integration_time+1,args.time_step)
    whole_traj = torchdiffeq.odeint(lorenz, x0, t_eval_point, method='rk4', rtol=1e-8) 
    training_traj = whole_traj[:args.integration_time*int(1/args.time_step), :]
    print("Finished Simulating")
    dataset = create_data(training_traj, n_train=args.num_train, n_test=args.num_test, n_nodes=dim, n_trans=args.num_trans)

    # Create model
    m = ODE_Lorenz(y_dim=dim, n_hidden=dim).to(device).double()

    # Train the model, return node
    loss_hist, test_loss_hist= train(dyn_sys_info, m, device, dataset, None, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, 28.0, args.reg_param, args.loss_type, new_loss= args.new_loss, multi_step=False, minibatch=args.minibatch, batch_size=args.batch_size)

    # Maximum weights
    print("Saving Results...")
    max_weight = []
    for param_tensor in m.state_dict():
        if "weight" in param_tensor:
            weights = m.state_dict()[param_tensor].squeeze()
            max_weight.append(torch.max(weights).cpu().tolist())

    lh = loss_hist[-1].tolist()
    tl = test_loss_hist[-1]

    with open(path, 'a') as f:
        entry = {'train loss': lh, 'test loss': tl}
        json.dump(entry, f, indent=2)

    # Save Trained Model
    model_path = "../test_result/expt_"+str(args.dyn_sys)+"/"+args.optim_name+"/"+str(args.time_step)+'/'+'model.pt'
    torch.save(m.state_dict(), model_path)
    print("Saved new model!")

    # Save Training/Test Loss
    loss_hist = torch.stack(loss_hist)
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"training_loss.csv", np.asarray(loss_hist.detach().cpu()), delimiter=",")
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"test_loss.csv", np.asarray(test_loss_hist), delimiter=",")

    # Save Vector Field Plot
    JAC_plot_path = '../plot/Vector_field/'+str(dyn_sys)+'_JAC.jpg'
    plot_vector_field(JAC, path=JAC_plot_path, idx=1, t=0., N=100, device='cuda')

