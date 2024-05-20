import torch
import torch.nn as nn
import argparse
import datetime
import sys
import json
import torch.optim as optim
import torchdiffeq
import numpy as np
from matplotlib.pyplot import *
import torch.autograd.functional as F

sys.path.append('..')

from dyn_sys.dim1 import *
from dyn_sys.dim2 import *
from dyn_sys.dim3 import *
from dyn_sys.dim4 import *
from dyn_sys.KS import *


class ODE_KS (nn.Module):
  def __init__( self , y_dim , n_hidden=4) :
    super(ODE_KS , self ).__init__()
    self.net = nn.Sequential(
      nn.Linear(y_dim, 512),
      nn.GELU(),
      nn.Linear(512, 256),
      nn.GELU(),
      nn.Linear(256, y_dim)
    )

  def forward(self, t , y): 
    res = self.net(y)
    return res


def reg_jacobian_loss(time_step, True_J, cur_model_J, output_loss, reg_param):
    #reg_param: 1e-5 #5e-4 was working well #0.11

    diff_jac = True_J - cur_model_J
    norm_diff_jac = torch.norm(diff_jac)

    total_loss = reg_param * norm_diff_jac + (1/time_step/time_step)*output_loss

    return total_loss

def create_data(traj, n_train, n_test, n_nodes, n_trans):
    ''' func: call simulate to create graph and train, test dataset
        args: ti, tf, init_state = param for simulate()
              n_train = num of training instance
              n_test = num of test instance
              n_nodes = num of nodes in graph
              n_trans = num of transition phase '''

    ##### create training dataset #####
    X = np.zeros((n_train, n_nodes))
    Y = np.zeros((n_train, n_nodes))

    if torch.is_tensor(traj):
        traj = traj.detach().cpu().numpy()
    for i in torch.arange(0, n_train, 1):
        i = int(i)
        X[i] = traj[n_trans+i]
        Y[i] = traj[n_trans+1+i]
        # print("X", X[i])

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


def define_optimizer(optim_name, model, lr, weight_decay):

    optim_mapping = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop}

    if optim_name in optim_mapping:
        optim_class = optim_mapping[optim_name]
        optimizer = optim_class(model.parameters(), lr=lr, weight_decay =weight_decay)
    else:
        print(optim_name, " is not in the optim_mapping!")
    
    return optimizer

def one_step_rk4(f, y0, t):

    h = t[1] - t[0]
    k1 = f(t, y0)
    k2 = f(t + h/2, y0 + k1 * h / 2.)
    k3 = f(t + h/2, y0 + k2 * h / 2.)
    k4 = f(t + h, y0 + k3 * h)
    new_y = y0 + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    # print("new_shape", new_y.shape)
    return new_y

def solve_odefunc(odefunc, t, y0):
    ''' Solve odefunction using torchdiffeq.odeint() '''

    solution = torchdiffeq.odeint(odefunc, y0, t, rtol=1e-9, atol=1e-9, method="rk4")
    final_state = solution[-1]
    return final_state

def update_lr(optimizer, epoch, total_e, origin_lr):
    """ A decay factor of 0.1 raised to the power of epoch / total_epochs. Learning rate decreases gradually as the epoch number increases towards the total number of epochs. """
    new_lr = origin_lr * (0.1 ** (epoch / float(total_e)))
    for params in optimizer.param_groups:
        params['lr'] = new_lr
    return

def jac_train(dyn_sys_info, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, rho, reg_param, new_loss=True, multi_step = False,minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    optimizer = define_optimizer(optim_name, model, lr, weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    num_train = X.shape[0]

    # dyn_sys_info = [dyn_sys_func, dim]
    dyn_sys, dyn_sys_name, dim = dyn_sys_info
    
    # Compute True Jacobian
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()

    True_J = torch.ones(num_train, dim, dim).to(device)
    if dyn_sys_name == "lorenz":
        for i in range(num_train):
            True_J[i] = Jacobian_Matrix(X[i, :], sigma=10.0, r=rho, b=8/3)
    elif dyn_sys_name == "coupled_brusselator":
        print("Computing Jacobian of Brusselator!!")
        for i in range(num_train):
            True_J[i] = Jacobian_Brusselator(dyn_sys, X[i, :])
    elif dyn_sys_name == "henon":
        print("henon")
        for i in range(num_train):
            True_J[i] = Jacobian_Henon(X[i, :])
    elif dyn_sys_name == "baker":
        print("baker")
        for i in range(num_train):
            True_J[i] = F.jacobian(baker, X[i, :])
            print(True_J[i])
    elif dyn_sys_name == "rossler":
        print("rossler")
        for i in range(num_train):
            # F.jacobian(lambda x: torchdiffeq.odeint(dyn_sys_f, x, t_eval_point, method="rk4"), x, vectorize=True)[1]
            True_J[i] = F.jacobian(lambda x: rossler(t_eval_point, x), X[i, :], vectorize=True)
    elif dyn_sys_name == "hyperchaos":
        print("hyperchaos")
        for i in range(num_train):
            True_J[i] = F.jacobian(lambda x: hyperchaos(t_eval_point, x), X[i, :], vectorize=True)
            # True_J[i] = Jacobian_Hyperchaos(X[i,:])
            print(True_J[i])

    elif dyn_sys_name == "hyperchaos_hu":
        print("hyperchaos_hu")
        for i in range(num_train):
            True_J[i] = F.jacobian(lambda x: hyperchaos_hu(t_eval_point, x), X[i, :], vectorize=True)
    elif dyn_sys_name == "plucked_tent_map":
        print("plucked_tent_map")
        for i in range(num_train):
            c = X[i,:].requires_grad_(True)
            t = plucked_tent_map(c)
            r = torch.autograd.grad(t, c)
            True_J[i] = r[0] #, create_graph=True
            print(True_J[i])
    elif dyn_sys_name == "tilted_tent_map":
        print("tilted_tent_map")
        for i in range(num_train):
            c = X[i,:].requires_grad_(True)
            t = tilted_tent_map(c)
            r = torch.autograd.grad(t, c)
            True_J[i] = r[0] #, create_graph=True
            print(True_J[i])
    elif dyn_sys_name == "pinched_tent_map":
        print("pinched_tent_map")
        for i in range(num_train):
            c = X[i,:].requires_grad_(True)
            t = pinched_tent_map(c)
            r = torch.autograd.grad(t, c)
            True_J[i] = r[0] #, create_graph=True
            print(True_J[i])
    elif dyn_sys_name == "KS":
        print("KS")
        for i in range(num_train):
            if i % 500 == 0:
                print(i)
            x0 = X[i, :].requires_grad_(True)
            dx = 1 # 0.25
            dt = 0.25
            c = 0.4

            cur_J = F.jacobian(lambda x: run_KS(x, c, dx, dt, dt*2, False, device), x0, vectorize=True)[-1]
            True_J[i] = cur_J

    print(True_J.shape, True_J[0:2])
    print("Finished Computing True Jacobian")

    # Training Loop
    for i in range(epochs):
        model.train()
        model.double()

        if minibatch == True:
            train_iter, test_iter = create_iterables(dataset, batch_size=batch_size)
            y_pred = torch.zeros(len(train_iter), batch_size, 3)
            y_true = torch.zeros(len(train_iter), batch_size, 3)
            k = 0

            for xk, yk in train_iter:
                xk = xk.to(device) # [batch size,3]
                yk = yk.to(device)
                output = model(xk)

                # save predicted node feature for analysis
                y_pred[k] = output
                y_true[k] = yk
                k += 1

            optimizer.zero_grad()
            loss = criterion(y_pred, y_true)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(y_true.detach().cpu().numpy())

        elif minibatch == False:

            if (dyn_sys_name == "henon") or (dyn_sys_name == "baker") or (dyn_sys_name == "tilted_tent_map") or (dyn_sys_name == "pinched_tent_map") or (dyn_sys_name == "plucked_tent_map"):
                y_pred = model(X).to(device)
            else: 
                y_pred = solve_odefunc(model, t_eval_point, X).to(device)

            optimizer.zero_grad()
            # MSE Output Loss
            MSE_loss = criterion(y_pred, Y)

            # Jacobian Diff Loss
            if (dyn_sys_name == "henon") or (dyn_sys_name == "baker") or (dyn_sys_name == "tilted_tent_map") or (dyn_sys_name == "plucked_tent_map") or (dyn_sys_name == "pinched_tent_map"):
                jacrev = torch.func.jacrev(model)
                compute_batch_jac = torch.vmap(jacrev, chunk_size=1000)
                cur_model_J = compute_batch_jac(X).to(device)

            elif (dyn_sys_name == "KS"):

                integrated_model = lambda x: one_step_rk4(model, x, t_eval_point).to(device)
                jacrev = torch.func.jacrev(integrated_model)
                compute_batch_jac = torch.func.vmap(jacrev, in_dims=(0), chunk_size=500)
                # print("X", X.shape)
                cur_model_J = compute_batch_jac(X).to(device)
                # print("J", cur_model_J.shape)

            else:
                jacrev = torch.func.jacrev(model, argnums=1)
                compute_batch_jac = torch.vmap(jacrev, in_dims=(None, 0), chunk_size=1000)
                cur_model_J = compute_batch_jac(t_eval_point, X).to(device)

                if i % 20 == 0:
                    idx=1
                    JAC_path = '../plot/Vector_field/train/'+str(dyn_sys_name)+'_JAC'+'_i='+str(i)+'_'+str(idx)+'.jpg'
                    plot_static_vector_field(model, dyn_sys_name, idx=idx, traj=None, path=JAC_path, t=0., N=50, device='cuda')
                    # print(i, cur_model_J)

            if new_loss == True:
                train_loss = reg_jacobian_loss(time_step, True_J, cur_model_J, MSE_loss, reg_param)
            else:
                train_loss = jacobian_loss(True_J, cur_model_J, MSE_loss, reg_param)
            train_loss.backward()
            optimizer.step()
            update_lr(optimizer, i, epochs, lr)

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(Y.detach().cpu().numpy())
        
        loss_hist.append(train_loss)
        if i % 1000 == 0:
            print(i, MSE_loss.item(), train_loss.item())

        ##### test one_step #####
        pred_test, test_loss = evaluate(dyn_sys_name, model, time_step, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)
        error = torch.tensor(0.)

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist, error


def MSE_train(dyn_sys_info, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, new_loss=True, multi_step = False, minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = [], [], [], []
    optimizer = define_optimizer(optim_name, model, lr, weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()

    # dyn_sys_info = [dyn_sys_func, dim]
    dyn_sys, dyn_sys_name, dim = dyn_sys_info

    for i in range(0, epochs, 1): # looping over epochs
        model.train()
        model.double()

        if minibatch == True:
            train_iter, test_iter = create_iterables(dataset, batch_size=batch_size)
            y_pred = torch.zeros(len(train_iter), batch_size, 3)
            y_true = torch.zeros(len(train_iter), batch_size, 3)
            k = 0

            for xk, yk in train_iter:
                xk = xk.to(device) # [batch size,3]
                yk = yk.to(device)

                output = model(xk)

                # save predicted node feature for analysis
                y_pred[k] = output
                y_true[k] = yk
                k += 1

            optimizer.zero_grad()
            loss = criterion(y_pred, y_true)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(y_true.detach().cpu().numpy())

        elif minibatch == False:

            if (dyn_sys_name == "henon") or (dyn_sys_name == "baker") or (dyn_sys_name == "tilted_tent_map") or (dyn_sys_name == "plucked_tent_map") or (dyn_sys_name == "pinched_tent_map"):
                y_pred = model(X).to(device)
            else: 
                y_pred = solve_odefunc(model, t_eval_point, X).to(device)

            optimizer.zero_grad()
            if new_loss == True:
                loss = criterion(y_pred, Y) * (1/time_step)
            else:
                loss = criterion(y_pred, Y)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(Y.detach().cpu().numpy())
        
        loss_hist.append(torch.tensor(train_loss))
        if i % 20 == 0:
            idx=1
            MSE_path = '../plot/Vector_field/train/'+str(dyn_sys_name)+'_MSE'+'_i='+str(i)+'_'+str(idx)+'.jpg'
            # plot_static_vector_field(model, dyn_sys_name, idx=idx, traj=None, path=MSE_path, t=0., N=50, device='cuda')
            print(i, train_loss)

        ##### test one_step #####
        pred_test, test_loss = evaluate(dyn_sys_name, model, time_step, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)

        ##### test multi_step #####
        if (i+1) == epochs and (multi_step == True):
            error = test_multistep(dyn_sys, dyn_sys_name, model, epochs, true_t, device, i, optim_name, lr, time_step, real_time, tran_state)
        else:
            error = torch.tensor(0)

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist, error

def evaluate(dyn_sys, model, time_step, X_test, Y_test, device, criterion, iter, optimizer_name):


  with torch.no_grad():
    model.eval()

    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
    # y_pred_test = solve_odefunc(model, t_eval_point, X_test).to(device)
    if (dyn_sys == "henon") or (dyn_sys == "baker") or (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        y_pred_test = model(X_test).to(device)
    else: 
        y_pred_test = solve_odefunc(model, t_eval_point, X_test).to(device)


    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y_test = Y_test.detach().cpu()

    test_loss = criterion(pred_test, Y_test).item()
    if iter % 1000 == 0:
        print("test loss:", test_loss)

  return pred_test, test_loss

def lyap_exps_ks(dyn_sys, dyn_sys_info, true_traj, iters, u_list, dx, L, c, T, dt, time_step, optim_name, method, path):
    ''' Compute Lyapunov Exponents 
        args: path = path to model '''

    # Initialize parameter
    dyn_sys_func, dyn_sys_name, org_dim = dyn_sys_info

    # reorthonormalization
    epsilon = 1e-6
    dim = 64
    N = 100
    print("d", dim)

    # QR Method where U = tangent vector, V = regular system
    # CHANGE IT TO DIM X M -> THEN IT WILL COMPUTE M LYAPUNOV EXPONENT.!
    U = torch.eye(*(org_dim, dim)).double()
    print("U", U)
    lyap_exp = [] #empty list to store the lengths of the orthogonal axes
    

    real_time = iters * time_step
    t_eval_point = torch.linspace(0, time_step, 2)
    tran = 0
    print("true traj ks", true_traj.shape)

    if method == "NODE":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t_eval_point = t_eval_point.to(device)

        # load the saved model
        model = ODE_KS(y_dim=org_dim, n_hidden=512).double().to(device)
        model.load_state_dict(torch.load(path), strict=False)
        model.eval()

        for i in range(iters):
            if i % 1000 == 0:
                print(i)

            #update x0
            x0 = true_traj[i].to(device).double()
            # cur_J = model(x0).clone().detach()
            if (dyn_sys_name =="henon") or (dyn_sys_name == "baker"):
                cur_J = F.jacobian(model, x0)
            else:
                cur_J = F.jacobian(lambda x: torchdiffeq.odeint(model, x, t_eval_point, method="rk4"), x0)[1]
            #print(cur_J)
            J = torch.matmul(cur_J.to("cpu"), U.to("cpu").double())

            # QR Decomposition for J
            Q, R = torch.linalg.qr(J)

            lyap_exp.append(torch.log(abs(R.diagonal())))
            U = Q #new axes after iteration

        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]


    else:
        
        for i in range(iters):
            if i % 1000 == 0:
                print("rk4", i) 
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            #update x0
            x0 = true_traj[i].requires_grad_(True)
            
            dx = 1 # 0.25
            dt = 0.25
            c = 0.4

            cur_J = F.jacobian(lambda x: run_KS(x, c, dx, dt, dt*2, False, device), x0, vectorize=True)[-1]

            J = torch.matmul(cur_J.to(device).double(), U.to(device).double())

            # QR Decomposition for J
            Q, R = torch.linalg.qr(J)

            lyap_exp.append(torch.log(abs(R.diagonal())))
            U = Q.double() #new axes after iteration

        lyap_exp = torch.stack(lyap_exp).detach().cpu().numpy()

        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]

    # plot
    # fig, ax = subplots(figsize=(12,6))
    # ax.plot(np.asarray(lyap_exp)[:, 1], color="lime", marker='o', alpha=0.8)
    # ax.xaxis.set_tick_params(labelsize=36)
    # ax.yaxis.set_tick_params(labelsize=36)
    # path = '../plot/'+'LE_conv'+'.png'
    # fig.savefig(path, format='png')

    return torch.tensor(LE)

if __name__ == '__main__':

    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    # Set arguments (hyperparameters)
    DYNSYS_MAP = {'KS': [run_KS, 127]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=3000) # 10000
    parser.add_argument("--integration_time", type=int, default=0) #100
    parser.add_argument("--num_train", type=int, default=3000) #3000
    parser.add_argument("--num_test", type=int, default=3000)#3000
    parser.add_argument("--num_trans", type=int, default=0) #10000
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--minibatch", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--loss_type", default="Jacobian", choices=["Jacobian", "MSE", "Auto_corr"])
    parser.add_argument("--reg_param", type=float, default=1.) #1e-6
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--dyn_sys", default="KS", choices=DYNSYS_MAP.keys())

    args = parser.parse_args()
    dyn_sys_func = run_KS
    dim = 127
    dyn_sys_info = [dyn_sys_func, args.dyn_sys, dim]
    
    print("args: ", args)
    print("dyn_sys_func: ", dyn_sys_func)

    # Save args
    timestamp = datetime.datetime.now()
    # Convert the argparse.Namespace object to a dictionary
    args_dict = vars(args)
    path = '../test_result/'+str(timestamp)+'.txt'
    with open(path, 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Assign Initial Point of Orbit
    L = 128 #128 # n = [128, 256, 512, 700]
    n = L-1 # num of internal node
    T = 1501 #1000 #100
    c = 0.4

    dx = L/(n+1)
    dt = args.time_step
    x = torch.arange(0, L+dx, dx) # [0, 0+dx, ... 128] shape: L + 1
    u0 = 2.71828**(-(x-64)**2/512).to(device).double().requires_grad_(True) # torch.exp(-(x-64)**2/512)
    # u_multi_0 = -0.5 + torch.rand(n+2)

    # Initialize Model and Dataset Parameters
    criterion = torch.nn.MSELoss()
    real_time = args.iters * args.time_step
    print("real time: ", real_time)

    # boundary condition
    u0[0], u0[-1] = 0, 0 
    u0 = u0.requires_grad_(True)

    # Generate Training/Test/Multi-Step Prediction Data
    torch.cuda.empty_cache()
    u_list = run_KS(u0, c, dx, dt, T, False, device)

    u_list = u_list[:, 1:-1] # remove the last boundary node and keep the first boundary node as it is initial condition

    print('u0', u_list[:, 0])
    print("u", u_list.shape)

    # Data split
    dataset = create_data(u_list, n_train=args.num_train, n_test=args.num_test, n_nodes=dim, n_trans=args.num_trans)
    X, Y, X_test, Y_test = dataset
    print(X.shape, Y.shape, X_test.shape, Y_test.shape)

    # Create model
    # m = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()
    m = ODE_KS(y_dim=dim, n_hidden=512).double().to(device)
    longer_traj = None
    torch.cuda.empty_cache()

    # Train the model, return node
    if args.loss_type == "Jacobian":
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = jac_train(dyn_sys_info, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, 0, args.reg_param, multi_step=False, minibatch=args.minibatch, batch_size=args.batch_size)
    else:
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = MSE_train(dyn_sys_info, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, multi_step=False, minibatch=args.minibatch, batch_size=args.batch_size)


    # Dump Results
    # if torch.isnan(multi_step_error):
        # multi_step_error = torch.tensor(0.)

    pred_train = torch.tensor(np.array(pred_train)).squeeze()
    true_train = torch.tensor(np.array(Y)).squeeze()
    pred_test = torch.tensor(np.array(pred_test)).squeeze()
    true_test = torch.tensor(np.array(Y_test)).squeeze()

    lh = loss_hist[-1].tolist()
    tl = test_loss_hist[-1]

    with open(path, 'a') as f:
        entry = {'train loss': lh, 'test loss': tl}
        json.dump(entry, f, indent=2)


    # Save Trained Model
    model_path = "../test_result/"+str(args.dyn_sys)+'model.pt'
    torch.save(m.state_dict(), model_path)
    print("Saved new model!")

    # plot trained
    print("time", args.num_train*dt)
    # print(Y.shape, true_train.shape, pred_train.shape, pred_test.shape)
    plot_KS(true_test, dx, n, c, true_test.shape[0]*dt, dt, True, False, args.loss_type)
    plot_KS(pred_test, dx, n, c, true_test.shape[0]*dt, dt, False, True, args.loss_type)

    torch.cuda.empty_cache()
    LE_NODE = lyap_exps_ks(args.dyn_sys, dyn_sys_info, u_list, args.iters, u_list, dx, L, c, T, dt, time_step= args.time_step, optim_name=args.optim_name, method="NODE", path=model_path)
    print("NODE LE: ", LE_NODE)

    # Compute Jacobian Matrix and Lyapunov Exponent of rk4
    LE_rk4 = lyap_exps_ks(args.dyn_sys, dyn_sys_info, u_list, args.iters, u_list, dx, L, c, T, dt, time_step= args.time_step, optim_name=args.optim_name, method="rk4", path=model_path)
    print("rk4 LE: ", LE_rk4)

    # Compute || LE_{NODE} - LE_{rk4} ||
    norm_difference = torch.linalg.norm(torch.tensor(LE_NODE) - torch.tensor(LE_rk4))
    print("Norm Difference: ", norm_difference/torch.norm(torch.tensor(LE_rk4)))

    with open(path, 'a') as f:
        entry = {'Nerual ODE LE': LE_NODE.detach().cpu().tolist(), 'rk4 LE': LE_rk4.detach().cpu().tolist(), 'norm difference': norm_difference.detach().cpu().tolist()}
        json.dump(entry, f, indent=2)

    # NODE LE:  [ 0.14183909  0.10256176  0.0642665   0.02245339 -0.00764637 -0.03447348
    # -0.06807815 -0.10917487 -0.17748009 -0.35903192]        
    # rk4 LE:  [ 0.12833937  0.08656747  0.05771196  0.00716725 -0.03819077 -0.07217354
    # -0.10693418 -0.12887281 -0.20769675 -0.37632966]        
    # Norm Difference:  tensor(0.0786)