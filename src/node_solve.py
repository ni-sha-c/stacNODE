import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
# import ray
from scipy import stats
import numpy as np
from matplotlib.pyplot import *
import multiprocessing

from .NODE import *
import sys
sys.path.append('..')
from examples.Brusselator import *
from examples.Lorenz_periodic import *
from examples.Lorenz import *
from examples.Sin import *
from examples.KS import *
from examples.Tilted_tent_map import *
from examples.Pinched_tent_map import *
from examples.Plucked_tent_map import *
from examples.Henon import *
from examples.Coupled_Brusselator import *
from examples.Baker import *
from examples.LV import *


def simulate(dyn_system, ti, tf, init_state, time_step):
    ''' func: call derivative function
        param:
              dyn_system = dynamical system of our interest 
              ti, tf = interval of integration
              init_state = initial state, in array format like [1,3]
              time_step = time step size used for time integrator '''

    init = torch.Tensor(init_state)
    t_eval_point = torch.arange(ti,tf,time_step)
    traj = torchdiffeq.odeint(dyn_system, init, t_eval_point, method='rk4', rtol=1e-8) 
    print("Finished Simulating")

    return traj





def simulate_NODE(dyn_system, model, ti, tf, init_state, time_step):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.Tensor(init_state).double()
    t_eval_point = torch.arange(ti,tf,time_step)
    num_step = tf*int(1/time_step)
    traj = torch.zeros(num_step, 3).to(device)

    for i in range(num_step):
        traj[i] = x # shape [3]
        cur_pred = model(x)
        x = cur_pred
    
    print("Finished Simulating")
    return traj


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


def define_dyn_sys(dyn_sys):
    DYNSYS_MAP = {'sin' : [sin, 1],
                  'tilted_tent_map' : [tilted_tent_map, 1],
                  'pinched_tent_map' : [pinched_tent_map, 1],
                  'plucked_tent_map' : [plucked_tent_map, 1],
                  'KS': [run_KS, 127],
                  'henon' : [henon, 2],
                  'baker' : [baker, 2],
                  'lv' : [lv, 2],
                  'brusselator' : [brusselator, 2],
                  'lorenz_periodic' : [lorenz_periodic, 3],
                  'lorenz' : [lorenz, 3],
                  'rossler' : [rossler, 3],
                  'coupled_brusselator': [coupled_brusselator, 4],
                  'hyperchaos': [hyperchaos, 4],
                  'hyperchaos_hu': [hyperchaos_hu, 7]}
    dyn_sys_info = DYNSYS_MAP[dyn_sys]
    dyn_sys_func, dim = dyn_sys_info

    return dyn_sys_func, dim



def create_NODE(device, dyn_sys, n_nodes, n_hidden, T):
    ''' Create Neural ODE based on dynamical system of our interest 
        '''

    DYNSYS_NN_MAP = {'sin' : ODE_Sin,
                  'tilted_tent_map' : ODE_Tent,
                  'pinched_tent_map' : ODE_Tent,
                  'plucked_tent_map' : ODE_Tent,
                  'KS': ODE_KS,
                  'henon': ODE_henon,
                  'baker' : ODE_baker,
                  'lv': ODE_LV,
                  'brusselator' : ODE_Brusselator,
                  'lorenz_periodic' : ODE_Lorenz_periodic,
                  'lorenz' : ODE_Lorenz,
                  'rossler' : ODE_Lorenz,
                  'coupled_brusselator': ODE_Coupled_Brusselator,
                  'hyperchaos': ODE_Coupled_Brusselator,
                  'hyperchaos_hu': ODE_hyperchaos_hu}
    ODEFunc = DYNSYS_NN_MAP[dyn_sys]
    neural_func = ODEFunc(y_dim=n_nodes, n_hidden=n_hidden).to(device)
    return neural_func

def vectorized_simulate(model, X, t_eval_point, len_T, device):
    torch.cuda.empty_cache()
    integrated_model = lambda x: one_step_rk4(model, x, t_eval_point).to(device)
    compute_batch = torch.func.vmap(integrated_model, in_dims=(0), chunk_size=2000)
    
    traj = torch.zeros(len_T, X.shape[0], X.shape[1]) # len_T x num_init x dim
    traj[0] = X
    for i in range(1, len_T):
        traj[i] = compute_batch(X.double().to(device)).detach() 
        X = traj[i]
    return traj

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



def create_iterables(dataset, batch_size):
    X, Y, X_test, Y_test = dataset

    # Dataloader
    train_data = torch.utils.data.TensorDataset(X, Y)
    test_data = torch.utils.data.TensorDataset(X_test, Y_test)

    # Data iterables
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_iter, test_iter

# custom rk4
def one_step_rk4(f, y0, t):

    h = t[1] - t[0]
    k1 = f(t, y0)
    k2 = f(t + h/2, y0 + k1 * h / 2.)
    k3 = f(t + h/2, y0 + k2 * h / 2.)
    k4 = f(t + h, y0 + k3 * h)
    new_y = y0 + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    # print("new_shape", new_y.shape)
    return new_y


def auto_corr(y_hat, device):
    ''' computing auto_corr in batch 
        input = 1D time series 
        output = scalar '''

    window_size = 9000 #90 in real time
    step = 1
    tau = y_hat.shape[0] - window_size #100
    # corr = torch.zeros(tau).to(device)
    list_i = torch.arange(0, tau+1, step) # 0 ... tau
    component = 2 # indicator function

    base_line = y_hat[0:window_size, component]
    subseq = torch.stack([y_hat[i:i+window_size, component] for i in list_i]).to(device)

    def corr_val(subseq):
        return torch.dot(torch.flatten(base_line).to(device), torch.flatten(subseq))
        # return _corr(base_line.to(device), subseq)

    # Use torch.vmap
    batch_corr_val = torch.vmap(corr_val)
    corr = batch_corr_val(subseq).to(device)
    print(corr.shape)

    corr = corr/window_size - torch.mean(y_hat)**2
    tau_x = torch.arange(0, tau+1, step)

    return tau_x, corr


# ------------- #
# Jacobian Loss #
# ------------- #


def jacobian_loss(True_J, cur_model_J, output_loss, reg_param):
    #reg_param: 1e-5 #5e-4 was working well #0.11

    diff_jac = True_J - cur_model_J
    norm_diff_jac = torch.norm(diff_jac)

    total_loss = reg_param * norm_diff_jac**2 + output_loss

    return total_loss


def reg_jacobian_loss(time_step, True_J, cur_model_J, output_loss, reg_param):
    #reg_param: 1e-5 #5e-4 was working well #0.11

    diff_jac = True_J - cur_model_J
    norm_diff_jac = torch.norm(diff_jac)

    total_loss = reg_param * norm_diff_jac**2 + (1/time_step)*output_loss

    return total_loss



def jacobian_parallel(dyn_sys, model, X, t_eval_point, device, node):

    dyn_sys_f, dim = define_dyn_sys(dyn_sys)

    with multiprocessing.Pool(processes=20) as pool:
        results = pool.map(single_jacobian, [(dyn_sys_f, model, x, t_eval_point, device, node) for x in X])

    return results


def single_jacobian(args):
    '''Compute Jacobian of dyn_sys
    Param:  '''
    dyn_sys_f, model, x, t_eval_point, device, node = args

    if node == True:
        jac = torch.squeeze(F.jacobian(model, x))
    else:
        jac = F.jacobian(lambda x: torchdiffeq.odeint(dyn_sys_f, x, t_eval_point, method="rk4"), x, vectorize=True)[1]
    
    return jac


def Jacobian_Matrix(input, sigma, r, b):
    ''' Jacobian Matrix of Lorenz '''

    x, y, z = input
    return torch.stack([torch.tensor([-sigma, sigma, 0]), torch.tensor([r - z, -1, -x]), torch.tensor([y, x, -b])])



def Jacobian_Brusselator(dyn_sys_f, x):
    ''' Jacobian Matrix of Coupled Brusselator '''
    a = 2.0
    b = 6.375300171526684
    lambda_1 = 1.2
    lambda_2 = 80.0
    x1, y1, x2, y2 = x

    matrix = torch.stack([
    torch.tensor([-(b+1) + 2*x1*y1 -lambda_1, x1**2, lambda_1, 0]), 
    torch.tensor([b-2*x1*y1, - x1**2 - lambda_2, 0, lambda_2]), 
    torch.tensor([lambda_1, 0, -(b+1) + 2*x2*y2 -lambda_1, x2**2]), 
    torch.tensor([0, lambda_2, b - 2*x2*y2, -x2**2 - lambda_2])])

    return matrix

def Jacobian_Henon(X):
    x, y = X
    a=1.4
    b=0.3

    return torch.stack([torch.tensor([- 2*a*x, 1]), torch.tensor([b, 0])])

def Jacobian_Hyperchaos(X):
    a= 35
    b= 10
    c= 1
    d= 17

    x1, x2, x3, x4 = X

    # return torch.stack([torch.tensor([-a, a, 0, 0]), torch.tensor([b, b, 0, 0]), torch.tensor([0, 0, -c, c]), torch.tensor([0, 0, 0, -d])])
    return torch.stack([torch.tensor([-a, a+x3*x4, x2*x4, x2*x3]), torch.tensor([b-x3*x4, b, -x1*x4, -x1*x3]), torch.tensor([x2*x4, x1*x4, -c, x1*x2]), torch.tensor([x2*x3, x1*x3, x1*x2, -d])])


# ------------- #
# Training Loop #
# ------------- #


def ac_train(dyn_sys, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, rho, minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    optimizer = define_optimizer(optim_name, model, lr, weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    num_train = X.shape[0]
    
    # Compute True Autocorr
    print("Auto_corr")
    reg_param = 1e-5
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
    tau_x, true_list = auto_corr(Y, device)


    for i in range(epochs): # looping over epochs
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

            y_pred = solve_odefunc(model, t_eval_point, X).to(device)

            optimizer.zero_grad()
            # MSE Output Loss
            MSE_loss = criterion(y_pred, Y)

            # Jacobian Diff Loss
            tau_x, pred_list = auto_corr(y_pred, device)
            train_loss = reg_param * torch.norm(pred_list - true_list)**2 + MSE_loss
            train_loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(Y.detach().cpu().numpy())
        
        loss_hist.append(train_loss)
        print(i, MSE_loss.item(), train_loss.item())

        ##### test one_step #####
        pred_test, test_loss = evaluate(dyn_sys, model, time_step, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)

        ##### test multi_step #####
        if (i+1) == epochs:
           error = test_multistep(dyn_sys, model, epochs, true_t, device, i, optim_name, lr, time_step, real_time, tran_state)

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist, error



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

                if i % 1000 == 0:
                    print(i, cur_model_J)

            if new_loss == True:
                train_loss = reg_jacobian_loss(time_step, True_J, cur_model_J, MSE_loss, reg_param)
            else:
                train_loss = jacobian_loss(True_J, cur_model_J, MSE_loss, reg_param)
            train_loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(Y.detach().cpu().numpy())
        
        loss_hist.append(train_loss)
        if i % 1000 == 0:
            print(i, MSE_loss.item(), train_loss.item())

        ##### test one_step #####
        pred_test, test_loss = evaluate(dyn_sys_name, model, time_step, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)

        ##### test multi_step #####
        if (i+1) == epochs and (multi_step == True):
            error = test_multistep(dyn_sys, dyn_sys_name, model, epochs, true_t, device, i, optim_name, lr, time_step, real_time, tran_state)
        else:
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
        if i % 1000 == 0:
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


def hyperparam_gridsearch(dyn_sys, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, rho, reg_param, minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    optimizer = define_optimizer(optim_name, model, lr, weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    num_train = X.shape[0]
    
    print("data", X[:10])
    
    # Compute True Jacobian
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()

    True_J = torch.ones(num_train, 3, 3).to(device)
    for i in range(num_train):
        True_J[i] = Jacobian_Matrix(X[i, :], sigma=10.0, r=rho, b=8/3)
    print(True_J.shape)
    print("Finished Computing True Jacobian")

    for i in range(epochs): # looping over epochs
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

            y_pred = solve_odefunc(model, t_eval_point, X).to(device)

            optimizer.zero_grad()
            # MSE Output Loss
            MSE_loss = criterion(y_pred, Y)
            jacrev = torch.func.jacrev(model, argnums=1)

            # Jacobian Diff Loss
            compute_batch_jac = torch.vmap(jacrev, in_dims=(None, 0))
            cur_model_J = compute_batch_jac(t_eval_point, X).to(device)
            train_loss = jacobian_loss(True_J, cur_model_J, MSE_loss, reg_param)
            train_loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(Y.detach().cpu().numpy())
        
        loss_hist.append(train_loss)
        if i % 1000 == 0:
            print(i, MSE_loss.item(), train_loss.item())

        ##### test one_step #####
        pred_test, test_loss = evaluate(dyn_sys, model, time_step, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)

        local_loss_hist = torch.stack(loss_hist)
        np.savetxt('/storage/home/hcoda1/6/jpark3141/p-nisha3-0/GDEExpts/test_result/expt_'+str(dyn_sys)+'/'+ optim_name + '/' + str(time_step) + '/' +"training_loss_"+str(reg_param)+"_"+str(tran_state)+".csv", np.asarray(local_loss_hist.detach().cpu()), delimiter=",")
        np.savetxt('/storage/home/hcoda1/6/jpark3141/p-nisha3-0/GDEExpts/test_result/expt_'+str(dyn_sys)+'/'+ optim_name + '/' + str(time_step) + '/' +"test_loss_"+str(reg_param)+"_"+str(tran_state)+".csv", np.asarray(test_loss_hist), delimiter=",")


        ##### test multi_step #####
        # if (i+1) == epochs:
        #    error = test_multistep(dyn_sys, model, epochs, true_t, device, i, optim_name, lr, time_step, real_time, tran_state)
        error = 0

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist, error


def hyperparam_gridsearch_MSE(dyn_sys, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, rho, reg_param, minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    optimizer = define_optimizer(optim_name, model, lr, weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    num_train = X.shape[0]
    
    print("data", X[:10])
    
    # Compute True Jacobian
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()

    True_J = torch.ones(num_train, 3, 3).to(device)
    for i in range(num_train):
        True_J[i] = Jacobian_Matrix(X[i, :], sigma=10.0, r=rho, b=8/3)
    print(True_J.shape)
    print("Finished Computing True Jacobian")

    for i in range(epochs): # looping over epochs
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

            y_pred = solve_odefunc(model, t_eval_point, X).to(device)

            optimizer.zero_grad()
            # MSE Output Loss
            MSE_loss = criterion(y_pred, Y)
            MSE_loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(Y.detach().cpu().numpy())
        
        loss_hist.append(MSE_loss)
        if i % 1000 == 0:
            print(i, MSE_loss.item())

        ##### test one_step #####
        pred_test, test_loss = evaluate(dyn_sys, model, time_step, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)

        local_loss_hist = torch.stack(loss_hist)
        np.savetxt('/storage/home/hcoda1/6/jpark3141/p-nisha3-0/GDEExpts/test_result/expt_'+str(dyn_sys)+'/'+ optim_name + '/' + str(time_step) + '/' +"training_loss_MSE_"+str(reg_param)+"_"+str(tran_state)+".csv", np.asarray(local_loss_hist.detach().cpu()), delimiter=",")
        np.savetxt('/storage/home/hcoda1/6/jpark3141/p-nisha3-0/GDEExpts/test_result/expt_'+str(dyn_sys)+'/'+ optim_name + '/' + str(time_step) + '/' +"test_loss_MSE_"+str(reg_param)+"_"+str(tran_state)+".csv", np.asarray(test_loss_hist), delimiter=",")
        np.savetxt('/storage/home/hcoda1/6/jpark3141/p-nisha3-0/GDEExpts/test_result/expt_'+str(dyn_sys)+'/'+ optim_name + '/' + str(time_step) + '/' +"dataset_"+str(reg_param)+"_"+str(tran_state)+".csv", np.asarray(X), delimiter=",")


        ##### test multi_step #####
        # if (i+1) == epochs:
        #    error = test_multistep(dyn_sys, model, epochs, true_t, device, i, optim_name, lr, time_step, real_time, tran_state)
        error = 0

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



def test_multistep(dyn_sys, dyn_sys_name, model, epochs, true_traj, device, iter, optimizer_name, lr, time_step, integration_time, tran_state):

  print("true_traj", true_traj.shape)
  print("integration_time", integration_time)

  # num_of_extrapolation_dataset
  t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
  num_data, dim = true_traj.shape
  test_t = torch.linspace(0, integration_time, num_data)
  pred_traj = torch.zeros(num_data, dim).to(device)

  with torch.no_grad():
    model.eval()
    model.double()
    model.to(device)

    # initialize X
    print(true_traj[0])
    X = true_traj[0].to(device)

    # calculating outputs
    for i in range(num_data):
        pred_traj[i] = X # shape [3]
        # cur_pred = model(t.to(device), X.double())
        cur_pred = solve_odefunc(model, t_eval_point, X.double()).to(device)
        X = cur_pred

    # save predicted trajectory
    pred_traj_csv = np.asarray(pred_traj.detach().cpu())
    true_traj_csv = np.asarray(true_traj.detach().cpu())

    # plot traj
    # plot_multi_step_traj_3D(dyn_sys_name, time_step, optimizer_name, test_t, pred_traj, true_traj)

    # Plot Error ||pred - true||
    error = multi_step_pred_error_plot(dyn_sys, dyn_sys_name, device, epochs, pred_traj, true_traj, optimizer_name, lr, time_step, integration_time, tran_state)

  return error



def plot_multi_step_traj_3D(dyn_sys, time_step, optim_n, test_t, pred_traj, true_traj):
    #plot the x, y, z

    fig, axs = subplots(2, figsize=(18, 9), sharex=True)
    fig.suptitle("Multi-Step Predicted Trajectory of Lorenz", fontsize=24)
    axs[0].plot(test_t, pred_traj[:, 0].detach().cpu(), c='C0', ls='--', label='Prediction of x', alpha=0.7)
    axs[0].plot(test_t, pred_traj[:, 1].detach().cpu(), c='C1', ls='--', label='Prediction of y', alpha=0.7)
    axs[0].plot(test_t, pred_traj[:, 2].detach().cpu(), c='C2', ls='--', label='Prediction of z', alpha=0.7)
    axs[0].grid(True)
    axs[0].legend(loc='best', fontsize=20)
    axs[0].set_ylabel(r'$\Phi_{NODE}(t)$', fontsize=24)
    axs[0].tick_params(labelsize=24)

    axs[1].plot(test_t, true_traj[:, 0].detach().cpu(), c='C3', marker=',', label='Ground Truth of x', alpha=0.7)
    axs[1].plot(test_t, true_traj[:, 1].detach().cpu(), c='C4', marker=',', label='Ground Truth of y', alpha=0.7)
    axs[1].plot(test_t, true_traj[:, 2].detach().cpu(), c='C5', marker=',', label='Ground Truth of z', alpha=0.7)
    axs[1].grid(True)
    axs[1].legend(loc='best', fontsize=20)
    axs[1].tick_params(labelsize=24)
    axs[1].set_ylabel(r'$\Phi_{rk4}(t)$', fontsize=24)

    xlabel('t', fontsize=24)
    tight_layout()
    savefig('../test_result/expt_'+str(dyn_sys)+"/"+str(optim_n)+"/"+str(time_step)+"/"+'multi_step_pred.svg', format='svg', dpi=600, bbox_inches ='tight', pad_inches = 0.1)

    return



def multi_step_pred_error_plot(dyn_sys, dyn_sys_name, device, num_epoch, pred_traj, Y, optimizer_name, lr, time_step, integration_time, tran_state):
    ''' func: plot error vs real time
        args:   pred_traj = pred_traj by Neural ODE (csv file)
                Y = true_traj (csv file) '''

    one_iter = int(1/time_step)
    # test_x = torch.arange(0, integration_time, time_step)[tran_state:]
    test_x = torch.arange(0, integration_time, time_step)
    pred = pred_traj.detach().cpu()
    Y = Y.cpu()

    # calculate error
    error_x = np.abs(pred[:, 0] - Y[:, 0]) # np.linalg.norm
    slope = [np.exp(0.9*x)+error_x[15] for x in test_x[:500]]
    slope = np.array(slope)
    
    fig, ax = subplots(figsize=(24, 12))
    ax.semilogy(test_x[2:], error_x[2:], linewidth=2, alpha=0.9, color="b")
    ax.semilogy(test_x[2:500], slope[2:], linewidth=2, ls="--", color="gray", alpha=0.9)
    ax.grid(True)
    ax.set_xlabel(r"$n \times \delta t$", fontsize=24)
    ax.set_ylabel(r"$log |\Phi_{rk4}(t) - \Phi_{NODE}(t)|$", fontsize=24)
    ax.legend(['x component', 'approx slope'], fontsize=20)
    ax.tick_params(labelsize=24)
    tight_layout()
    fig.savefig('../test_result/expt_'+str(dyn_sys_name)+"/"+str(optimizer_name)+"/"+str(time_step)+"/"+'error_plot_' + str(time_step) +'.svg', format='svg', dpi=800, bbox_inches ='tight', pad_inches = 0.1)

    print("multi step pred error: ", error_x[-1])

    return error_x[-1]
