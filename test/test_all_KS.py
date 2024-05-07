import argparse
from test_metrics import *
import datetime
import json
import numpy as np
import argparse
import logging
import time
import os
import csv
from matplotlib.pyplot import *

import sys
sys.path.append('..')

from dyn_sys.dim1 import *
from dyn_sys.dim2 import *
from dyn_sys.dim3 import *
from dyn_sys.dim4 import *
from dyn_sys.KS import *


if __name__ == '__main__':

    # python test_all_KS.py --dyn_sys=KS --loss_type=MSE --time_step=0.1 --iters=0

    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    # Set arguments (hyperparameters)
    DYNSYS_MAP = {'sin' : [sin, 1],
                  'tent_map' : [tent_map, 1],
                  'KS': [run_KS, 127],
                  'henon': [henon, 2],
                  'brusselator' : [brusselator, 2],
                  'lorenz_fixed' : [lorenz_fixed, 3],
                  'lorenz_periodic' : [lorenz_periodic, 3],
                  'lorenz' : [lorenz, 3]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=3000) # 10000
    parser.add_argument("--integration_time", type=int, default=0) #100
    parser.add_argument("--num_train", type=int, default=3000) #3000
    parser.add_argument("--num_test", type=int, default=2000)#3000
    parser.add_argument("--num_trans", type=int, default=0) #10000
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--minibatch", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--loss_type", default="Jacobian", choices=["Jacobian", "MSE", "Auto_corr"])
    parser.add_argument("--reg_param", type=float, default=1e-6)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--dyn_sys", default="KS", choices=DYNSYS_MAP.keys())

    args = parser.parse_args()
    dyn_sys_func, dim = define_dyn_sys(args.dyn_sys)
    dyn_sys_info = [dyn_sys_func, args.dyn_sys, dim]
    
    print("args: ", args)
    print("dyn_sys_func: ", dyn_sys_func)

    # Save args
    timestamp = datetime.datetime.now()
    # Convert the argparse.Namespace object to a dictionary
    args_dict = vars(args)
    path = '../test_result/expt_'+str(args.dyn_sys)+'/'+str(timestamp)+'.txt'
    with open(path, 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Assign Initial Point of Orbit
    L = 128 #128 # n = [128, 256, 512, 700]
    n = L-1 # num of internal node
    T = 1001 #1000 #100
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
    m = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()
    longer_traj = None
    torch.cuda.empty_cache()

    # Train the model, return node
    if args.loss_type == "Jacobian":
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = jac_train(dyn_sys_info, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, 0, args.reg_param, multi_step=False, minibatch=args.minibatch, batch_size=args.batch_size)      
    else:
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = MSE_train(dyn_sys_info, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, multi_step=False, minibatch=args.minibatch, batch_size=args.batch_size)

    


    # Maximum weights
    print("Saving Results...")
    max_weight = []
    for param_tensor in m.state_dict():
        if "weight" in param_tensor:
            weights = m.state_dict()[param_tensor].squeeze()
            max_weight.append(torch.max(weights).cpu().tolist())

    # Maximum solution
    pred_train = torch.tensor(np.array(pred_train)).squeeze()
    true_train = torch.tensor(np.array(Y)).squeeze()
    pred_test = torch.tensor(np.array(pred_test)).squeeze()
    true_test = torch.tensor(np.array(Y_test)).squeeze()

    max_solution = [torch.max(pred_train[:, 0]).cpu().tolist(), torch.max(pred_train[:, 1]).cpu().tolist(), torch.max(pred_train[:, 2]).cpu().tolist()]

    # Dump Results
    if torch.isnan(multi_step_error):
        multi_step_error = torch.tensor(0.)

    lh = loss_hist[-1].tolist()
    tl = test_loss_hist[-1]
    ms = max_solution
    mw = max_weight
    mse = multi_step_error.cpu().tolist()

    with open(path, 'a') as f:
        entry = {'train loss': lh, 'test loss': tl, 'max of solution': ms, 'max of weight': mw, 'multi step prediction error': mse}
        json.dump(entry, f, indent=2)


    # Save Trained Model
    model_path = "../test_result/expt_"+str(args.dyn_sys)+"/"+args.optim_name+"/"+str(args.time_step)+'/'+'model.pt'
    torch.save(m.state_dict(), model_path)
    print("Saved new model!")

    # plot trained
    print("time", args.num_train*dt)
    print(Y.shape, true_train.shape, pred_train.shape, pred_test.shape)
    plot_KS(true_test, dx, n, c, true_test.shape[0]*dt, dt, True, False, args.loss_type)
    plot_KS(pred_test, dx, n, c, true_test.shape[0]*dt, dt, False, True, args.loss_type)

    # Save Training/Test Loss
    loss_hist = torch.stack(loss_hist)
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"training_loss.csv", np.asarray(loss_hist.detach().cpu()), delimiter=",")
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"test_loss.csv", np.asarray(test_loss_hist), delimiter=",")

    torch.cuda.empty_cache()
    LE_NODE = lyap_exps_ks(args.dyn_sys, dyn_sys_info, u_list, args.iters, u_list, dx, L, c, T, dt, time_step= args.time_step, optim_name=args.optim_name, method="NODE", path=model_path)
    print("NODE LE: ", LE_NODE)

    
    # Compute Jacobian Matrix and Lyapunov Exponent of rk4
    LE_rk4 = lyap_exps_ks(args.dyn_sys, dyn_sys_info, u_list, args.iters, u_list, dx, L, c, T, dt, time_step= args.time_step, optim_name=args.optim_name, method="rk4", path=model_path)
    print("rk4 LE: ", LE_rk4)

    # Compute || LE_{NODE} - LE_{rk4} ||
    norm_difference = torch.linalg.norm(torch.tensor(LE_NODE[:16]) - torch.tensor(LE_rk4[:16]))
    print("Norm Difference: ", norm_difference)

    with open(path, 'a') as f:
        entry = {'Nerual ODE LE': LE_NODE.detach().cpu().tolist(), 'rk4 LE': LE_rk4.detach().cpu().tolist(), 'norm difference': norm_difference.detach().cpu().tolist()}
        json.dump(entry, f, indent=2)

    # NODE LE:  [ 0.14183909  0.10256176  0.0642665   0.02245339 -0.00764637 -0.03447348
    # -0.06807815 -0.10917487 -0.17748009 -0.35903192]        
    # rk4 LE:  [ 0.12833937  0.08656747  0.05771196  0.00716725 -0.03819077 -0.07217354
    # -0.10693418 -0.12887281 -0.20769675 -0.37632966]        
    # Norm Difference:  tensor(0.0786)