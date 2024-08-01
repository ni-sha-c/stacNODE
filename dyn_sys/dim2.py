import torch
import math
import argparse
import datetime
import sys
import json

# sys.path.append('../test')
# from test_metrics import *

# csc.ucdavis.edu/~chaos/courses/nlp/Software/PartG_Code/BakersMap.py


# def baker(X):

#     a=0.3
#     x, y = X

#     # Assume (x,y) is on [0,1] x [0,1]
#     y = a* y
#     if x > 0.5:
#         y = y + 0.5
    
#     x = 2.0 * x
#     while x > 1.0:
#         x = x- 1.0 

    
#     return torch.stack([x, y])

def baker(X, s3=0.):

    '''From "Efficient Computation of Linear Response of Chaotic Attractors with
One-Dimensional Unstable Manifolds" by Chandramoorthy et al. 2022 '''

    x, y = X #[0., 0., 0.5, 0.]
    s1 = 0
    s2 = 0.
    # s3 = 0.6
    s4 = 0.
    x = 2*x + (s1 + s2*torch.sin((2*y)/2))*torch.sin(x) - torch.floor(x/torch.pi)*2*torch.pi
    y = (y + (s4 + s3*torch.sin(x))*torch.sin(2*y) + torch.floor(x/torch.pi)*2*torch.pi)/2

    x = x % (2*torch.pi)
    y = y % (2*torch.pi)
    
    return torch.stack([x, y])


if __name__ == '__main__':
    s3 = 0.8

    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)


    # Set arguments (hyperparameters)
    DYNSYS_MAP = {'baker' : [baker, 2]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=6000) # 10000
    parser.add_argument("--integration_time", type=int, default=19000) #100
    parser.add_argument("--num_train", type=int, default=10000) #3000
    parser.add_argument("--num_test", type=int, default=8000)#3000
    parser.add_argument("--num_trans", type=int, default=0) #10000
    parser.add_argument("--iters", type=int, default=5*(10**4))
    parser.add_argument("--minibatch", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--loss_type", default="Jacobian", choices=["Jacobian", "MSE", "Auto_corr"])
    parser.add_argument("--reg_param", type=float, default=1e-6)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--dyn_sys", default="baker", choices=DYNSYS_MAP.keys())


    args = parser.parse_args()
    dyn_sys_func, dim = define_dyn_sys(args.dyn_sys)
    dyn_sys_info = [dyn_sys_func, args.dyn_sys, dim]

    if args.dyn_sys == "lorenz":
        rho = 28.0
    elif args.dyn_sys == "lorenz_periodic":
        rho = 350.
    else:
        rho = 0.8
    print("args: ", args)
    print("dyn_sys_func: ", dyn_sys_func)

    # Save args
    timestamp = datetime.datetime.now()
    # Convert the argparse.Namespace object to a dictionary
    path = '../test_result/expt_'+str(args.dyn_sys)+'/'+str(s3)+'_'+str(timestamp)+'.txt'
    args_dict = vars(args)
    with open(path, 'w') as f:
        json.dump(args_dict, f, indent=2)

    

    x0 = torch.randn(dim)
    x_multi_0 = torch.randn(dim)
    
    print("initial point:", x0, x_multi_0)

    # Initialize Model and Dataset Parameters
    criterion = torch.nn.MSELoss()
    real_time = args.iters * args.time_step
    print("real time: ", real_time)

    # Generate Training/Test/Multi-Step Prediction Data
    if (str(args.dyn_sys) == "henon") or (str(args.dyn_sys) == "baker") or (str(args.dyn_sys) == "tent_map"):
        
        whole_traj = torch.zeros(args.integration_time*int(1/args.time_step), dim)
        longer_traj = torch.zeros(args.iters, dim)
        
        for i in range(args.integration_time*int(1/args.time_step)):
            next_x = dyn_sys_func(x0)
            whole_traj[i] = next_x
            x0 = next_x
        training_traj = whole_traj.requires_grad_(True)

        for j in range(args.iters):
            next_x = dyn_sys_func(x_multi_0)
            longer_traj[j] = next_x
            x_multi_0 = next_x

        # elif (str(args.dyn_sys) == "baker"):
        #     fig, (ax1, ax2) = subplots(1,2, figsize=(24,12))
        #     ax1.scatter(whole_traj[:args.num_train, 0], whole_traj[:args.num_train, 1], color=(0.25, 0.25, 0.25), s=20, alpha=0.7)
        #     ax2.scatter(whole_traj[:100, 0], whole_traj[:100, 1], color=(0.25, 0.25, 0.25), s=20, alpha=0.7)
        #     ax1.xaxis.set_tick_params(labelsize=24)
        #     ax1.yaxis.set_tick_params(labelsize=24)
        #     ax2.xaxis.set_tick_params(labelsize=24)
        #     ax2.yaxis.set_tick_params(labelsize=24)

        #     path = '../plot/baker_phase.jpg'
        #     fig.savefig(path, format='jpg', dpi=400)

    
    print("train", training_traj.shape)

    dataset = create_data(training_traj, n_train=args.num_train, n_test=args.num_test, n_nodes=dim, n_trans=args.num_trans)
    
    # Create model
    m = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()



    # Train the model, return node
    if args.loss_type == "Jacobian":
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = jac_train(dyn_sys_info, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, rho, args.reg_param, multi_step=False, minibatch=args.minibatch, batch_size=args.batch_size)

    elif args.loss_type == "Auto_corr":
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = ac_train(args.dyn_sys, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, rho, minibatch=args.minibatch, batch_size=args.batch_size)
        
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
    pred_train = torch.tensor(np.array(pred_train))
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
        json.dump(entry, f)


    # Save Trained Model
    model_path = "../test_result/expt_"+str(args.dyn_sys)+"/"+args.optim_name+"/"+str(args.time_step)+'/'+'model.pt'
    torch.save(m.state_dict(), model_path)
    print("Saved new model!")

    # Save whole trajectory
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"whole_traj.csv", np.asarray(whole_traj.detach().cpu()), delimiter=",")

    # Save Training/Test Loss
    loss_hist = torch.stack(loss_hist)
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"training_loss.csv", np.asarray(loss_hist.detach().cpu()), delimiter=",")
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"test_loss.csv", np.asarray(test_loss_hist), delimiter=",")

    if str(args.dyn_sys) == "tent_map":
        # def ddynamics(x, s):
        #     if x < 1+s:
        #         return 2/(1+s)                                      
        #     return -2/(1-s)
            
        def lyap_exp_1d(x, T, NODE):
            # x = 2*torch.rand(1)
            le = 0
            if NODE == True:
                 for t in range(x.shape[0]):
                    # x = dynamics(x, s)
                    # le += torch.log(torch.tensor(abs(ddynamics(x[t], s))))
                    le += torch.log(abs(F.jacobian(m, x[t].to(device).double())))
            else:
                for t in range(x.shape[0]):
                    # x = dynamics(x, s)
                    le += torch.log(abs(F.jacobian(tent_map, x[t].to(device))))
            return le/T
        
        LE_NODE = lyap_exp_1d(longer_traj, args.iters, True)
        LE_rk4 = lyap_exp_1d(longer_traj, args.iters, False)
        print(LE_NODE, LE_rk4)
    else:
        # Compute Jacobian Matrix and Lyapunov Exponent of Neural ODE
        LE_NODE = lyap_exps(args.dyn_sys, dyn_sys_info, longer_traj, iters=args.iters, time_step= args.time_step, optim_name=args.optim_name, method="NODE", path=model_path)
        print("NODE LE: ", LE_NODE)

        # Compute Jacobian Matrix and Lyapunov Exponent of rk4
        LE_rk4 = lyap_exps(args.dyn_sys, dyn_sys_info, longer_traj, iters=args.iters, time_step= args.time_step, optim_name=args.optim_name, method="rk4", path=model_path)
        print("rk4 LE: ", LE_rk4)

    # Compute || LE_{NODE} - LE_{rk4} ||
    norm_difference = torch.linalg.norm(LE_NODE - LE_rk4)
    print("Norm Difference: ", norm_difference)

    with open(path, 'a') as f:
        entry = {'Nerual ODE LE': LE_NODE.detach().cpu().tolist(), 'rk4 LE': LE_rk4.detach().cpu().tolist(), 'norm difference': norm_difference.detach().cpu().tolist()}
        json.dump(entry, f)