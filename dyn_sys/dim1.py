import torch
from matplotlib.pyplot import *

# # Tilted Tentmap, 0.5
def tilted_tent_map(x, s):
    s = torch.tensor(s, dtype=torch.float64).to('cuda')
    # x = x.to(torch.float64).to('cuda')
    # x = torch.tensor(x.clone().detach()).cuda()
    if isinstance(x, torch.Tensor):
        x = x.cuda()
    else:
        x = torch.tensor(x).cuda()

    if s == 0.:
        noise = 1e-12*torch.randn(1).to('cuda')
        if x < 1+s:
            return 2/(1+s)*x + noise
        else:
            return 2/(1-s)*(2-x) + noise
    else:
        if x < 1+s:
            print("1:", x, s, 2/(1+s)*x)
            return (2/(1+s))*x
        else:
            print("2:", x, s, 2/(1-s)*(2-x))
            return (2/(1-s))*(2-x)

# Pinched Tent Map
def pinched_tent_map(x, s):
    s = torch.tensor(s, dtype=torch.float64)
    if isinstance(x, torch.Tensor):
        x = x.cuda()
    else:
        x = torch.tensor(x).cuda() 

    if s == 0.:
        noise = 1e-12*torch.randn(1)

    if x < 1:
        return (4*x)/(1 + s + torch.sqrt((1+s)**2 - 4* s* x))
    elif 1 <= x <= 2:
        return (4*(2-x))/(1 + s + torch.sqrt((1+s)**2 - 4* s*(2-x)))

#Plucked Tentmap # 3, 0.6
def plucked_tent_map(x, s):
    n = torch.tensor(3)
    s = torch.tensor(s)
    # x = x.to(torch.float64) 
    
    noise = 1e-12*torch.randn(1)
    
    def f(x, s):
        if x < 1:
            return min(2*x/(1-s), 2 - 2*(1-x)/(1+s))
    
    def o(x):
        if x < 0.5:
            return f(2*x, s)/2
        return 2 - f(2-2*x, s)/2

    def l(x, n):
        return o((2**n)*x - torch.floor((2**n)*x))/2**n + 2*torch.floor((2**n)*x)/2**n
    
    if 0 < x < 2:
        return torch.min(l(x, n=3), l(2-x, n=3))
    elif x == 0.:
        return x
    elif x == 2.:
        return x


if __name__ == '__main__':

    # # savefig
    fig, ax = subplots(figsize=(24,13))
    pdf_path = '../plot/tilted_map_2'+'.jpg'
    T = 5000
    colors = cm.viridis(np.linspace(0, 1, 5))

    # create True Plot
    s_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(s_list)
    x0 = torch.tensor(0.63)
    # x = torch.linspace(0.01, 0.99, T)

    for idx, s in enumerate(s_list):

        whole_traj = torch.zeros(T)
        x = x0
        for i in range(T):
            next_x = tent_map(x, s)
            whole_traj[i] = next_x
            x = next_x

        ax.scatter(whole_traj[0:-1], whole_traj[1:], color=colors[idx], linewidth=6, alpha=0.8, label='s = ' + str(s))
        
    # ax.grid(True)
    ax.set_xlabel(r"$x_n$", fontsize=44)
    ax.set_ylabel(r"$x_{n+1}$", fontsize=44)
    ax.tick_params(labelsize=40)
    ax.legend(loc='best', fontsize=40)
    tight_layout()
    fig.savefig(pdf_path, format='jpg', dpi=400)





    # # savefig
    # fig, ax = subplots(figsize=(24,13))
    # pdf_path = '../plot/tilted_map_MSE'+'.jpg'
    # T = 50000
    # T_min = 500
    # colors = cm.viridis(np.linspace(0, 1, 6))

    # # create True Plot
    # s_list = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
    # print(s_list)
    # x0 = 0.63
    # x = torch.linspace(0.01, 1.99, T)

    # for idx, s in enumerate(s_list):

    #     whole_traj = torch.zeros(T)
    #     # x = x0
    #     for i in range(T):
    #         next_x = tent_map(x[i], s)
    #         whole_traj[i] = next_x
    #         # x = next_x

    #     # Train model
    #     dataset = create_data(whole_traj, n_train=10000, n_test=8000, n_nodes=1, n_trans=0)
    #     m = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()
    #     # MSE
    #     pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = MSE_train(dyn_sys_info, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, multi_step=False, minibatch=args.minibatch, batch_size=args.batch_size)

    #     # Plot
    #     traj = torch.zeros(T_min)
    #     for i in range(T_min):
    #         next_x = m(x[i])
    #         traj[i] = next_x

    #     ax.scatter(x[0:-1], traj[1:], color=colors[idx], linewidth=6, alpha=0.8, label='s = ' + str(s))
        
    # # ax.grid(True)
    # ax.set_xlabel(r"$x_n$", fontsize=44)
    # ax.set_ylabel(r"$x_{n+1}$", fontsize=44)
    # ax.tick_params(labelsize=40)
    # ax.legend(loc='best', fontsize=40)
    # tight_layout()
                
    # fig.savefig(pdf_path, format='jpg', dpi=400)




