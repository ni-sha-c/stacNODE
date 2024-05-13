import sys
sys.path.append('..')

from src.util import *



if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. define system
    dyn_sys= "lorenz"
    dyn_sys_f, dim = define_dyn_sys(dyn_sys)
    time_step= 0.01
    len_T = 10000*int(1/time_step)
    ind_func = 0
    s = 0.2
    hidden = 256
    model = 'MLP_skip'

    # 2. define num init points
    N = 1
    tran_phase = 0
    if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map") or (dyn_sys == "baker"):
        inits = torch.rand(N, dim).to(device)
    else:
        inits = torch.abs(torch.randn(N, dim).to(device))

    # 3. call models
    if (dyn_sys == "baker"):
        MSE_path = "../test_result/expt_"+str(dyn_sys)+"/" + str(s)+"(MSE)/model.pt"
        JAC_path = "../test_result/expt_"+str(dyn_sys)+"/" + str(s)+"(JAC)/model.pt"
    elif (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        MSE_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+ str(s)+"(MSE)/model.pt"
        JAC_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+ str(s)+"(JAC)/model.pt"
    else:
        MSE_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_MSE_fullbatch/best_model_MLPskip_MSE.pth"
        JAC_path = "../plot/Vector_field/"+str(dyn_sys)+"/"+str(model)+"_Jacobian_fullbatch/best_model_MLPskip_JAC.pth"

    if model == "MLP_skip":
        MSE = ODE_MLP_skip(y_dim=dim, n_hidden=512, n_layers=5).to(device)
        JAC = ODE_MLP_skip(y_dim=dim, n_hidden=256, n_layers=5).to(device)
    else:
        MSE = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=5).to(device)
        JAC = ODE_MLP(y_dim=dim, n_hidden=512, n_layers=5).to(device)

    MSE.load_state_dict(torch.load(MSE_path))
    JAC.load_state_dict(torch.load(JAC_path))
    MSE.eval()
    JAC.eval()

    # 4. generate 3 trajectories
    one_step = torch.linspace(0, time_step, 2).to(device)

    if (dyn_sys == "henon") or (dyn_sys == "baker") or (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        true_traj = torch.zeros(len_T, inits.shape[0], inits.shape[1])

        for j in range(inits.shape[0]):
            print("j: ", j)
            if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
                x = torch.abs(inits[j][0])
            else:
                x = inits[j]
            
            for i in range(len_T):
                next_x = dyn_sys_f(x)
                true_traj[i, j] = next_x
                x = next_x
        MSE_traj = vectorized_simulate_map(MSE, inits, one_step, len_T, device)
        JAC_traj = vectorized_simulate_map(JAC, inits, one_step, len_T, device)
    else:
        true_traj = vectorized_simulate(dyn_sys_f, inits, one_step, len_T, device)
        MSE_traj = vectorized_simulate(MSE, inits, one_step, len_T, device)
        JAC_traj = vectorized_simulate(JAC, inits, one_step, len_T, device)
    print(JAC_traj.shape)

    # 4-1. Remove exploding traj
    MSE_traj = np.asarray(MSE_traj)
    mask = MSE_traj < 10**2  # np.max(np.array(JAC_traj)[:, :, ind_func]) + 5
    if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map") or (dyn_sys == "baker"):
        mask = MSE_traj < 10**1
    row_sums = np.sum(mask, axis=0)

    columns_with_all_true = np.where(row_sums[:, ind_func] == mask.shape[0])
    valid_col = np.unique(columns_with_all_true[0])
    MSE_traj_cleaned = MSE_traj[:, valid_col, :]
    print("MSE cleaned", MSE_traj_cleaned.shape)

    # 4-1. Remove exploding traj
    JAC_traj = np.asarray(JAC_traj)
    if (dyn_sys == "baker"):
        mask = JAC_traj < 10**1
    else:
        mask = JAC_traj < 10**2
    row_sums = np.sum(mask, axis=0) # print("row", row_sums, row_sums.shape)
    columns_with_all_true = np.where(row_sums[:, ind_func] == mask.shape[0])
    valid_col = np.unique(columns_with_all_true[0])
    JAC_traj = JAC_traj[:, valid_col, :]
    print("JAC cleaned", JAC_traj.shape)


    # 5. indicator function
    # len_T x num_init x dim
    print("init", true_traj[0, :5, ind_func])
    print("sample", true_traj[tran_phase:10+tran_phase, :5, ind_func])
    print("sample 2", np.mean(true_traj[tran_phase:10+tran_phase, :5, ind_func].detach().cpu().numpy(), axis=1))


    true_avg_traj = true_traj[:, :, ind_func].detach().cpu().numpy().flatten()
    MSE_avg_traj = MSE_traj_cleaned[:, :, ind_func].flatten()
    JAC_avg_traj = JAC_traj[:, :, ind_func].flatten()
    print("avg traj shape:", JAC_avg_traj.shape, MSE_avg_traj.shape)
    print(true_avg_traj[:-20])
    print(MSE_avg_traj[:-20])

    # 6. plot dist
    pdf_path = '../plot/dist_'+str(dyn_sys)+'_'+str(N)+'_'+str(len_T)+'_'+str(ind_func)+'_'+str(model)+'.jpg'

    fig, ax1 = subplots(1,figsize=(16,8)) #, sharey=True
    sns.set_theme()

    if str(dyn_sys) == "lorenz":     # lorenz (before -> 2)
        if ind_func == 0:
            kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':3})  

            ax = sns.distplot(JAC_avg_traj, bins=100, color="slateblue", hist=True,  **kwargs)
            ax1 = sns.distplot(true_avg_traj, bins=100, color="salmon", hist=True,  **kwargs)
            ax2 = sns.distplot(MSE_avg_traj, bins=100, color="turquoise", hist=True,  **kwargs) #histtype='step', linewidth=2., 
        elif ind_func == 2:
            kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':3})  

            ax = sns.distplot(JAC_avg_traj, bins=200, color="slateblue", hist=True,   **kwargs)
            ax1 = sns.distplot(true_avg_traj, bins=200, color="salmon", hist=True, **kwargs)
            ax2 = sns.distplot(MSE_avg_traj, bins=200, color="turquoise", hist=True, **kwargs) #histtype='step', 
        elif ind_func == 1:
            sns.displot(JAC_avg_traj, bins=200, color="slateblue", density=True, histtype='step', linewidth=2., range=lorenz_range_x) #range=lorenz_range_x
            sns.displot(true_avg_traj, bins=200, color="salmon", density=True, histtype='step', linewidth=2., range=lorenz_range_x)
            sns.displot(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2., range=lorenz_range_x)
    elif str(dyn_sys) == "rossler": 
        sns.displot(JAC_avg_traj, bins=200, color="slateblue", density=True, histtype='step', linewidth=2.)
        sns.displot(true_avg_traj, bins=200, color="salmon", density=True, histtype='step', linewidth=2.)
        sns.displot(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2.)
    elif str(dyn_sys) == "hyperchaos":
        ax1.hist(JAC_avg_traj, bins=300, color="slateblue", density=True, histtype='step', linewidth=2.)
        ax1.hist(true_avg_traj, bins=300, color="salmon", density=True, histtype='step', linewidth=2.)
        ax1.hist(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2.)
    elif (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        ax1.hist(JAC_avg_traj, bins=200, color="slateblue", density=True, histtype='step', linewidth=2.)
        ax1.hist(true_avg_traj, bins=200, color="salmon", density=True, histtype='step', linewidth=2.)
        ax1.hist(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2.)
    elif str(dyn_sys) == "baker":
        ax1.hist(JAC_avg_traj, bins=200, color="slateblue", density=True, histtype='step', linewidth=2.)
        ax1.hist(true_avg_traj, bins=200, color="salmon", density=True, histtype='step', linewidth=2.)
        ax1.hist(MSE_avg_traj, bins=200, color="turquoise", density=True, histtype='step', linewidth=2.)


    ax1.grid(True)
    ax1.legend(['JAC', 'True', 'MSE'], fontsize=30)
    # ax1.legend(['True', 'MSE'], fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=34)
    ax1.yaxis.set_tick_params(labelsize=34)
    tight_layout()
    savefig(pdf_path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)