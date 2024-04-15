import EntropyHub as eh
import sys
sys.path.append('..')

from src.util import *




if __name__ == '__main__':

    '''To compute entropy, generate trajectories in 200 time unit for multiple intial points to choose any single trajectory of MSE model that does not explode. Then, on the same initial point, compute entropy for all true system, MSE model, JAC model'''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. define system
    dyn_sys= "lorenz"
    dyn_sys_f, dim = define_dyn_sys(dyn_sys)
    time_step= 0.01
    tau= 100
    len_T = tau*int(1/time_step)

    # 2. define num init points
    N = 2000
    inits = torch.randn(N, dim).double().to(device)
    if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        inits = torch.abs(inits) % 2

    every = 100
    ind_func = 2
    s = 0.
    vec_len = 100 #300

    # 3. call models
    if (dyn_sys == "baker"):
        MSE_path = "../test_result/expt_"+str(dyn_sys)+"/" + str(s)+"(MSE)/model.pt"
        JAC_path = "../test_result/expt_"+str(dyn_sys)+"/" + str(s)+"(JAC)/model.pt"
    elif (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        MSE_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+ str(s)+"(MSE)/model.pt"
        JAC_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+ str(s)+"(JAC)/model.pt"
    else:
        MSE_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+'new_MSE_0/model.pt'
        JAC_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+'new_JAC_0/model.pt'

    MSE = create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
    JAC = create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
    MSE.load_state_dict(torch.load(MSE_path))
    JAC.load_state_dict(torch.load(JAC_path))
    MSE.eval()
    JAC.eval()

    # 4. generate 3 trajectories
    one_step = torch.linspace(0, time_step, 2).to(device)

    if (dyn_sys == "henon") or (dyn_sys == "baker") or (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        true_traj = torch.zeros(len_T, inits.shape[0], inits.shape[1])
        prev_i = tau

        for j in range(inits.shape[0]):
            x = inits[j]
            for i in range(len_T):
                next_x = dyn_sys_f(x)
                true_traj[i, j] = next_x
                if x == 0.:
                    if i < prev_i:
                        print("time step that became 0", i)
                        prev_i = i
                x = next_x

        MSE_traj = vectorized_simulate_map(MSE, inits, one_step, len_T, device)
        JAC_traj = vectorized_simulate_map(JAC, inits, one_step, len_T, device)
    else:
        true_traj = vectorized_simulate(dyn_sys_f, inits, one_step, len_T, device)
        MSE_traj = vectorized_simulate(MSE, inits, one_step, len_T, device)
        JAC_traj = vectorized_simulate(JAC, inits, one_step, len_T, device)

    # 4-1. Remove exploding traj
    MSE_traj = np.asarray(MSE_traj)
    mask = MSE_traj < 10**6

    if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map") or (dyn_sys == "baker"):
        mask = MSE_traj < 10**1
    row_sums = np.sum(mask, axis=0)

    if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        columns_with_all_true = np.where(row_sums[:] == mask.shape[0])
    else:    
        columns_with_all_true = np.where(row_sums[:, ind_func] == mask.shape[0])
    valid_col = np.unique(columns_with_all_true[0])
    print("valid col", valid_col)
    valid_idx = valid_col[0]

    MSE_traj_cleaned = MSE_traj[:, valid_col, :]

    print("MSE cleaned", MSE_traj_cleaned.shape)
    print("JAC shape", JAC_traj.shape)
    print("True shape", true_traj.shape)
    print(MSE_traj_cleaned)

    # 5. indicator function     len_T x num_init x dim
    if N == 1:
        true_avg_traj = true_traj[:, :, ind_func].detach().cpu().numpy()
        MSE_avg_traj = MSE_traj_cleaned[:, :, ind_func]
        JAC_avg_traj = JAC_traj[:, :, ind_func].detach().cpu().numpy()
    else: 
        true_avg_traj = true_traj[:, valid_idx, ind_func].detach().cpu().numpy()
        MSE_avg_traj = MSE_traj[:, valid_idx, ind_func]
        JAC_avg_traj = JAC_traj[:, valid_idx, ind_func].detach().cpu().numpy()
    print("shape", MSE_avg_traj.shape, JAC_avg_traj.shape)

    # r_val = max_distance_between_points(true_avg_traj)
    # print("function r", 0.2*np.std(true_avg_traj))

    # list_tau=np.arange(1,10,1)
    # print("l", list_tau)
    # l = len(true_avg_traj)
    # tau_arr = []


    # for t in range(0,len(list_tau)):
    #     mi = mutual_info_regression(true_avg_traj[:l-int(list_tau[t])], true_avg_traj[int(list_tau[t]):].flatten())
    #     tau_arr.append(mi)
    #     print(t, mi)

    # tau_arr = np.array(tau_arr)
    # arr_wo_zeros = np.where(tau_arr == 0, np.inf, tau_arr)
    # min_non_zero = np.argmin(arr_wo_zeros)
    # print("min tau=", min_non_zero)

    true_entropy, true_ci = eh.K2En(true_avg_traj, m=1) #tau=8, Logx=np.exp(2), r=1e-5 r=1.2 , r=r_val.astype(float) , tau=int(min_non_zero+1)
    MSE_entropy, MSE_ci = eh.K2En(MSE_avg_traj, m=1)
    JAC_entropy, JAC_ci = eh.K2En(JAC_avg_traj, m=1)

    print("True", true_entropy, true_ci)
    print("MSE", MSE_entropy, MSE_ci)
    print("JAC", JAC_entropy, JAC_ci)

    # 6. plot dist
    pdf_path = '../plot/prob_dist_'+str(dyn_sys)+'_all'+'_'+str(ind_func)+'_'+str(len_T)+'.jpg'


    if (dyn_sys == "tilted_tent_map") or (dyn_sys == "plucked_tent_map") or (dyn_sys == "pinched_tent_map"):
        pdf_path = '../plot/prob_dist_'+str(dyn_sys)+'_all'+'_'+str(s)+'_'+str(len_T)+'.jpg'

        fig, ax1 = subplots(1,figsize=(16,8)) #, sharey=True
        sns.set_theme()

        ax1.hist(JAC_avg_traj, color="slateblue", density=True, alpha=0.5, bins=200)
        ax1.hist(true_avg_traj, color="salmon", density=True, alpha=0.5, bins=200)
        ax1.hist(MSE_avg_traj,  color="turquoise", density=True, alpha=0.5, bins=200)
    
        ax1.grid(True)
        ax1.legend(['JAC', 'True', 'MSE'], fontsize=30)
        # ax1.legend(['True'], fontsize=30)
        ax1.xaxis.set_tick_params(labelsize=34)
        ax1.yaxis.set_tick_params(labelsize=34)
        tight_layout()
        savefig(pdf_path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    