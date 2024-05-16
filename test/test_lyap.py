from torch import *
from matplotlib.pyplot import *
import pandas as pd

# true_les = pd.read_csv('TRUE_LE.csv')
mse_les = pd.read_csv('MSE_LE.csv')
jac_les = pd.read_csv('JAC_LE.csv')

# true_les = tensor(true_les.values)
mse_les = tensor(mse_les.values)
jac_les = tensor(jac_les.values)

# true_les_mean = true_les.mean(axis=0)
mse_les_mean = mse_les.mean(axis=0)
jac_les_mean = jac_les.mean(axis=0)


n = mse_les.shape[0]
rn = arange(0, n)
sn = sqrt(tensor(n))

# true_les = true_les[::30,:]
mse_les = mse_les[::30,:]
jac_les = jac_les[::30,:]
n = mse_les.shape[0]
rn = rn[::30]*0.01
# true_les_sem = true_les.std(axis=0)
mse_les_sem = mse_les.std(axis=0)
jac_les_sem = jac_les.std(axis=0)

for i in arange(0, mse_les.shape[1]):
    fig, ax = subplots()

    # ax.errorbar(rn, true_les[:,i], yerr=true_les_sem[i]*ones(n), linestyle='none', alpha=0.5, color="k", marker="s", mfc="k", mec="k", ms=5, label="true")
    ax.errorbar(rn, mse_les[:,i], yerr=mse_les_sem[i]*ones(n), linestyle='none', alpha=0.5, color="r", marker="P", mfc="r", mec="r", ms=5, label="mse")
    ax.errorbar(rn, jac_les[:,i], yerr=jac_les_sem[i]*ones(n), linestyle='none', alpha=0.5, color="b", marker="o", mfc="b", mec="b", ms=5, label="jac")
    # ax.plot(rn, true_les_mean[i]*ones(n), "k", lw=1.5)
    ax.plot(rn,mse_les_mean[i]*ones(n), "r", lw=1.5)
    ax.plot(rn, jac_les_mean[i]*ones(n), "b", lw=1.5)   

    ax.set_xlabel("time", fontsize=24)
    ax.set_ylabel("LE {}".format(i+1), fontsize=24)
    ax.legend(fontsize=24,markerscale=3)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.grid(True)
    tight_layout()
    show()

