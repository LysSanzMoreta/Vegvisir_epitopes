import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score,roc_curve,auc
import multiprocessing
MAX_WORKERs = ( multiprocessing. cpu_count() - 1 )


def pauc(onehot_labels,y_prob,idx):
    micro_roc_auc_ovr = roc_auc_score(
        onehot_labels[idx],
        y_prob[idx],
        multi_class="ovr",
        average="micro",
    )
    fprs = dict()
    tprs = dict()
    roc_aucs = dict()
    for i in range(2):
          fprs[i], tprs[i], _ = roc_curve(onehot_labels[:, i], y_prob[:, i])
          roc_aucs[i] = auc(fprs[i], tprs[i])

    return [micro_roc_auc_ovr,fprs,tprs,roc_aucs]
def calculate():

    labels = np.array([0,1,0,1,1,0,1,0,0,0])
    onehot = np.zeros((labels.shape[0],2))
    onehot[np.arange(0,labels.shape[0]),labels] = 1
    probs = np.abs(np.random.randn(labels.shape[0],3,2))
    probs = np.transpose(probs,(1,0,2))
    idx = np.ones_like(labels).astype(bool)


    micro_roc_auc_ovr = roc_auc_score(
        onehot[idx],
        probs[0][idx],
        multi_class="ovr",
        average="micro",
    )
    print(micro_roc_auc_ovr)

    r = Parallel(n_jobs=MAX_WORKERs)(delayed(pauc)(onehot,i,idx) for i in probs)

    print(r)
    return r


calculate()

