import torch
from sklearn.metrics import mutual_info_score
import numpy as np


def calculate_mi(data,max_len):
    if data.size != 0:
        n_data = data.shape[0]
        data_idx = list(range(max_len))
        mi_matrix = np.zeros((max_len,max_len))

        for i in data_idx: #for site in the sequence
            if i+1 <= max_len:
                for j in data_idx[i+1:]: #for next site in the sequence
                    mi = mutual_info_score(data[:,i],data[:,j])
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
        return mi_matrix



def joint_sample_seq(seq,aa_types):
    seq = seq.squeeze(0)
    nseq = seq.shape[0]  # number of sequences
    maxlen = seq.shape[1]
    mi = calculate_mi(seq, 3)

    mode = torch.mode(seq, dim=0)
    freqs = torch.stack([torch.bincount(x_i, minlength=aa_types) for i, x_i in
                         enumerate(torch.unbind(seq.type(torch.int64), dim=1), 0)], dim=1)
    freqs = freqs / nseq
    common_seq = torch.zeros(maxlen)

    print(mode)
    #print(freqs)
    print(common_seq)
    print(mi)
    exit()

    for idx,(mode, pos) in enumerate(zip(mode.values,mode.indices)):
        if idx != 0:
            argmax_mi = mi[idx].argmax()
            if argmax_mi == pos:
                common_seq[idx] = mode
            else:
                second_most_freq = np.argpartition(freqs[idx],-2)[-1]
                common_seq[idx] = second_most_freq
        else:
            common_seq[idx] = mode
    return common_seq

#b = torch.tensor([[1,5,6],[1,5,10],[1,4,10],[2,3,11]])

b = torch.tensor([[1,5,6],[1,4,11],[1,5,10],[2,5,8],[1,5,7]])

common_seq = joint_sample_seq(b,21)




