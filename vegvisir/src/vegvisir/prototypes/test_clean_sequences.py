import numpy as np

def clean_generated_sequences(seq_int,seq_mask,zero_character,min_len,max_len,keep_truncated=False):
    """"""
    seq_mask = np.array(seq_mask)
    seq_int = np.array(seq_int)
    if zero_character is not None:
        idx = np.where(seq_int == zero_character)[0]
        print(idx)
        if idx.size == 0:
            seq_mask = np.ones_like(seq_int).astype(bool)
            return (seq_int[None,:],seq_mask[None,:])
        else:
            print(("Here"))
            if idx[0] > min_len -1: #truncate sequences
                seq_int[idx[0]:] = zero_character
                seq_mask[idx[0]:] = False
                return (seq_int[None,:],seq_mask[None,:])
            else: #remove the intermediate gaps and join the remainindings to max len keep if it fits the minimum length criteria
                if keep_truncated:
                    idx_mask = np.ones_like(seq_int).astype(bool)
                    idx_mask[idx] = False
                    seq_int_masked = seq_int[idx_mask] #select only no gaps
                    seq_int_list = seq_int_masked.tolist() + [0]*(max_len-len(seq_int_masked))
                    seq_int = np.array(seq_int_list).astype(int)
                    seq_mask = seq_int.astype(bool)
                    if np.sum(seq_mask) >= min_len:
                        return (seq_int[None,:],seq_mask[None,:])

    else:
        return (seq_int[None,:],seq_mask[None,:])



seqs = np.array([[1,0,3,8,19,11,5,7,0,0,2],
                 [3,4,6,1,10,7,8,11,12,1,9],
                 [5,4,6,7,8,10,3,1,0,0,0],
                 [5,4,6,7,8,10,3,1,9,2,8],
                 ])

#seqs = np.array([[1,0,3,8,19,11,5,7,0,0,2]])

mask = np.ones_like(seqs).astype(bool)


zero_character = 0
clean_results = list(map(lambda seq_int, seq_mask: clean_generated_sequences(seq_int,
                                                                             seq_mask,
                                                                             zero_character,
                                                                             min_len=8,
                                                                             max_len=11),seqs.tolist(),mask.tolist()))

print(clean_results)