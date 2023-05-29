import numpy as np

def cosine_similarity(a,b,correlation_matrix=False,parallel=False):
    """Calculates the cosine similarity between matrices of k-mers.
    :param numpy array a: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param numpy array b: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param bool:Calculate matrix correlation(as in numpy coorcoef)"""
    n_a = a.shape[0]
    n_b = b.shape[0]
    diff_sizes = False
    if n_a != n_b:
        dummy_row = np.zeros((np.abs(n_a-n_b),) + a.shape[1:])
        diff_sizes = True
        if n_a < n_b:
            a = np.concatenate((a,dummy_row),axis=0)
        else:
            b = np.concatenate((b,dummy_row),axis=0)
    if np.ndim(a) == 1:
        num = np.dot(a,b)
        #p1 = np.linalg.norm(a)
        p1 = np.sqrt(np.sum(a**2))
        #p2 = np.linalg.norm(b)
        p2 = np.sqrt(np.sum(b**2))
        cosine_sim = num/(p1*p2)
        return cosine_sim

    elif np.ndim(a) == 2:
        if correlation_matrix:
            b = b - b.mean(axis=1)[:, None]
            a = a - a.mean(axis=1)[:, None]

        num = np.dot(a, b.T) #[seq_len,21]@[21,seq_len] = [seq_len,seq_len]
        p1 =np.sqrt(np.sum(a**2,axis=1))[:,None] #[seq_len,1]
        p2 = np.sqrt(np.sum(b ** 2, axis=1))[None, :] #[1,seq_len]
        #print(p1*p2)
        cosine_sim = num / (p1 * p2)
        if parallel:
            return cosine_sim[None,:]
        else:
            return cosine_sim
    else: #TODO: use elipsis for general approach?
        if correlation_matrix:
            b = b - b.mean(axis=2)[:, :, None]
            a = a - a.mean(axis=2)[:, :, None]
        num = np.matmul(a[:, None], np.transpose(b, (0, 2, 1))[None,:]) #[n,n,seq_len,seq_len]
        p1 = np.sqrt(np.sum(a ** 2, axis=2))[:, :, None] #Equivalent to np.linalg.norm(a,axis=2)[:,:,None]
        p2 = np.sqrt(np.sum(b ** 2, axis=2))[:, None, :] #Equivalent to np.linalg.norm(b,axis=2)[:,None,:]
        cosine_sim = num / (p1[:,None]*p2[None,:])

        if diff_sizes: #remove the dummy creation that was made avoid shape conflicts
            remove = np.abs(n_a-n_b)
            if n_a < n_b:
                cosine_sim = cosine_sim[:-remove]
            else:
                cosine_sim = cosine_sim[:,:-remove]

        return cosine_sim


def information_gain(arr,arr_mask,diag_idx_maxlen):
    """
    Calculates the amount of vector similarity/distance change between the hidden representations of the positions in the sequence for both backward and forward RNN hidden states.
    1) For a given sequence with 2 sequences of hidden states [2,L,Hidden_dim]
        []
        A) Calculate cosine similarities_old for each of the forward and backward hidden states of an RNN
        Cos_sim([L,Hidden_dim],[L,Hidden_dim]]

        b) Compute the average information gain as follows

        Forward states:        [0->1][1->2][2->3][3->4]
        Backward states: [0<-1][1<-2][2<-3][3<-4]
        ------------------------------------

    2) Make the average among the information gains of the forward and backward states (overlapping)
        Pos 0 : [0<-1]
        Pos 1 : ([0->1] + [1<-2])/2
        Pos 2 : ([1->2] + [2<-3])/2
        Pos 3 : ([2->3] + [3<-4])/2
        Pos 4 : [3->4]


    :param arr:
    :param arr_mask:
    :param diag_idx_maxlen:
    :param max_len:
    :return:
    """
    forward = None
    backward = None
    for idx in [0,1]:
        cos_sim_arr = cosine_similarity(arr[idx],arr[idx],correlation_matrix=False)
        cos_sim_diag = cos_sim_arr[diag_idx_maxlen[0][:-1],diag_idx_maxlen[1][1:]] #k=1 offset diagonal
        #Highlight: ignore the positions that have paddings
        n_paddings = (arr_mask.shape[0] - arr_mask.sum())
        keep = cos_sim_diag.shape[0] - n_paddings #number of elements in the offset diagonal - number of "False" or paddings along the sequence
        if keep <= 0: #when all the sequence is paddings or only one position is not a padding, every position gets value 0
            if idx == 0:
                forward = np.zeros((max_len-1))
            else:
                backward = np.zeros((max_len-1))
        else:
            information_gain = 1-cos_sim_diag[:keep] #or cosine distance
            #information_gain = np.abs(cos_sim_diag[:-1] -cos_sim_diag[1:])
            #Highlight: Set to 0 the information gain in the padding positions
            information_gain = np.concatenate([information_gain,np.zeros((n_paddings,))])
            if idx == 0:
                forward = information_gain
            else:
                backward = information_gain

            assert information_gain.shape[0] == max_len-1

    forward = np.insert(forward,obj=0,values=0,axis=0)
    backward = np.append(backward,np.zeros((1,)),axis=0)
    weights = (forward + backward)/2
    #weights = np.exp(weights - np.max(weights)) / np.exp(weights - np.max(weights)).sum() #softmax
    weights = (weights - weights.min()) / (weights - weights.min()).sum()
    weights*= arr_mask
    return weights[None,:]

arr =np.array([[[0.2,0.5,-0.7],[0.25,0.5,-0.6],[0.3,0.6,-0.8],[0.5,0.4,-0.3],[0.7,0.2,0.3],[-0.1,0.8,0.6]],
               [[0.1,0.9,-0.7],[0.6,0.5,-0.9],[0.3,0.6,-0.3],[0.3,-0.4,-0.7],[0.7,0.1,0.3],[-0.1,0.5,0.6]]])

arr_mask = np.array([True,True,True,True,False,False])
max_len = 6
diag_idx_maxlen = np.diag_indices(max_len)


information_gain(arr,arr_mask,diag_idx_maxlen)
