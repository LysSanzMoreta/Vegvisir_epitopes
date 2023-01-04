import numpy as np

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def calculate_covariance(X):
    # Extract the number of rows and columns
    N, M = X.shape
    # Calculate the covariance matrix
    cov = np.zeros((M, M))

    for i in range(M):

        # Mean of column "i"
        mean_i = np.sum(X[:, i]) / N

        for j in range(M):
            # Mean of column "j"
            mean_j = np.sum(X[:, j]) / N

            # Covariance between column "i" and column "j"
            cov[i, j] = np.sum((X[:, i] - mean_i) * (X[:, j] - mean_j)) / (N - 1)
    return cov

rng = np.random.default_rng(seed=42)
xarr = rng.random((3, 3))
crr1 = calculate_covariance(xarr)
crr2 = np.cov(xarr)

def cosine_similarity(a,b,correlation_matrix=False):
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
        return cosine_sim
    else: #TODO: use elipsis?
        if correlation_matrix:
            b = b - b.mean(axis=2)[:, :, None]
            a = a - a.mean(axis=2)[:, :, None]
        num = np.matmul(a[:, None], np.transpose(b, (0, 2, 1))[None,:]) #[n,n,seq_len,seq_len]
        p1 = np.sqrt(np.sum(a ** 2, axis=2))[:, :, None] #Equivalent to np.linalg.norm(a,axis=2)
        p2 = np.sqrt(np.sum(b ** 2, axis=2))[:, None, :]
        #p1 = np.linalg.norm(a,axis=2)[:,:,None]#TODO: 90% this is correct?
        #p2 = np.linalg.norm(b,axis=2)[:,None,:]
        cosine_sim = num / (p1[:,None]*p2[None,:])

        if diff_sizes: #remove the dummy creation that was made avoid shape conflicts
            remove = np.abs(n_a-n_b)
            if n_a < n_b:
                cosine_sim = cosine_sim[:-remove]
            else:
                cosine_sim = cosine_sim[:,:-remove]

        return cosine_sim

from scipy.spatial import distance



r = np.array([-4. ,  5. ,  0. ,  2. , -2.,  0. , -1. , -1. ,  0. ,  1. , -3. , -2.,  -2.,-1. , -3. , -3. ,-2. , -1. , -3. , -2. , -3.]) #R
h = np.array([-4.,   0. ,  8. , -1. , -1. ,  0.,  -1.,  -2.,   1.,   0.,  -3.,  -2. , -2.,-2.,  -3.,  -3. , -3.,  -2.,  -1. ,  2.,  -2.]) #H
k= np.array([-4.,   2.,  -1.,   5.,  -1.,   1.,   0.,  -1.,   0.,   1. , -3.,  -2.,  -1.,-1.,  -2.,  -3.,  -2.,  -1. , -3. , -2.,  -3.]) #K
d = np.array([-4.,  -2. , -1. , -1. ,  6.,   2.,   0.,  -1.,   1.,   0.,  -3.,  -1.,  -1.,-2. , -3. , -3.,  -4.,  -3. , -3. , -3. , -4.]) #D
#cos1 = 1- distance.cosine(a,b)
cos1= cosine_similarity(r,h)
#cos3 = 1- distance.cosine(b,d)
cos2 = cosine_similarity(h,d)

seq1 = np.concatenate((r[None,:],h[None,:],k[None,:]),axis=0)
seq2 = np.concatenate((h[None,:],k[None,:],d[None,:]),axis=0)

cos3 = cosine_similarity(seq1,seq1,correlation_matrix=True)
cos4 = cosine_similarity(seq1,seq2,correlation_matrix=True)
cos5 = cosine_similarity(seq2,seq2,correlation_matrix=True)
cos6 = cosine_similarity(seq2,seq1,correlation_matrix=True)
group1 = np.concatenate((seq1[None,:],seq2[None,:]),axis=0)
group2 = np.concatenate((seq1[None,:],seq2[None,:]),axis=0)


cos7 = cosine_similarity(group1,group2,correlation_matrix=True)
