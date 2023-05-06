import itertools
import time,os,sys
import datetime
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import dask.array as da

import scipy
import zarr

import multiprocessing

local_repository=True
script_dir = os.path.dirname(os.path.abspath(__file__))
if local_repository: #TODO: The local imports are extremely slow
     sys.path.insert(1, "/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src")
     import vegvisir
else:#pip installed module
     import vegvisir
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.utils as VegvisirUtils

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

def extract_windows_vectorized(array, clearing_time_index, max_time, sub_window_size,only_windows=True):
    """
    Creates indexes to extract kmers from a sequence, such as:
         seq =  [A,T,R,P,V,L]
         kmers_idx = [0,1,2,1,2,3,2,3,4,3,4,5]
         seq[kmers_idx] = [A,T,R,T,R,P,R,V,L,P,V,L]
    From https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    :param int clearing_time_index: Indicates the starting index (0-python idx == 1 clearing_time_index;-1-python idx == 0 clearing_time_index)
    :param max_time: max sequence len
    :param sub_window_size:kmer size
    """
    start = clearing_time_index + 1 - sub_window_size + 1
    sub_windows = (
            start +
            # expand_dims are used to convert a 1D array to 2D array.
            np.arange(sub_window_size)[None,:]  + #[0,1,2] ---> [[0,1,2]]
            np.arange(max_time + 1)[None,:].T  #[0,...,max_len+1] ---expand dim ---> [[[0,...,max_len+1] ]], indicates the
    ) # The first row is the sum of the first row of a + the first element of b, and so on (in the diagonal the result of a[None,:] + b[None,:] is placed (without transposing b). )

    if only_windows:
        return sub_windows
    else:
        return array[:,sub_windows]


class KmersFilling(object):
    """Fills in the cosine similarities of the overlapping kmers (N,nkmers,ksize) onto (N,max_len)"""
    def __init__(self,rows_idx_a,rows_idx_b,cols_idx_a,cols_idx_b):
        self.rows_idx_a = rows_idx_a
        self.rows_idx_b = rows_idx_b
        self.cols_idx_a = cols_idx_a
        self.cols_idx_b = cols_idx_b

    def run(self,weights,hotspots):

        #return list(map(lambda row,idx_col: self.fill_kmers_array1(row,idx_col, frequencies,array),self.rows,self.idx_cols))
        return list(map(lambda row_a,row_b,idx_col: self.fill_kmers_array2(row_a,row_b,idx_col,weights,hotspots),self.rows_idx_a,self.rows_idx_b,self.cols_idx_a,self.cols_idx_b))

    def fill_kmers_array1(self,row_a,row_b,col_a,col_b,a, b):

        a[row_a, col_a[0]:col_a[1]] = b[col_b]
        return a
    def fill_kmers_array2(self,row_a,row_b,idx_col, cosine_sim_kmers,array):
        array = array.copy()
        #print(cosine_sim_kmers.shape)
        #print(cosine_sim_kmers[row_b,idx_col[0]])
        #print("iiiii")
        #array = np.array(array)
        cosine_sim_kmers = np.array(cosine_sim_kmers)
        array[row_a, idx_col[0]:idx_col[1]] += cosine_sim_kmers[row_b,idx_col[0]]
        return array

def experiment(kmers_matrix_cosine_diag_ij,diag_idx_nkmers,ksize):
    # Highlight: Finding most popular positions for conserved kmers
    n_seqs_ij = kmers_matrix_cosine_diag_ij.shape[0]
    diag_idx_ij = np.diag_indices(n_seqs_ij)

    hotspots = kmers_matrix_cosine_diag_ij.copy()
    hotspots[diag_idx_ij[0], diag_idx_ij[1]] = 0
    hotspots_mean = hotspots.mean(-1)  # [N,N,nkmers,nkmers]
    max2d = np.amax(hotspots_mean, axis=(-1, -2))  # scores of the most similar kmers between sequences
    max2d_flat = max2d.reshape(ksize * ksize)
    max2d_flat = max2d_flat[max2d_flat != 0]  # remove the diagonal 0
    max2d_idx = (hotspots_mean[..., None] > max2d_flat).any(
        -1)  # Highlight: Indicates the kmers whose values are higher than the highests (across all sequences (N vs N), not sequence specific (N vs M))
    hotspots_max_kmer_values = hotspots.copy()
    hotspots_max_kmer_values[~max2d_idx] = 0  # set to 0 the "non conserved" kmers
    hotspots_max_kmer_values = hotspots_max_kmer_values.sum(axis=1) / (hotspots_max_kmer_values.astype(bool).sum(
        axis=1))  # for the most popular/conserved kmers, find their average cosine similarity values across the sequences
    hotspots_max_kmer_values[np.isnan(hotspots_max_kmer_values)] = 0
    hotspots_max_kmer_values = hotspots_max_kmer_values.max(axis=1)

    # print(hotspots_mean[max2d_idx])
    popular_kmers_frequency = max2d_idx.astype(int).mean(axis=1)[:, diag_idx_nkmers[0],
                              diag_idx_nkmers[1]]  # frequency of the similar kmers in this position
    popular_kmers_frequency_argmax = np.argmax(popular_kmers_frequency, axis=-1)
    hotspots_max_kmer_values = hotspots_max_kmer_values[np.arange(0, n_seqs_ij), popular_kmers_frequency_argmax]

    popular_kmers_frequency_max = np.max(popular_kmers_frequency, axis=-1)
    popular_positions_frequency = np.zeros((kmers_matrix_cosine_diag_ij.shape[0], max_len))
    popular_positions_cosine = np.zeros((kmers_matrix_cosine_diag_ij.shape[0], max_len))

    idx_kmers_to_seq = np.concatenate(
        [popular_kmers_frequency_argmax[:, None], popular_kmers_frequency_argmax[:, None] + ksize], axis=1).tolist()
    idx_rows = np.arange(0, n_seqs_ij).tolist()
    popular_positions_filled_frequency = \
    KmersFilling(idx_rows, idx_kmers_to_seq).run(popular_kmers_frequency_max, popular_positions_frequency)[0]
    popular_positions_filled_cosine = \
    KmersFilling(idx_rows, idx_kmers_to_seq).run(hotspots_max_kmer_values, popular_positions_cosine)[0]
    conserved_positions = popular_positions_frequency * popular_positions_filled_cosine

    return conserved_positions

def calculate_similarity_matrix(array, max_len, array_mask, batch_size=200, ksize=3):
    """Batched method to calculate the cosine similarity and percent identity/pairwise distance between the blosum encoded sequences.
    :param numpy array: Blosum encoded sequences [n,max_len,aa_types] NOTE: TODO fix to make it work with: Integer representation [n,max_len] ?
    NOTE: Use smaller batches for faster results ( obviously to certain extent, check into balancing the batch size and the number of for loops)
    returns
        percent_identity_mean = (n_data,n_data) : 1 means the two aa sequences are identical.
        cosine_similarity_mean = (n_data,n_data):  1 means the two aa sequences are identical.
        kmers_pid_similarity = (n_data,n_data)
        kmers_cosine_similarity = (n_data,n_data)
                            """
    # TODO: Make it run with Cython (faster indexing): https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
    #array = array[:400]

    n_data = array.shape[0]
    array_mask = array_mask[:n_data]
    split_size = [int(array.shape[0] / batch_size) if not batch_size > array.shape[0] else 1][0]
    splits = np.array_split(array, split_size)
    mask_splits = np.array_split(array_mask, split_size)
    print("Generated {} splits from {} data points".format(len(splits),n_data))
    idx = list(range(len(splits)))
    if ksize >= max_len:
        ksize = max_len
    overlapping_kmers = extract_windows_vectorized(splits[0], 1, max_len - ksize, ksize,only_windows=True)

    diag_idx_ndata = np.diag_indices(n_data)
    diag_idx = np.diag_indices(ksize)
    nkmers = overlapping_kmers.shape[0]
    diag_idx_nkmers = np.diag_indices(nkmers)
    diag_idx_maxlen = np.diag_indices(max_len)

    #Highlight: Initialize the storing matrices (in the future perhaps dictionaries? but seems to withstand quite a bit)
    percent_identity_mean = np.zeros((n_data,n_data))
    cosine_similarity_mean = np.zeros((n_data,n_data))
    kmers_pid_similarity = np.zeros((n_data,n_data))
    kmers_cosine_similarity = np.zeros((n_data,n_data))

    start_store_point = 0
    end_store_point = splits[0].shape[0]
    store_point_helper = 0
    shift = 0 # avoid computing things twice

    start_store_points = []
    end_store_points = []
    start_store_points_i = []
    end_store_points_i = []
    store_point_helpers = []
    shifts = []
    i_idx = []
    j_idx=[]

    start = time.time()
    for i in idx:
        print("i ------------ {}----------------------------".format(i))
        curr_array = splits[i]
        curr_mask = mask_splits[i]
        n_data_curr = curr_array.shape[0]
        rest_splits = splits.copy()[shift:]
        # Highlight: Define intermediate storing arrays #TODO: They can be even smaller to have shape sum(rest_splits.shape)
        percent_identity_mean_i = np.zeros((n_data_curr, n_data))
        cosine_similarity_mean_i = np.zeros((n_data_curr, n_data))
        kmers_pid_similarity_i = np.zeros((n_data_curr, n_data))
        kmers_cosine_similarity_i = np.zeros((n_data_curr, n_data))
        start_store_point_i = 0 + store_point_helper
        end_store_point_i = rest_splits[0].shape[0] + store_point_helper # initialize
        start_i = time.time()
        for j, r_j in enumerate(rest_splits):  # calculate distance among all kmers per sequence in the block (n, n_kmers,n_kmers)
            print("j {}".format(j))
            r_j_mask = mask_splits[j + shift]
            cosine_sim_j = cosine_similarity(curr_array, r_j, correlation_matrix=False)
            if np.ndim(curr_array) == 2: #Integer encoded
                pairwise_sim_j = (curr_array[None, :] == r_j[:, None]).astype(int)
                pairwise_matrix_j = (curr_array[:, None, :, None] == r_j[None, :, None, :]).astype(int)
            else:
                pairwise_sim_j = (curr_array[:, None] == r_j[None, :]).all((-1)).astype(int)  # .all((-2,-1))
                # TODO: Problem when calculating self.similarity because np.nan == np.nan is False
                pairwise_matrix_j = (curr_array[:, None, :, None] == r_j[None, :, None, :]).all((-1)).astype(float)  # .all((-2,-1))

            # Highlight: Create masks to ignore the paddings of the sequences
            kmers_mask_curr_i = curr_mask[:, overlapping_kmers]
            kmers_mask_r_j = r_j_mask[:, overlapping_kmers]
            kmers_mask_ij = (kmers_mask_curr_i[:, None] * kmers_mask_r_j[None, :]).mean(-1)
            kmers_mask_ij[kmers_mask_ij != 1.] = 0.
            kmers_mask_ij = kmers_mask_ij.astype(bool)
            pid_mask_ij = curr_mask[:, None] * r_j_mask[None, :]
            # Highlight: further transformations: Basically slice the overlapping kmers and organize them to have shape
            #  [m,n,kmers,nkmers,ksize,ksize], where the diagonal contains the pairwise values between the kmers
            kmers_matrix_pid_ij = pairwise_matrix_j[:, :, :, overlapping_kmers][:, :, overlapping_kmers].transpose(0, 1, 4, 2,3, 5)
            kmers_matrix_cosine_ij = cosine_sim_j[:, :, :, overlapping_kmers][:, :, overlapping_kmers].transpose(0, 1, 4, 2,3, 5)
            # Highlight: Apply masks to calculate the similarities_old. NOTE: To get the data with the filled value use k = np.ma.getdata(kmers_matrix_diag_masked)
            ##PERCENT IDENTITY (all vs all binary  comparison) ###############
            curr_mask_expanded = np.repeat(curr_mask[:, :, None], max_len, axis=2)
            curr_mask_expanded = np.repeat(curr_mask_expanded[:,None,:],r_j_mask.shape[0],axis=1)
            r_j_mask_expanded = np.repeat(r_j_mask[:, :, None], max_len, axis=2)
            r_j_mask_expanded = np.repeat(r_j_mask_expanded[None,:], curr_mask.shape[0], axis=0)
            matrix_mask_ij = curr_mask_expanded  * r_j_mask_expanded.transpose((0,1,3,2))
            pairwise_matrix_ij = np.ma.masked_array(pairwise_matrix_j, mask=~matrix_mask_ij, fill_value=0.)
            ##PERCENT IDENTITY (binary pairwise comparison) ###############
            percent_identity_mean_ij = np.ma.masked_array(pairwise_sim_j, mask=~pid_mask_ij, fill_value=0.).mean(-1)  # Highlight: In the mask if True means to mask and ignore!!!!
            percent_identity_mean_i[:,start_store_point_i:end_store_point_i] = percent_identity_mean_ij #TODO: Probably no need to store this either
            ##COSINE SIMILARITY (all vs all cosine simlarity)########################
            cosine_similarity_ij = np.ma.masked_array(cosine_sim_j,mask=~matrix_mask_ij, fill_value=0.)  # Highlight: In the mask if True means to mask and ignore!!!!
            ##COSINE SIMILARITY (pairwise comparison of cosine similarities_old)########################
            cosine_similarity_mean_ij = np.ma.masked_array(cosine_sim_j[:, :, diag_idx_maxlen[0], diag_idx_maxlen[1]],mask=~pid_mask_ij, fill_value=0.).mean(-1)  # Highlight: In the mask if True means to mask and ignore!!!!
            cosine_similarity_mean_i[:,start_store_point_i:end_store_point_i] = cosine_similarity_mean_ij
            # KMERS PERCENT IDENTITY ############
            kmers_matrix_pid_diag_ij = kmers_matrix_pid_ij[:, :, :, :, diag_idx[0], diag_idx[1]]  # does not seem expensive
            kmers_matrix_pid_diag_mean_ij = np.mean(kmers_matrix_pid_diag_ij, axis=4)[:, :, diag_idx_nkmers[0],diag_idx_nkmers[1]]  # if we mask this only it should be fine
            kmers_pid_similarity_ij = np.ma.masked_array(kmers_matrix_pid_diag_mean_ij, mask=~kmers_mask_ij, fill_value=0.).mean(axis=2)
            kmers_pid_similarity_i[:,start_store_point_i:end_store_point_i] = kmers_pid_similarity_ij
            # KMERS COSINE SIMILARITY ########################
            kmers_matrix_cosine_diag_ij = kmers_matrix_cosine_ij[:, :, :, :, diag_idx[0],diag_idx[1]]  # does not seem expensive

            kmers_matrix_cosine_diag_mean_ij = np.nanmean(kmers_matrix_cosine_diag_ij, axis=4)[:, :, diag_idx_nkmers[0],diag_idx_nkmers[1]]
            kmers_cosine_similarity_ij = np.ma.masked_array(kmers_matrix_cosine_diag_mean_ij, mask=~kmers_mask_ij,fill_value=0.).mean(axis=2)
            kmers_cosine_similarity_i[:,start_store_point_i:end_store_point_i] = kmers_cosine_similarity_ij
            if i == j:
                # Highlight: round to nearest integer the diagonal values, due to precision issues, it computes 0.999999999 or 1.00000002 instead of 1. sometimes
                # Faster method that unravels the 2D array to 1D. Equivalent to: kmers_cosine_similarity_ij[np.diag_indices_from(cosine_similarity_mean_ij)] = np.rint(np.diagonal(kmers_cosine_similarity_ij))
                kmers_cosine_similarity_ij.ravel()[:kmers_cosine_similarity_ij.shape[1] ** 2:kmers_cosine_similarity_ij.shape[1] + 1] = np.rint(kmers_cosine_similarity_ij.ravel()[:kmers_cosine_similarity_ij.shape[1] ** 2:kmers_cosine_similarity_ij.shape[1] + 1])
                #Faster method that unravels the 2D array to 1D. Equivalent to: cosine_similarity_mean_ij[np.diag_indices_from(cosine_similarity_mean_ij)] = np.rint(np.diagonal(cosine_similarity_mean_ij))
                cosine_similarity_mean_ij.ravel()[:cosine_similarity_mean_ij.shape[1] ** 2:cosine_similarity_mean_ij.shape[1] + 1] = np.rint(cosine_similarity_mean_ij.ravel()[:cosine_similarity_mean_ij.shape[1] ** 2:cosine_similarity_mean_ij.shape[1] + 1])
                #idx = np.argwhere(diag_vals_cosine != 1.)
            #Freeing memory: Might help
            percent_identity_mean_ij = None
            cosine_similarity_mean_ij = None
            kmers_pid_similarity_ij = None
            kmers_cosine_similarity_ij = None
            del percent_identity_mean_ij
            del cosine_similarity_mean_ij
            del kmers_pid_similarity_ij
            del kmers_cosine_similarity_ij
            i_idx.append(i)
            j_idx.append(j)
            start_store_points_i.append(start_store_point_i)
            end_store_points_i.append(end_store_point_i)
            start_store_points.append(start_store_point)
            end_store_points.append(end_store_point)
            store_point_helpers.append(store_point_helper)
            shifts.append(shift)
            start_store_point_i = end_store_point_i #+ store_point_helper
            if j + 1  < len(rest_splits) :
                end_store_point_i +=  rest_splits[j + 1].shape[0]  #+ store_point_helper# it has to be the next r_j

        end_i = time.time()
        print("Time for finishing loop (i vs j) {}".format(str(datetime.timedelta(seconds=end_i - start_i))))
        percent_identity_mean[start_store_point:end_store_point] = percent_identity_mean_i
        cosine_similarity_mean[start_store_point:end_store_point] = cosine_similarity_mean_i
        kmers_cosine_similarity[start_store_point:end_store_point] = kmers_cosine_similarity_i
        kmers_pid_similarity[start_store_point:end_store_point] = kmers_pid_similarity_i
        percent_identity_mean_ij = None
        cosine_similarity_mean_ij = None
        kmers_pid_similarity_ij = None
        kmers_cosine_similarity_ij = None
        del percent_identity_mean_i
        del cosine_similarity_mean_i
        del kmers_cosine_similarity_i
        del kmers_pid_similarity_i
        start_store_point = end_store_point
        if i + 1 != len(splits):
            end_store_point += splits[i + 1].shape[0]  # it has to be the next curr_array
        shift += 1
        if i + 1 < len(splits):
            store_point_helper += splits[i + 1].shape[0]

    end = time.time()
    print("Overall calculation time {}".format(str(datetime.timedelta(seconds=end - start))))
    # print("i_idx : {}\n".format(i_idx))
    # print("j_idx : {}\n".format(j_idx))
    # print("shifts : {}\n".format(shifts))
    # print("start store points : {}\n".format(start_store_points))
    # print("end store points : {}\n".format(end_store_points))
    # print("start store points_i : {}\n".format(start_store_points_i))
    # print("end store points_i : {}\n".format(end_store_points_i))
    # print("store point helpers: {}\n".format(store_point_helpers))
    # exit()
    print("Kmers % ID")
    print(kmers_pid_similarity[0][0:4])
    print("Kmers Cosine similarity")
    print(kmers_cosine_similarity[0][0:4])
    print("Percent ID")
    print(percent_identity_mean[0][0:4])
    print("Cosine similarity")
    print(cosine_similarity_mean[0][0:4])
    print("--------------------")
    print("Kmers % ID")
    print(kmers_pid_similarity[1][0:4])
    print("Kmers Cosine similarity")
    print(kmers_cosine_similarity[1][0:4])
    print("Percent ID")
    print(percent_identity_mean[1][0:4])
    print("Cosine similarity")
    print(cosine_similarity_mean[1][0:4])
    #triu_idx = np.triu_indices(n_data,k=-1)
    percent_identity_mean = np.maximum( percent_identity_mean, percent_identity_mean.transpose() )
    cosine_similarity_mean = np.maximum(cosine_similarity_mean, cosine_similarity_mean.transpose() )
    kmers_pid_similarity = np.maximum(kmers_pid_similarity,kmers_pid_similarity.transpose())
    kmers_cosine_similarity = np.maximum(kmers_cosine_similarity,kmers_cosine_similarity.transpose())

    return np.ma.getdata(percent_identity_mean), np.ma.getdata(cosine_similarity_mean), np.ma.getdata(
        kmers_pid_similarity), np.ma.getdata(kmers_cosine_similarity)

def importance_weight(hotspots,nkmers,ksize,max_len,positional_mask):
    """Weighting cosine similarities across kmers to find which positions in the sequence are more conserved
    :param kmers_matrix_cosine_diag_ij : [N,N,nkmers,nkmers,ksize]
    """
    # Highlight: Finding most popular positions for conserved kmers
    print("Here1")
    n_seqs_i = hotspots.shape[0]
    print("Here2")

    diag_idx_i = np.diag_indices(n_seqs_i)
    print("Here3")

    #hotspots[diag_idx_i[0], diag_idx_i[1]] = 0
    print("Here4")

    hotspots_mean = hotspots.sum(axis=1) / hotspots.astype(bool).astype(int).sum(axis=1)
    print("Here5")

    hotspots_mean = hotspots_mean.sum(axis=2) / hotspots_mean.astype(bool).astype(int).sum(axis=2)
    print("Here6")

    positions_weights = da.zeros((n_seqs_i, max_len))
    rows_idx_a = da.from_array(np.repeat(np.arange(0, n_seqs_i), nkmers))
    rows_idx_b = da.from_array(np.repeat(np.arange(0, n_seqs_i), nkmers))
    cols_idx_a_0 = da.from_array(np.tile(np.arange(0, nkmers), n_seqs_i))  # [0,1,0,1,0,1]
    cols_idx_a_1 = da.from_array(np.tile(np.arange(ksize, ksize * nkmers - 1), n_seqs_i))
    cols_idx_a = da.concatenate([cols_idx_a_0[:, None], cols_idx_a_1[:, None]], axis=1)
    if nkmers % 2 == 0:
        divisors = da.from_array(np.concatenate([np.arange(0, ksize - 1), np.arange(0, ksize - 1)[
                                                            ::-1]])) + 1  # divide each position in the sequence by the number of kmers that contributed to the result (weighted average)
    else:
        divisors = da.from_array(np.concatenate([np.arange(0, ksize), np.arange(0, ksize - 1)[::-1]])) + 1  # divide each position in the sequence by the number of kmers that contributed to the result (weighted average)
    positional_weights = sum(KmersFilling(rows_idx_a, rows_idx_b, cols_idx_a,cols_idx_b).run(hotspots_mean, positions_weights))/divisors
    print("Here8")

    positional_weights = (positional_weights - positional_weights.min()) / (positional_weights.max() - positional_weights.min()) #min max scale
    positional_weights*= positional_mask

    return positional_weights

def process_value(iterables_args,fixed_args):

    i,j,shift,start_store_point,end_store_point,store_point_helper,start_store_point_i,end_store_point_i = iterables_args
    splits, mask_splits, n_data,max_len, overlapping_kmers, diag_idx_ksize, diag_idx_maxlen, diag_idx_nkmers = fixed_args
    print("i ------------ {}----------------------------".format(i))
    curr_array = splits[i]
    curr_mask = mask_splits[i]
    n_data_curr = curr_array.shape[0]
    rest_splits = splits.copy()[shift:]
    # Highlight: Define intermediate storing arrays #TODO: They can be even smaller to have shape sum(rest_splits.shape)
    start_i = time.time()
    print("###### j {} ###".format(j))
    r_j = rest_splits[j]
    r_j_mask = mask_splits[j + shift]
    cosine_sim_j = cosine_similarity(curr_array, r_j, correlation_matrix=False)
    if np.ndim(curr_array) == 2:  # Integer encoded
        pairwise_sim_j = (curr_array[None, :] == r_j[:, None]).astype(int)
        pairwise_matrix_j = (curr_array[:, None, :, None] == r_j[None, :, None, :]).astype(int)
    else:
        pairwise_sim_j = (curr_array[:, None] == r_j[None, :]).all((-1)).astype(int)  # .all((-2,-1)) #[1,L]
        # TODO: Problem when calculating self.similarity because np.nan == np.nan is False
        pairwise_matrix_j = (curr_array[:, None, :, None] == r_j[None, :, None, :]).all((-1)).astype(float)  # .all((-2,-1)) #[1,L,L]
    # Highlight: Create masks to ignore the paddings of the sequences
    kmers_mask_curr_i = curr_mask[:, overlapping_kmers]
    kmers_mask_r_j = r_j_mask[:, overlapping_kmers]
    kmers_mask_ij = (kmers_mask_curr_i[:, None] * kmers_mask_r_j[None, :]).mean(-1)
    kmers_mask_ij[kmers_mask_ij != 1.] = 0.
    kmers_mask_ij = kmers_mask_ij.astype(bool)
    pid_mask_ij = curr_mask[:, None] * r_j_mask[None, :]
    # Highlight: further transformations: Basically slice the overlapping kmers and organize them to have shape
    #  [m,n,kmers,nkmers,ksize,ksize], where the diagonal contains the pairwise values between the kmers
    kmers_matrix_pid_ij = pairwise_matrix_j[:, :, :, overlapping_kmers][:, :, overlapping_kmers].transpose(0, 1,
                                                                                                           4, 2,
                                                                                                           3, 5)
    kmers_matrix_cosine_ij = cosine_sim_j[:, :, :, overlapping_kmers][:, :, overlapping_kmers].transpose(0, 1,
                                                                                                         4, 2,
                                                                                                         3, 5)
    # Highlight: Apply masks to calculate the similarities_old. NOTE: To get the data with the filled value use k = np.ma.getdata(kmers_matrix_diag_masked)
    ##PERCENT IDENTITY (all vs all comparison)

    curr_mask_expanded = np.repeat(curr_mask[:, :, None], max_len, axis=2)
    curr_mask_expanded = np.repeat(curr_mask_expanded[:, None, :], r_j_mask.shape[0], axis=1)
    r_j_mask_expanded = np.repeat(r_j_mask[:, :, None], max_len, axis=2)
    r_j_mask_expanded = np.repeat(r_j_mask_expanded[None, :], curr_mask.shape[0], axis=0)
    matrix_mask_ij = curr_mask_expanded * r_j_mask_expanded.transpose((0, 1, 3, 2))
    pid_pairwise_matrix_ij = np.ma.masked_array(pairwise_matrix_j, mask=~matrix_mask_ij, fill_value=0.) #[1,L,L]
    ##PERCENT IDENTITY (binary pairwise comparison) ###############
    percent_identity_mean_ij = np.ma.masked_array(pairwise_sim_j, mask=~pid_mask_ij, fill_value=0.).mean(-1)  # Highlight: In the mask if True means to mask and ignore!!!!
    #percent_identity_i[:,start_store_point_i:end_store_point_i] = percent_identity_ij  # TODO: Probably no need to store this either
    #percent_identity_mean_i[:,start_store_point_i:end_store_point_i] = percent_identity_mean_ij  # TODO: Probably no need to store this either
    ##COSINE SIMILARITY (all vs all cosine simlarity)########################
    cosine_sim_pairwise_matrix_ij = np.ma.masked_array(cosine_sim_j, mask=~matrix_mask_ij, fill_value=0.) # [1,L,L] # Highlight: In the mask if True means to mask and ignore!!!!
    ##COSINE SIMILARITY (pairwise comparison of cosine similarities_old)########################
    cosine_similarity_mean_ij = np.ma.masked_array(cosine_sim_j[:, :, diag_idx_maxlen[0], diag_idx_maxlen[1]],mask=~pid_mask_ij, fill_value=0.).mean(-1)  # Highlight: In the mask if True means to mask and ignore!!!!
    #cosine_similarity_mean_i[:, start_store_point_i:end_store_point_i] = cosine_similarity_mean_ij
    # KMERS PERCENT IDENTITY ############
    kmers_matrix_pid_diag_ij = kmers_matrix_pid_ij[:, :, :, :, diag_idx_ksize[0],diag_idx_ksize[1]]  # does not seem expensive
    kmers_matrix_pid_diag_mean_ij = np.mean(kmers_matrix_pid_diag_ij, axis=4)[:, :, diag_idx_nkmers[0],diag_idx_nkmers[1]]  # if we mask this only it should be fine
    kmers_pid_similarity_ij = np.ma.masked_array(kmers_matrix_pid_diag_mean_ij, mask=~kmers_mask_ij,fill_value=0.).mean(axis=2)
    #kmers_pid_similarity_i[:, start_store_point_i:end_store_point_i] = kmers_pid_similarity_ij
    # KMERS COSINE SIMILARITY ########################
    kmers_matrix_cosine_diag_ij = kmers_matrix_cosine_ij[:, :, :, :, diag_idx_ksize[0],diag_idx_ksize[1]]  # does not seem expensive
    kmers_matrix_cosine_diag_mean_ij = np.nanmean(kmers_matrix_cosine_diag_ij, axis=4)[:, :, diag_idx_nkmers[0],diag_idx_nkmers[1]]
    kmers_cosine_similarity_ij = np.ma.masked_array(kmers_matrix_cosine_diag_mean_ij, mask=~kmers_mask_ij,fill_value=0.).mean(axis=2)
    #kmers_cosine_similarity_i[:, start_store_point_i:end_store_point_i] = kmers_cosine_similarity_ij
    if i == j:
        # Highlight: round to nearest integer the diagonal values, due to precision issues, it computes 0.999999999 or 1.00000002 instead of 1. sometimes
        # Faster method that unravels the 2D array to 1D. Equivalent to: kmers_cosine_similarity_ij[np.diag_indices_from(cosine_similarity_mean_ij)] = np.rint(np.diagonal(kmers_cosine_similarity_ij))
        kmers_cosine_similarity_ij.ravel()[:kmers_cosine_similarity_ij.shape[1] ** 2:kmers_cosine_similarity_ij.shape[1] + 1] = np.rint(kmers_cosine_similarity_ij.ravel()[:kmers_cosine_similarity_ij.shape[1] ** 2:kmers_cosine_similarity_ij.shape[1] + 1])
        # Faster method that unravels the 2D array to 1D. Equivalent to: cosine_similarity_mean_ij[np.diag_indices_from(cosine_similarity_mean_ij)] = np.rint(np.diagonal(cosine_similarity_mean_ij))
        cosine_similarity_mean_ij.ravel()[:cosine_similarity_mean_ij.shape[1] ** 2:cosine_similarity_mean_ij.shape[1] + 1] = np.rint(cosine_similarity_mean_ij.ravel()[:cosine_similarity_mean_ij.shape[1] ** 2:cosine_similarity_mean_ij.shape[1] + 1])
    end_i = time.time()
    print("Time for finishing loop (i vs j) {}".format(str(datetime.timedelta(seconds=end_i - start_i))))

    return percent_identity_mean_ij,\
        cosine_similarity_mean_ij,\
        kmers_cosine_similarity_ij,\
        kmers_pid_similarity_ij, \
        kmers_matrix_cosine_diag_ij,\
        start_store_point,end_store_point,start_store_point_i,end_store_point_i

def inner_loop(params):
    # fn = CosineClass()
    iterables, fixed = params
    return process_value(iterables,fixed_args=fixed)

class SimilarityParallel:
   def __init__(self,iterables,fixed_args):
       self.fixed = fixed_args
       self.i_idx = iterables["i_idx"]
       self.j_idx = iterables["j_idx"]
       self.shifts = iterables["shifts"]
       self.start_store_points = iterables["start_store_points"]
       self.end_store_points = iterables["end_store_points"]
       self.store_point_helpers = iterables["store_point_helpers"]
       self.end_store_points_i = iterables["end_store_points_i"]
       self.start_store_points_i = iterables["start_store_points_i"]
       self.iterables = self.i_idx,self.j_idx,self.shifts,self.start_store_points,self.end_store_points,self.store_point_helpers,self.start_store_points_i,self.end_store_points_i

   def outer_loop(self, pool):

       return list(pool.map(inner_loop, list(zip(zip(*self.iterables), itertools.repeat(self.fixed)))))

def fill_array(array_fixed,ij,start,end,start_i,end_i):
    array_fixed[start:end,start_i:end_i] = ij
    return array_fixed

def fill_array_map(array_fixed,ij_arrays,starts,ends,starts_i,ends_i):
     results = list(map(lambda ij,start,end,start_i,end_i: fill_array(array_fixed,ij,start,end,start_i,end_i),ij_arrays,starts,ends,starts_i,ends_i))
     return results[0]

def calculate_similarity_matrix_parallel(array, max_len, array_mask, batch_size=200, ksize=3):
    """Batched method to calculate the cosine similarity and percent identity/pairwise distance between the blosum encoded sequences.
    :param numpy array: Blosum encoded sequences [n,max_len,aa_types] NOTE: TODO fix to make it work with: Integer representation [n,max_len] ?
    NOTE: Use smaller batches for faster results ( obviously to certain extent, check into balancing the batch size and the number of for loops)
    returns
        percent_identity_mean = (n_data,n_data) : 1 means the two aa sequences are identical.
        cosine_similarity_mean = (n_data,n_data):  1 means the two aa sequences are identical.
        kmers_pid_similarity = (n_data,n_data)
        kmers_cosine_similarity = (n_data,n_data)
                            """
    # array = array[:400]

    n_data = array.shape[0]
    array_mask = array_mask[:n_data]
    assert array_mask.shape == (n_data,max_len)
    split_size = [int(array.shape[0] / batch_size) if not batch_size > array.shape[0] else 1][0]
    splits = np.array_split(array, split_size)
    mask_splits = np.array_split(array_mask, split_size)
    print("Generated {} splits from {} data points".format(len(splits), n_data))


    if ksize >= max_len:
        ksize = max_len
    overlapping_kmers = extract_windows_vectorized(splits[0], 1, max_len - ksize, ksize, only_windows=True)

    diag_idx_ndata =np.diag_indices(n_data)
    diag_idx_ksize = np.diag_indices(ksize)
    nkmers = overlapping_kmers.shape[0]
    diag_idx_nkmers = np.diag_indices(nkmers)
    diag_idx_maxlen = np.diag_indices(max_len)
    print("maybe")
    kmers_cosine_similarity_matrix_diag = da.random.randint(-800,800,(10000,10000,11,11,3))
    print("ggggggggggg")
    array_mask = da.ones((10000,11)).astype(bool)
    positional_weights = importance_weight(kmers_cosine_similarity_matrix_diag,nkmers,ksize,array_mask)
    exit()

    # Highlight: Initialize the storing matrices (in the future perhaps dictionaries? but seems to withstand quite a bit)
    percent_identity_mean = np.zeros((n_data, n_data))
    #pid_pairwise_matrix= np.zeros((n_data, n_data,max_len,max_len))
    cosine_similarity_mean = np.zeros((n_data, n_data))
    #cosine_sim_pairwise_matrix= np.zeros((n_data, n_data,max_len,max_len))
    kmers_pid_similarity = np.zeros((n_data, n_data))
    kmers_cosine_similarity = np.zeros((n_data, n_data))
    kmers_cosine_similarity_matrix_diag = np.zeros((n_data, n_data,nkmers,nkmers,ksize))


    #Iterables
    idx = list(range(len(splits)))
    #shifts = list(range(len(splits)))
    shifts = []
    start_store_points = []
    start_store_points_i = []
    store_point_helpers = []
    end_store_points = []
    end_store_points_i = []
    i_idx = []
    j_idx = []
    start_store_point = 0
    store_point_helper = 0
    end_store_point = splits[0].shape[0]
    #TODO: Find pattern?
    for i in idx:
        shift = i
        rest_splits = splits.copy()[shift:]
        start_store_point_i = 0 + store_point_helper
        end_store_point_i = rest_splits[0].shape[0] + store_point_helper  # initialize
        for j, r_j in enumerate(rest_splits):  # calculate distance among all kmers per sequence in the block (n, n_kmers,n_kmers)
            i_idx.append(i)
            shifts.append(shift)
            j_idx.append(j)
            start_store_points.append(start_store_point)
            store_point_helpers.append(store_point_helper)
            end_store_points.append(end_store_point)
            start_store_points_i.append(start_store_point_i)
            end_store_points_i.append(end_store_point_i)
            start_store_point_i = end_store_point_i  # + store_point_helper
            if j + 1 < len(rest_splits):
                end_store_point_i += rest_splits[j + 1].shape[0]  # + store_point_helper# it has to be the next r_j
        start_store_point = end_store_point
        if i + 1 < len(splits):
            store_point_helper += splits[i + 1].shape[0]
        if i + 1 != len(splits):
            end_store_point += splits[i + 1].shape[0]  # it has to be the next curr_array
        else:
            pass
    start = time.time()
    #args_fixed = splits, mask_splits, n_data,max_len, overlapping_kmers, diag_idx, diag_idx_maxlen, diag_idx_nkmers, percent_identity_mean,percent_identity, cosine_similarity_mean, kmers_cosine_similarity, kmers_pid_similarity
    args_fixed = splits, mask_splits, n_data,max_len, overlapping_kmers, diag_idx_ksize, diag_idx_maxlen, diag_idx_nkmers
    args_iterables = {"i_idx":i_idx,
                      "j_idx":j_idx,
                      "shifts":shifts,
                      "start_store_points":start_store_points,
                      "start_store_points_i": start_store_points_i,
                      "store_point_helpers":store_point_helpers,
                      "end_store_points":end_store_points,
                      "end_store_points_i": end_store_points_i
                      }

    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        results = SimilarityParallel(args_iterables,args_fixed).outer_loop(pool)
        zipped_results =list(zip(*results))
        starts,ends,starts_i,ends_i = zipped_results[5],zipped_results[6],zipped_results[7],zipped_results[8]
        #pid_pairwise_matrix_ij = zipped_results[0]
        #pid_pairwise_matrix= fill_array_map(pid_pairwise_matrix,pid_pairwise_matrix_ij,starts,ends,starts_i,ends_i) #TODO: How to improve this?
        #cosine_sim_pairwise_matrix_ij = zipped_results[1]
        #cosine_sim_pairwise_matrix= fill_array_map(cosine_sim_pairwise_matrix,cosine_sim_pairwise_matrix_ij,starts,ends,starts_i,ends_i)
        percent_identity_mean_ij = zipped_results[0]
        percent_identity_mean= fill_array_map(percent_identity_mean,percent_identity_mean_ij,starts,ends,starts_i,ends_i)
        cosine_similarity_mean_ij = zipped_results[1]
        cosine_similarity_mean= fill_array_map(cosine_similarity_mean,cosine_similarity_mean_ij,starts,ends,starts_i,ends_i)
        kmers_cosine_similarity_ij = zipped_results[2]
        kmers_cosine_similarity= fill_array_map(kmers_cosine_similarity,kmers_cosine_similarity_ij,starts,ends,starts_i,ends_i)
        kmers_pid_similarity_ij = zipped_results[3]
        kmers_pid_similarity= fill_array_map(kmers_pid_similarity,kmers_pid_similarity_ij,starts,ends,starts_i,ends_i)
        kmers_matrix_cosine_diag_ij = zipped_results[4]
        kmers_cosine_similarity_matrix_diag = fill_array_map(kmers_cosine_similarity_matrix_diag,kmers_matrix_cosine_diag_ij,starts,ends,starts_i,ends_i)

    end = time.time()
    print("Overall calculation time {}".format(str(datetime.timedelta(seconds=end - start))))
    #Highlight: Mirror values across the diagonal
    #pid_pairwise_matrix = np.maximum(pid_pairwise_matrix, pid_pairwise_matrix.transpose(1,0,2,3))
    #cosine_sim_pairwise_matrix = np.maximum(cosine_sim_pairwise_matrix, cosine_sim_pairwise_matrix.transpose(1,0,2,3))
    positional_weights = importance_weight(kmers_cosine_similarity_matrix_diag,nkmers,ksize,max_len,array_mask)
    percent_identity_mean = np.maximum(percent_identity_mean, percent_identity_mean.transpose())
    cosine_similarity_mean = np.maximum(cosine_similarity_mean, cosine_similarity_mean.transpose())
    kmers_pid_similarity = np.maximum(kmers_pid_similarity, kmers_pid_similarity.transpose())
    kmers_cosine_similarity = np.maximum(kmers_cosine_similarity, kmers_cosine_similarity.transpose())

    return np.ma.getdata(positional_weights),\
        np.ma.getdata(percent_identity_mean), \
        np.ma.getdata(cosine_similarity_mean), \
        np.ma.getdata(
        kmers_pid_similarity), \
        np.ma.getdata(kmers_cosine_similarity)

if __name__ == '__main__':  # <- prevent RuntimeError for 'spawn'

    max_len = 4
    seqs = ["AHPD","ALSW","VLPY","TRMF","IKNM"]#,"FYRA"]
    #seqs = ["AHPD"]
    sequences_padded = VegvisirLoadUtils.SequencePadding(seqs, max_len, "ends",False).run()
    sequences, mask  = zip(*sequences_padded)  # unpack list of tuples onto 2 lists
    blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(21, "BLOSUM62",
                                                                               zero_characters=["#"],
                                                                               include_zero_characters=True)
    sequences_array = np.array(sequences)

    aa_dict = VegvisirUtils.aminoacid_names_dict(21, zero_characters=["#"])
    sequences_int = np.vectorize(aa_dict.get)(sequences_array)
    sequences_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(sequences_int)
    #sequences_mask = np.ones_like(sequences_blosum).astype(bool)[:,:,0]
    sequences_mask = np.array([[True,True,True,False],[True,True,True,True],[True,False,False,False],[True,True,True,False],[True,True,False,False]])#,[True,True,True,True]])
    #a,b,c,d = calculate_similarity_matrix(sequences_blosum,max_len,sequences_mask,batch_size=2)



    a,b,c,d,e = calculate_similarity_matrix_parallel(sequences_blosum,max_len,sequences_mask,batch_size=2)
    zarr.save("a.zarr",a)
    w = zarr.load('a.zarr')
    print(w)

    exit()
