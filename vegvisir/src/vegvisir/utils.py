"""
=======================
2022-2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import argparse
import ast,warnings
import Bio.Align
import numpy as np
from collections import defaultdict
import time,datetime
def str2bool(v):
    """Converts a string into a boolean, useful for boolean arguments
    :param str v"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2None(v):
    """Converts a string into None
    :param str v"""

    if v.lower() in ('None'):
        return None
    else:
        v = ast.literal_eval(v)
        return v
def aminoacid_names_dict(aa_types,zero_characters = []):
    """ Returns an aminoacid associated to a integer value
    All of these values are mapped to 0:
        # means empty value/padding
        - means gap in an alignment
        * means stop codon
    :param int aa_types: amino acid probabilities, this number correlates to the number of different aa types in the input alignment
    :param list : character(s) to be set to 0
    """
    if aa_types == 20:
        aminoacid_names = {"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20}
    else :
        aminoacid_names = {"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20,"B":21,"Z":22,"X":23}
    if zero_characters:
        for element in zero_characters:
                aminoacid_names[element] = 0
    aminoacid_names = {k: v for k, v in sorted(aminoacid_names.items(), key=lambda item: item[1])} #sort dict by values (for dicts it is an overkill, but I like ordered stuff)
    return aminoacid_names


def create_blosum(aa_types,subs_matrix_name):
    """
    Builds an array containing the blosum scores per character
    :param aa_types: amino acid probabilities, determines the choice of BLOSUM matrix
    :param str subs_matrix_name: name of the substitution matrix, check availability at /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data"""

    if aa_types > 20 and not subs_matrix_name.startswith("PAM"):
        warnings.warn("Your dataset contains special amino acids. Switching your substitution matrix to PAM70")
        subs_matrix_name = "PAM70"
    subs_matrix = Bio.Align.substitution_matrices.load(subs_matrix_name)
    aa_list = list(aminoacid_names_dict(aa_types,zero_characters=["#"]).keys())
    index_gap = aa_list.index("#")
    aa_list[index_gap] = "*" #in the blosum matrix gaps are represanted as *

    subs_dict = defaultdict()
    subs_array = np.zeros((len(aa_list) , len(aa_list) ))
    for i, aa_1 in enumerate(aa_list):
        for j, aa_2 in enumerate(aa_list):
            if aa_1 != "*" and aa_2 != "*":
                subs_dict[(aa_1,aa_2)] = subs_matrix[(aa_1, aa_2)]
                subs_dict[(aa_2, aa_1)] = subs_matrix[(aa_1, aa_2)]
            else:
                subs_dict[(aa_1, aa_2)] = -1 #gap penalty

            subs_array[i, j] = subs_matrix[(aa_1, aa_2)]
            subs_array[j, i] = subs_matrix[(aa_2, aa_1)]

    names = np.concatenate((np.array([float("-inf")]), np.arange(0,len(aa_list))))
    subs_array = np.c_[ np.arange(0,len(aa_list)), subs_array ]
    subs_array = np.concatenate((names[None,:],subs_array),axis=0)
    #subs_array[1] = np.zeros(aa_types+1)  #replace the gap scores for zeroes , instead of [-4,-4,-4...]
    #subs_array[:,1] = np.zeros(aa_types+1)  #replace the gap scores for zeroes , instead of [-4,-4,-4...]

    #blosum_array_dict = dict(enumerate(subs_array[1:,2:])) # Highlight: Changed to [1:,2:] instead of [1:,1:] to skip the scores for non-aa elements
    blosum_array_dict = dict(enumerate(subs_array[1:,1:])) # Highlight: Changed to [1:,2:] instead of [1:,1:] to skip the scores for non-aa elements

    #blosum_array_dict[0] = np.full((aa_types+1),np.nan)  #np.nan == np.nan is False #TODO: 2 dicts, one for cosine one for %ID

    return subs_array, subs_dict, blosum_array_dict

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

def view1D(a): # a is array #TODO: Remove or investigate, is supposed to speed the comparisons up
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel()

def calculate_similarity_matrix(array, max_len, array_mask, batch_size=200, ksize=3):
    """Batched method to calculate the cosine similarity and percent identity/pairwise distance between the blosum encoded sequences.
    :param numpy array: Integer representation [n,max_len] or Blosum encoded [n,max_len,aa_types]
    :param numpy array_nan: Integer representation as [n,max_len] or Blosum encoded as [n,max_len,aa_types]. The values of the padding values (#) are represented as np.nan
    NOTE: Use smaller batches for faster results ( obviously to certain extent, check into balancing the batch size and the number of for loops)
    returns
        pairwise_similarity_matrices: [n,n,max-len,max_len] : Per sequence compare all amino acids from one sequence compared against all amino acids of the other sequence ---> useful for k-mers calculation
        percent_identity: [n,n,max_len] ---> Percent identity
        cosine_similarities: [n,n,max-len,max_len] ---> Per sequence calculate the cosine similarity among all the "amino acids blosum vectors" from one sequence compared against all "amino acids blosum vectors" of the other sequence ---> Useful for k-mers calculation
                            1 means the two aa are identical and −1 means the two aa are not similar."""
    # TODO: Make it run with Cython (faster indexing): https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
    array = array[:100]
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
    start = time.time()
    for i in idx:
        print("i ------------ {}".format(i))
        curr_array = splits[i]
        curr_mask = mask_splits[i]
        n_data_curr = curr_array.shape[0]
        rest_splits = splits.copy()
        # Highlight: Define intermediate storing arrays
        percent_identity_mean_i = np.zeros((n_data_curr, n_data))
        cosine_similarity_mean_i = np.zeros((n_data_curr, n_data))
        kmers_pid_similarity_i = np.zeros((n_data_curr, n_data))
        kmers_cosine_similarity_i = np.zeros((n_data_curr, n_data))
        start_store_point_i = 0
        end_store_point_i = rest_splits[0].shape[0]  # initialize
        start_i = time.time()
        for j, r_j in enumerate(rest_splits):  # calculate distance among all kmers per sequence in the block (n, n_kmers,n_kmers)
            # print("j {}".format(j))
            r_j_mask = mask_splits[j]
            cosine_sim_j = cosine_similarity(curr_array, r_j, correlation_matrix=False)
            if np.ndim(curr_array) == 2:
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
            # Highlight: Apply masks to calculate the similarities. NOTE: To get the data with the filled value use k = np.ma.getdata(kmers_matrix_diag_masked)
            ##PERCENT IDENTITY (binary pairwise comparison) ###############
            percent_identity_mean_ij = np.ma.masked_array(pairwise_sim_j, mask=~pid_mask_ij, fill_value=0.).mean(-1)  # Highlight: In the mask if True means to mask and ignore!!!!
            percent_identity_mean_i[:,start_store_point_i:end_store_point_i] = percent_identity_mean_ij #TODO: Probably no need to store this either
            ##COSINE SIMILARITY (pairwise comparison of cosine similarities)########################
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
            #Freeing memory: Might help
            percent_identity_mean_ij = None
            cosine_similarity_mean_ij = None
            kmers_pid_similarity_ij = None
            kmers_cosine_similarity_ij = None
            del percent_identity_mean_ij
            del cosine_similarity_mean_ij
            del kmers_pid_similarity_ij
            del kmers_cosine_similarity_ij
            start_store_point_i = end_store_point_i
            if j + 1 != len(rest_splits):
                end_store_point_i += rest_splits[j + 1].shape[0]  # it has to be the next r_j
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
    end = time.time()
    print("Overall calculation time {}".format(str(datetime.timedelta(seconds=end - start))))

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

    return np.ma.getdata(percent_identity_mean), np.ma.getdata(cosine_similarity_mean), np.ma.getdata(
        kmers_pid_similarity), np.ma.getdata(kmers_cosine_similarity)


def minmax_scale(df,column_name,low=0.,high=1.):
    """Following https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale
    WARNING USE SEPARATELY FOR TRAIN AND TEST DATASETS, OTHERWISE INFORMATION FROM THE TEST GETS TRANSFERRED TO THE TRAIN
    """
    mean_val = df[column_name].mean()
    std_val = df[column_name].std()
    max_val = df[column_name].max()
    min_val = df[column_name].min()
    df[column_name] = (df[column_name] - min_val)/max_val-min_val
    df[column_name] = df[column_name] * (high - low) + low #Scale in range min_val,max_val
    return df

def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()
        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)
        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        if p_height > 0.95:  # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width() / 2., label_position,
                '%d' % int(height),
                ha='center', va='bottom',fontsize=8,fontweight="bold")

def euclidean_2d_norm(A,B,squared=True):
    """
    Computes euclidean distance among matrix/arrays according to https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f
    Equivalent to scipy.spatial.distance_matrix(A,B)
    Note: To calculate vector euclidean distance or euclidean_1d_norm, use:
        euclidean_1d_norm = torch.sqrt(torch.sum((X1[:, None, :] - X2) ** 2,dim=2))  # equal to torch.cdist(X1,X2) or scipy.spatial.distance.cdist , which is for 1D space, for more dimensions we need the dot product
    """

    diag_AA_T = np.sum(A**2,axis=1)[:,None]
    diag_BB_T = np.sum(B**2,axis=1)
    third_component = -2*np.dot(A,B.T)
    distance = diag_AA_T + third_component + diag_BB_T
    if squared:
        distance = np.sqrt(distance)
        return distance.clip(min=0) #to avoid nan/negative values, set them to 0
    else:
        return distance.clip(min=0)