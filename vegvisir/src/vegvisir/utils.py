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
import os,shutil
from collections import defaultdict
import time,datetime
from sklearn import preprocessing

import pandas as pd
import torch
import vegvisir.plots as VegvisirPlots
from scipy import stats
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
        try:
            v = ast.literal_eval(v)
        except:
            v = str(v)
        return v

def folders(folder_name,basepath):
    """ Creates a folder at the indicated location. It rewrites folders with the same name
    :param str folder_name: name of the folder
    :param str basepath: indicates the place where to create the folder
    """
    #basepath = os.getcwd()
    if not basepath:
        newpath = folder_name
    else:
        newpath = basepath + "/%s" % folder_name
    if not os.path.exists(newpath):
        try:
            original_umask = os.umask(0)
            os.makedirs(newpath, 0o777)
        finally:
            os.umask(original_umask)
    else:
        print("removing subdirectories") #if this is reached if because you are running the folders function twice with the same folder name
        shutil.rmtree(newpath)  # removes all the subdirectories!
        os.makedirs(newpath,0o777)

def aminoacid_names_dict(aa_types,zero_characters = []):
    """ Returns an aminoacid associated to a integer value
    All of these values are mapped to 0:
        # means empty value/padding
        - means gap in an alignment
        * means stop codon
    :param int aa_types: amino acid probabilities, this number correlates to the number of different aa types in the input alignment
    :param list zero_characters: character(s) to be set to 0
    """
    if aa_types == 20 :
        aminoacid_names = {"R":0,"H":1,"K":2,"D":3,"E":4,"S":5,"T":6,"N":7,"Q":8,"C":9,"G":10,"P":11,"A":12,"V":13,"I":14,"L":15,"M":16,"F":17,"Y":18,"W":19}
    elif aa_types == 21:
        aminoacid_names = {"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20}
    else :
        aminoacid_names = {"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20,"B":21,"Z":22,"X":23}
    if zero_characters:
        for element in zero_characters:
                aminoacid_names[element] = 0
    aminoacid_names = {k: v for k, v in sorted(aminoacid_names.items(), key=lambda item: item[1])} #sort dict by values (for dicts it is an overkill, but I like ordered stuff)
    return aminoacid_names

def convert_to_onehot(a,dimensions):
    #ncols = a.max() + 1
    out = np.zeros((a.size, dimensions), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (dimensions,)
    return out

def create_blosum(aa_types,subs_matrix_name,zero_characters=[],include_zero_characters=False):
    """
    Builds an array containing the blosum scores per character
    :param aa_types: amino acid probabilities, determines the choice of BLOSUM matrix
    :param str subs_matrix_name: name of the substitution matrix, check availability at /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data
    :param bool include_zero_characters : If True the score for the zero characters is kept in the blosum encoding, so the vector will have size 21 instead of just 20
    """

    if aa_types > 21 and not subs_matrix_name.startswith("PAM"):
        warnings.warn("Your dataset contains special amino acids. Switching your substitution matrix to PAM70")
        subs_matrix_name = "PAM70"

    subs_matrix = Bio.Align.substitution_matrices.load(subs_matrix_name)
    aa_list = list(aminoacid_names_dict(aa_types,zero_characters=zero_characters).keys())
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
    if include_zero_characters:
        blosum_array_dict = dict(enumerate(subs_array[1:,1:]))
    else:
        blosum_array_dict = dict(enumerate(subs_array[1:, 2:])) # Highlight: Changed to [1:,2:] instead of [1:,1:] to skip the scores for non-aa elements
    #blosum_array_dict[0] = np.full((aa_types),0)  #np.nan == np.nan is False ...

    return subs_array, subs_dict, blosum_array_dict

class AUK:
    """Slighlty re-adapted implementation from https://towardsdatascience.com/auk-a-simple-alternative-to-auc-800e61945be5
      """
    def __init__(self, probabilities, labels, integral='trapezoid'):
        self.probabilities = probabilities
        self.labels = labels
        self.integral = integral
        if integral not in ['trapezoid', 'max', 'min']:
            raise ValueError('"' + str(
                integral) + '"' + ' is not a valid integral value. Choose between "trapezoid", "min" or "max"')
        self.probabilities_set = sorted(list(set(probabilities)))#[0.1]
        self.n_data = len(probabilities)
    # make predictions based on the threshold value and self.probabilities
    def _make_predictions(self, threshold):
        predictions = np.zeros(self.n_data)
        probabilities_arr = np.array(self.probabilities)
        idx, = np.where(probabilities_arr >= threshold)
        predictions[idx] = 1
        return predictions

    # make list with kappa scores for each threshold
    def kappa_curve(self):
        kappa_list = []

        for thres in self.probabilities_set:
            preds = self._make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            k = self.calculate_kappa(tp, tn, fp, fn)
            kappa_list.append(k)
        return self._add_zero_to_curve(kappa_list)

    # make list with fpr scores for each threshold
    def fpr_curve(self):
        fpr_list = []
        for thres in self.probabilities_set:
            preds = self._make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            fpr = self.calculate_fpr(fp, tn)
            fpr_list.append(fpr)

        return self._add_zero_to_curve(fpr_list)

    # calculate confusion matrix
    def confusion_matrix(self, predictions):
        """Calculates true positives, true negatives, false positives and false negatives"""
        labels_arr = self.labels
        idx = predictions == self.labels
        positives, = np.where(predictions[idx]==1)
        tp = len(positives)
        negatives, = np.where(predictions[idx]!=1)
        tn = len(negatives)
        idx_positives, = np.where((predictions==1)&(predictions!=labels_arr))
        fp = len(predictions[idx_positives])
        idx_negatives, = np.where((predictions!=1)&(predictions!=labels_arr))
        fn = len(predictions[idx_negatives])
        total = tp + tn + fp + fn

        return tp / total, tn / total, fp / total, fn / total

    # Calculate AUK
    def calculate_auk(self):
        auk = 0
        fpr_list = self.fpr_curve()

        for i, prob in enumerate(self.probabilities_set[:-1]):
            x_dist = abs(fpr_list[i + 1] - fpr_list[i])

            preds = self._make_predictions(prob)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp1 = self.calculate_kappa(tp, tn, fp, fn)

            preds = self._make_predictions(self.probabilities_set[i + 1])
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp2 = self.calculate_kappa(tp, tn, fp, fn)

            y_dist = abs(kapp2 - kapp1)
            bottom = min(kapp1, kapp2) * x_dist
            auk += bottom
            if self.integral == 'trapezoid':
                top = (y_dist * x_dist) / 2
                auk += top
            elif self.integral == 'max':
                top = (y_dist * x_dist)
                auk += top
            else:
                continue
        return auk

    # Calculate the false-positive rate
    def calculate_fpr(self, fp, tn):
        return fp / (fp + tn)

    # Calculate kappa score
    def calculate_kappa(self, tp, tn, fp, fn):
        acc = tp + tn
        p = tp + fn
        p_hat = tp + fp
        n = fp + tn
        n_hat = fn + tn
        p_c = p * p_hat + n * n_hat
        return (acc - p_c) / (1 - p_c)

    # Add zero to appropriate position in list
    def _add_zero_to_curve(self, curve):
        min_index = curve.index(min(curve))
        if min_index > 0:
            curve.append(0)
        else:
            curve.insert(0, 0)
        return curve  # Add zero to appropriate position in list

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

def view1D(a): # a is array #TODO: Remove or investigate, is supposed to speed the comparisons up
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel()

def calculate_similarity_matrix_slow(array, max_len, array_mask, batch_size=200, ksize=3):
    """Batched method to calculate the cosine similarity and percent identity/pairwise distance between the blosum encoded sequences.
    :param numpy array: Integer representation [n,max_len] or Blosum encoded [n,max_len,aa_types]
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

    d = kmers_cosine_similarity.ravel()[:kmers_cosine_similarity.shape[1] ** 2:kmers_cosine_similarity.shape[1] + 1]
    idx = np.argwhere(d != 1.)
    print(idx)

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

def minmax_scale(array,suffix=None,column_name=None,low=0.,high=1.):
    """Following https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale
    WARNING USE SEPARATELY FOR TRAIN AND TEST DATASETS, OTHERWISE INFORMATION FROM THE TEST GETS TRANSFERRED TO THE TRAIN
    """
    if isinstance(array,pd.DataFrame):
        assert column_name is not None, "Please select a column name"
        assert suffix is not None, "Please select a suffix of the new column or give an empty string to overwrite it"
        array["{}{}".format(column_name,suffix)]  =preprocessing.MinMaxScaler().fit_transform(array[column_name].to_numpy()[:,None])
        return array
    elif isinstance(array,np.ndarray):
        return preprocessing.MinMaxScaler().fit_transform(array)
    elif isinstance(array,torch.Tensor):
        return torch.from_numpy(preprocessing.MinMaxScaler().fit_transform(array.to_numpy()))
    else:
        raise ValueError("Not implemented for this data type")

def features_preprocessing(array,method="minmax"):
    """Applies a preprocessing procedure to each feature independently
    Notes:
        - https://datascience.stackexchange.com/questions/54908/data-normalization-before-or-after-train-test-split
    :returns
    array_scaled
    scaler: sklearm method
    """
    if array.ndim == 1:
        array = array[:,None]
    if method == "minmax":
        scaler = preprocessing.MinMaxScaler()
        array_scaled =scaler.fit_transform(array)
        return array_scaled,scaler
    else:
        raise ValueError("method {} not available yet".format(method))

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

def manage_predictions(samples_dict,args,predictions_dict):
    """

    :param samples_dict: Collects the binary, logits and probabilities predicted for args.num_samples  from the posterior predictive after training
    :param args:
    :param predictions_dict: Collects the binary, logits and probabilities predicted for 1 sample from the posterior predictive during training
    :return:
    """
    binary_predictions_samples = samples_dict["binary"]
    true_labels_samples = samples_dict["true"]
    #assert  ((true_labels == true_labels_samples).sum())/true_labels.shape[0] == 1., "Review, the labels are off"
    logits_predictions_samples = samples_dict["logits"]
    probs_predictions_samples = samples_dict["probs"]

    n_data = true_labels_samples.shape[0]
    #probs_predictions_samples_true = probs_predictions_samples[np.arange(0, n_data),:, true_labels.long()]  # pick the probability of the true target for every sample
    #probs_positive_class = probs_predictions_samples[:,:, 1]  # pick the probability of the positive class for every sample

    class_logits_predictions_samples_argmax = np.argmax(logits_predictions_samples,axis=-1)
    class_logits_predictions_samples_argmax_mode = stats.mode(class_logits_predictions_samples_argmax, axis=1,keepdims=True).mode.squeeze(-1)
    binary_predictions_samples_mode = stats.mode(binary_predictions_samples, axis=1,keepdims=True).mode.squeeze(-1)

    # binary_frequencies = torch.stack([torch.bincount(x_i, minlength=args.num_classes) for i, x_i in
    #                                  enumerate(torch.unbind(binary_predictions_samples.type(torch.int64), dim=0),
    #                                            0)], dim=0)
    binary_frequencies = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.num_classes), axis=1, arr=binary_predictions_samples.astype("int64"))
    binary_frequencies = binary_frequencies / args.num_samples
    # argmax_frequencies = torch.stack([torch.bincount(x_i, minlength=args.num_classes) for i, x_i in
    #                                   enumerate(torch.unbind(class_logits_predictions_samples_argmax.type(torch.int64), dim=0),
    #                                             0)], dim=0)
    argmax_frequencies = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.num_classes), axis=1, arr=class_logits_predictions_samples_argmax.astype("int64")).T
    argmax_frequencies = argmax_frequencies / args.num_samples

    if predictions_dict is not None:
        summary_dict = {  "class_binary_predictions_samples": binary_predictions_samples,
                          "class_binary_predictions_samples_mode": binary_predictions_samples_mode,
                          "class_binary_prediction_samples_frequencies": binary_frequencies,
                          "class_logits_predictions_samples": logits_predictions_samples,
                          "class_logits_predictions_samples_argmax": class_logits_predictions_samples_argmax,
                          "class_logits_predictions_samples_argmax_frequencies": argmax_frequencies,
                          "class_logits_predictions_samples_argmax_mode": class_logits_predictions_samples_argmax_mode,
                          "class_probs_predictions_samples": probs_predictions_samples,
                          "class_probs_predictions_samples_average": np.mean(probs_predictions_samples,axis=1),
                          "class_binary_prediction_single_sample": predictions_dict["binary"],
                          "class_logits_prediction_single_sample": predictions_dict["logits"],
                          "class_logits_prediction_single_sample_argmax": np.argmax(predictions_dict["logits"],axis=-1),
                          "class_probs_prediction_single_sample_true": predictions_dict["probs"][np.arange(0,n_data),predictions_dict["true"].astype(int)],
                          "class_probs_prediction_single_sample": predictions_dict["probs"],
                          "samples_average_accuracy":samples_dict["accuracy"],
                          "true_samples": true_labels_samples,
                          "true_samples_onehot": samples_dict["true_onehot"],
                          "true_single_sample": predictions_dict["true"],
                          "true_onehot_single_sample": predictions_dict["true_onehot"],
                          "confidence_scores_samples": samples_dict["confidence_scores"],
                          "confidence_scores_single_sample": predictions_dict["confidence_scores"]
                          }
    else:
        summary_dict = {"class_binary_predictions_samples": binary_predictions_samples,
                        "class_binary_predictions_samples_mode": binary_predictions_samples_mode,
                        "class_binary_prediction_samples_frequencies": binary_frequencies,
                        "class_logits_predictions_samples": logits_predictions_samples,
                        "class_logits_predictions_samples_argmax": class_logits_predictions_samples_argmax,
                        "class_logits_predictions_samples_argmax_frequencies": argmax_frequencies,
                        "class_logits_predictions_samples_argmax_mode": class_logits_predictions_samples_argmax_mode,
                        "class_probs_predictions_samples": probs_predictions_samples,
                        "class_probs_predictions_samples_average": np.mean(probs_predictions_samples, axis=1),
                        "class_binary_prediction_single_sample": None,
                        "class_logits_prediction_single_sample": None,
                        "class_logits_prediction_single_sample_argmax": None,
                        "class_probs_prediction_single_sample_true": None,
                        "class_probs_prediction_single_sample": None,
                        "samples_average_accuracy": samples_dict["accuracy"],
                        "true_samples": true_labels_samples,
                        "true_samples_onehot": samples_dict["true_onehot"],
                        "true_single_sample": None,
                        "true_onehot_single_sample": None,
                        "confidence_scores_samples": samples_dict["confidence_scores"],
                        "confidence_scores_single_sample": None
                        }

    return summary_dict

