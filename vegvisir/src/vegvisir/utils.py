"""
=======================
2022-2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import argparse
import ast,warnings
import Bio.Align
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import os,shutil
from collections import defaultdict, Counter
import time,datetime
from sklearn import preprocessing
import pandas as pd
import torch
import scipy
import vegvisir.plots as VegvisirPlots
import vegvisir.load_utils as VegvisirLoadUtils
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing
from collections import namedtuple
from sklearn.metrics import mutual_info_score

PeptideFeatures = namedtuple("PeptideFeatures",["gravy_dict","volume_dict","radius_dict","side_chain_pka_dict","isoelectric_dict","bulkiness_dict"])
MAX_WORKERs = ( multiprocessing. cpu_count() - 1 )
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

def print_divisors(n) :
    """Calculates the number of divisors of a number
    :param int n: number"""
    i = 1
    divisors = []
    while i <= n :
        if (n % i==0) :
            divisors.append(i)
        i = i + 1
    return divisors

def folders(folder_name,basepath,overwrite=True):
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
        if overwrite:
            print("removing subdirectories") #if this is reached is because you are running the folders function twice with the same folder name
            shutil.rmtree(newpath)  # removes all the subdirectories!
            os.makedirs(newpath,0o777)
        else:
            pass

def replace_nan(x,x_unique,replace_val=0.0):
    """Detects nan values and replaces them with a given values
    :param x numpy array
    :param: x_unique numpy array of unique values from x
    """
    if np.isnan(x_unique).any():
        x = np.nan_to_num(replace_val)
        x_unique = x_unique[~np.isnan(x_unique)]
        if not np.any(x_unique == replace_val):
            np.append(x_unique,[0])
    return x,x_unique

def aminoacid_names_dict(aa_types,zero_characters = []):
    """ Returns an aminoacid associated to a integer value
    All of these values are mapped to 0:
        # means empty value/padding
        - means gap in an alignment
        * means stop codon
    :param int aa_types: amino acid probabilities, this number correlates to the number of different aa types in the input alignment
    :param list zero_characters: character(s) to be set to 0
    """
    #TODO: Can be improved, no need to assign R to 0
    if aa_types == 20 :
        assert len(zero_characters) == 0, "No zero characters allowed, please set zero_characters to empty list"
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

def aminoacids_groups(aa_dict):
    others = ([],"black",0)
    positive_charged = (["R","H","K"],"red",1)
    negative_charged = (["D","E"],"lawngreen",2)
    uncharged = (["S","T","N","Q"],"aqua",3)
    special = (["C","U","G","P"],"yellow",4)
    hydrophobic = (["A","V","I","L","M"],"orange",5)
    aromatic = (["F","Y","W"],"magenta",6)
    groups_names_colors_dict = {"positive":positive_charged[1],"negative":negative_charged[1],"uncharged":uncharged[1],"special":special[1],"hydrophobic":hydrophobic[1],"aromatic":aromatic[1],"others":others[1]}
    #aa_by_groups_dict = {"positive":positive_charged[0],"negative":negative_charged[0],"uncharged":uncharged[0],"special":special[0],"hydrophobic":hydrophobic[0],"aromatic":aromatic[0],"others":others[0]}

    aa_by_groups_dict = dict(zip( positive_charged[0] + negative_charged[0] + uncharged[0] + special[0] + hydrophobic[0] + aromatic[0],\
                         len(positive_charged[0])*["positive"] + len(negative_charged[0])*["negative"] + len(uncharged[0])*["uncharged"]  + len(special[0])*["special"] + len(hydrophobic[0])*["hydrophobic"]  + len(aromatic[0])*["aromatic"] ))

    aa_groups_colors_dict = defaultdict()
    aa_groups_dict =defaultdict()
    for aa,i in aa_dict.items():
        if aa in positive_charged[0]:
            aa_groups_colors_dict[i] = positive_charged[1]
            aa_groups_dict[i] = positive_charged[2]
        elif aa in negative_charged[0]:
            aa_groups_colors_dict[i] = negative_charged[1]
            aa_groups_dict[i] = negative_charged[2]
        elif aa in uncharged[0]:
            aa_groups_colors_dict[i] = uncharged[1]
            aa_groups_dict[i] = uncharged[2]
        elif aa in special[0]:
            aa_groups_colors_dict[i] = special[1]
            aa_groups_dict[i] = special[2]
        elif aa in hydrophobic[0]:
            aa_groups_colors_dict[i] = hydrophobic[1]
            aa_groups_dict[i] = hydrophobic[2]
        elif aa in aromatic[0]:
            aa_groups_colors_dict[i] = aromatic[1]
            aa_groups_dict[i] = aromatic[2]
        else:
            aa_groups_colors_dict[i] = others[1]
            aa_groups_dict[i] = others[2]


    #return {"aa_groups_colors_dict":aa_groups_colors_dict,"aa_groups_dict":aa_groups_dict,"groups_names_colors_dict":groups_names_colors_dict}
    return  aa_groups_colors_dict,aa_groups_dict,groups_names_colors_dict,aa_by_groups_dict

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
    :param bool include_zero_characters : If True the score for the zero characters is kept in the blosum encoding for each amino acid, so the vector will have size 21 instead of just 20
    """

    if aa_types > 21 and not subs_matrix_name.startswith("PAM"):
        warnings.warn("Your dataset contains special amino acids. Switching your substitution matrix to PAM70")
        subs_matrix_name = "PAM70"
    elif aa_types == 20 and len(zero_characters) !=0:
        raise ValueError("No zero characters allowed, please set zero_characters to empty list")

    subs_matrix = Bio.Align.substitution_matrices.load(subs_matrix_name)
    aa_list = list(aminoacid_names_dict(aa_types,zero_characters=zero_characters).keys())

    if zero_characters:
        index_gap = aa_list.index("#")
        aa_list[index_gap] = "*" #in the blosum matrix gaps are represented as *

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
    if include_zero_characters or not zero_characters:
        blosum_array_dict = dict(enumerate(subs_array[1:,1:]))
    else:
        blosum_array_dict = dict(enumerate(subs_array[1:, 2:])) # Highlight: Changed to [1:,2:] instead of [1:,1:] to skip the scores for non-aa elements

    #blosum_array_dict[0] = np.full((aa_types),0)  #np.nan == np.nan is False ...

    return subs_array, subs_dict, blosum_array_dict

def calculate_aa_frequencies(dataset,freq_bins):
    """Calculates a frequency for each of the aa & gap at each position.The number of bins (of size 1) is one larger than the largest value in x. This is done for numpy arrays
    :param tensor dataset
    :param int freq_bins
    """
    if isinstance(dataset,np.ndarray):
        freqs = np.apply_along_axis(lambda x: np.bincount(x, minlength=freq_bins), axis=0, arr=dataset.astype("int64")).T
        freqs = freqs/dataset.shape[0]
        return freqs
    elif isinstance(dataset,torch.Tensor):
        freqs = torch.stack([torch.bincount(x_i, minlength=freq_bins) for i, x_i in
                             enumerate(torch.unbind(dataset.type(torch.int64), dim=1), 0)], dim=1)
        freqs = freqs.T
        freqs = freqs / dataset.shape[0]
        return freqs
    else:
        print("Data type not supported for bincount")

def process_blosum(blosum,aa_freqs,align_seq_len,aa_probs):
    """
    Computes the matrices required to build a blosum embedding
    :param tensor blosum: BLOSUM likelihood  scores
    :param tensor aa_freqs : amino acid frequencies per position
    :param align_seq_len: alignment length
    :param aa_probs: amino acid probabilities, types of amino acids in the alignment
    :out tensor blosum_max [align_len,aa_prob]: blosum likelihood scores for the most frequent aa in the alignment position
    :out tensor blosum_weighted [align_len,aa_prob: weighted average of blosum likelihoods according to the aa frequency
    :out variable_core: [align_len] : counts the number of different elements (amino acid diversity) per alignment position"""

    if isinstance(aa_freqs,np.ndarray):
        aa_freqs = torch.from_numpy(aa_freqs)
    if isinstance(blosum,np.ndarray):
        blosum = torch.from_numpy(blosum)

    aa_freqs_max = torch.argmax(aa_freqs, dim=1).repeat(aa_probs, 1).permute(1, 0) #[max_len, aa_probs]
    blosum_expanded = blosum[1:, 1:].repeat(align_seq_len, 1, 1)  # [max_len,aa_probs,aa_probs]
    blosum_max = blosum_expanded.gather(1, aa_freqs_max.unsqueeze(1)).squeeze(1)  # [align_seq_len,21] Seems correct

    blosum_weighted = aa_freqs[:,:,None]*blosum_expanded #--> replace 0 with nans? otherwise the 0 are in the mean as well....
    blosum_weighted = blosum_weighted.mean(dim=1)

    variable_score = torch.count_nonzero(aa_freqs, dim=1)/aa_probs #higher score, more variable

    return blosum_max,blosum_weighted, variable_score

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
        p1 = np.linalg.norm(a)
        #p1 = np.sqrt(np.sum(a**2))
        p2 = np.linalg.norm(b)
        #p2 = np.sqrt(np.sum(b**2))
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

def view1D(a): # a is array #TODO: Remove or investigate, is supposed to speed the comparisons up
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel()

def calculate_similarity_matrix_slow(array, max_len, array_mask, batch_size=200, ksize=3): #TODO: Remove
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
            # Highlight: Apply masks to calculate the similarities_old. NOTE: To get the data with the filled value use k = np.ma.getdata(kmers_matrix_diag_masked)
            ##PERCENT IDENTITY (binary pairwise comparison) ###############
            percent_identity_mean_ij = np.ma.masked_array(pairwise_sim_j, mask=~pid_mask_ij, fill_value=0.).mean(-1)  # Highlight: In the mask if True means to mask and ignore!!!!
            percent_identity_mean_i[:,start_store_point_i:end_store_point_i] = percent_identity_mean_ij #TODO: Probably no need to store this either
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
        kmers_pid_similarity), np.ma.getdata(kmers_cosine_similarity)#TO#TTODO: remove

def calculate_similarity_matrix(array, max_len, array_mask, batch_size=200, ksize=3):#TODO: Remove
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
        start_store_point_i = 0 + store_point_helper #TODO: Why the + 0? Typo?
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
            ##PERCENT IDENTITY (binary pairwise comparison) ###############
            percent_identity_mean_ij = np.ma.masked_array(pairwise_sim_j, mask=~pid_mask_ij, fill_value=0.).mean(-1)  # Highlight: In the mask if True means to mask and ignore!!!! #TODO: does it?
            percent_identity_mean_i[:,start_store_point_i:end_store_point_i] = percent_identity_mean_ij #TODO: Probably no need to store this either
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

def manage_predictions_generative(args,generative_dict):

    mode = "samples" if args.num_samples > 1 else "single_sample"
    class_logits_predictions_generative_argmax = np.argmax(generative_dict["logits"], axis=-1)
    #class_logits_predictions_generative_argmax_mode = stats.mode(class_logits_predictions_generative_argmax, axis=1,keepdims=True).mode.squeeze(-1)
    class_logits_predictions_generative_argmax_mode = class_logits_predictions_generative_argmax
    probs_predictions_generative = generative_dict["probs"]

    binary_frequencies = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.num_classes), axis=1, arr=generative_dict["binary"].astype("int64"))
    binary_frequencies = binary_frequencies / args.num_samples


    #Highlight: Stack 2 data_int to maintain the latter format

    data_int = np.concatenate([generative_dict["data_int"][:,None],generative_dict["data_int"][:,None]],axis=1)

    # Highlight: Calculate the 90% confidence interval
    num_samples_generated = probs_predictions_generative.shape[1]
    lower_bound = 1 if int(num_samples_generated * 0.05) == 0 else int(num_samples_generated * 0.05)
    upper_bound = num_samples_generated - 1 if int(num_samples_generated * 0.95) >= num_samples_generated else int(num_samples_generated * 0.95)
    # probs_5_class_0 = probs_predictions_generative[:, :, 0].kthvalue(lower_bound, dim=1)[0]
    # probs_95_class_0 = probs_predictions_generative[:, :, 0].kthvalue(upper_bound, dim=1)[0]
    # probs_5_class_1 = probs_predictions_generative[:, :, 1].kthvalue(lower_bound, dim=1)[0]
    # probs_95_class_1 = probs_predictions_generative[:, :, 1].kthvalue(upper_bound, dim=1)[0]

    probs_5_class_0 = np.partition(probs_predictions_generative[:,:,0],lower_bound, axis=1)[:,lower_bound]
    probs_95_class_0 = np.partition(probs_predictions_generative[:,:,0],upper_bound, axis=1)[:, upper_bound]
    probs_5_class_1 = np.partition(probs_predictions_generative[:,:,1],lower_bound, axis=1)[:,lower_bound]
    probs_95_class_1 = np.partition(probs_predictions_generative[:,:,1],upper_bound, axis=1)[:,upper_bound]

    generative_dict = {
                        "data_int_single_sample": data_int,
                        "data_int_samples": data_int,
                        "data_mask_single_sample": generative_dict["data_mask"],
                        "data_mask_samples": generative_dict["data_mask"],
                        "class_binary_predictions_{}".format(mode): generative_dict["binary"] if args.num_samples > 1 else generative_dict["binary"].squeeze(1) ,
                        "true_single_sample": generative_dict["true"],
                        "true_samples": generative_dict["true"],
                        "class_binary_predictions_{}_mode".format(mode): stats.mode(generative_dict["binary"], axis=1,keepdims=True).mode.squeeze(-1),
                        "class_binary_predictions_samples_frequencies": binary_frequencies, #I name it samples to avoid errors
                        "class_logits_predictions_single_sample": generative_dict["logits"],
                        "class_logits_predictions_samples": generative_dict["logits"],
                        "class_logits_predictions_single_sample_argmax": class_logits_predictions_generative_argmax,
                        "class_logits_predictions_samples_argmax": class_logits_predictions_generative_argmax,
                        "class_logits_predictions_single_sample_argmax_frequencies".format(mode): None,
                        "class_logits_predictions_samples_argmax_frequencies".format(mode): None,
                        "class_logits_predictions_single_sample_argmax_mode": class_logits_predictions_generative_argmax_mode,
                        "class_logits_predictions_samples_argmax_mode": class_logits_predictions_generative_argmax_mode,
                        "class_probs_predictions_samples_5%CI_class_0": probs_5_class_0,
                        "class_probs_predictions_samples_95%CI_class_0": probs_95_class_0,
                        "class_probs_predictions_samples_5%CI_class_1": probs_5_class_1,
                        "class_probs_predictions_samples_95%CI_class_1": probs_95_class_1,
                        "class_probs_predictions_single_sample": probs_predictions_generative,
                        "class_probs_predictions_samples": probs_predictions_generative,
                        #"class_probs_predictions_{}_average".format(mode): np.mean(probs_predictions_generative, axis=1) if args.num_samples > 1 else probs_predictions_generative,
                        "class_probs_predictions_{}_average".format(mode): probs_predictions_generative,
                        #"class_binary_predictions_{}_logits_average_argmax".format(mode): np.argmax(np.mean(probs_predictions_generative, axis=1), axis=1) if args.num_samples > 1 else np.argmax(probs_predictions_generative, axis=1)
                        "class_binary_predictions_{}_logits_average_argmax".format(mode): np.argmax(probs_predictions_generative, axis=1)
                       }


    return generative_dict

def manage_predictions(samples_dict,args,predictions_dict, generative_dict=None):
    """

    :param samples_dict: Collects the binary, logits and probabilities predicted for args.num_samples  from the posterior predictive after training
    :param NamedTuple args:
    :param predictions_dict: Collects the binary, logits and probabilities predicted for 1 sample ("single sample") from the posterior predictive during training
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
    
    #Highlight: Calculate the 90% confidence interval

    lower_bound = 1 if int(args.num_samples * 0.05) == 0 else int(args.num_samples * 0.05)
    upper_bound = args.num_samples - 1 if int(args.num_samples* 0.95) >= args.num_samples else int(args.num_samples * 0.95)
    
    # probs_5_class_0 = probs_predictions_samples[:,:,0].kthvalue(lower_bound, dim=1)[0]
    # probs_95_class_0 = probs_predictions_samples[:,:,0].kthvalue(upper_bound, dim=1)[0]
    # probs_5_class_1 = probs_predictions_samples[:,:,1].kthvalue(lower_bound, dim=1)[0]
    # probs_95_class_1 = probs_predictions_samples[:,:,1].kthvalue(upper_bound, dim=1)[0]

    probs_5_class_0 = np.partition(probs_predictions_samples[:,:,0],lower_bound, axis=1)[:,lower_bound]
    probs_95_class_0 = np.partition(probs_predictions_samples[:,:,0],upper_bound, axis=1)[:,upper_bound]
    probs_5_class_1 = np.partition(probs_predictions_samples[:,:,1],lower_bound, axis=1)[:,lower_bound]
    probs_95_class_1 = np.partition(probs_predictions_samples[:,:,1],upper_bound, axis=1)[:,upper_bound]
    

    if predictions_dict is not None:
        summary_dict = {"data_int_single_sample":predictions_dict["data_int"],
                        "data_int_samples": samples_dict["data_int"],
                        "data_mask_single_sample": predictions_dict["data_mask"],
                        "data_mask_samples": samples_dict["data_mask"],
                        "class_binary_predictions_samples": binary_predictions_samples,
                        "class_binary_predictions_samples_mode": binary_predictions_samples_mode,
                        "class_binary_predictions_samples_frequencies": binary_frequencies,
                        "class_logits_predictions_samples": logits_predictions_samples,
                        "class_logits_predictions_samples_argmax": class_logits_predictions_samples_argmax,
                        "class_logits_predictions_samples_argmax_frequencies": argmax_frequencies,
                        "class_logits_predictions_samples_argmax_mode": class_logits_predictions_samples_argmax_mode,
                        "class_probs_predictions_samples": probs_predictions_samples,
                        "class_probs_predictions_samples_average": np.mean(probs_predictions_samples,axis=1),
                        "class_probs_predictions_samples_5%CI_class_0": probs_5_class_0,
                        "class_probs_predictions_samples_95%CI_class_0": probs_95_class_0,
                        "class_probs_predictions_samples_5%CI_class_1": probs_5_class_1,
                        "class_probs_predictions_samples_95%CI_class_1": probs_95_class_1,
                        "class_binary_predictions_samples_logits_average_argmax": np.argmax(np.mean(probs_predictions_samples,axis=1),axis=1),
                        "class_binary_predictions_single_sample": predictions_dict["binary"],
                        "class_logits_predictions_single_sample": predictions_dict["logits"],
                        "class_logits_predictions_single_sample_argmax": np.argmax(predictions_dict["logits"],axis=-1),
                        "class_probs_predictions_single_sample_true": predictions_dict["probs"][np.arange(0,n_data),predictions_dict["observed"].astype(int)],
                        "class_probs_predictions_single_sample": predictions_dict["probs"],
                        "samples_average_accuracy":samples_dict["accuracy"],
                        "true_samples": true_labels_samples,
                        "true_onehot_samples": samples_dict["true_onehot"],
                        "true_single_sample": predictions_dict["true"],
                        "true_onehot_single_sample": predictions_dict["true_onehot"],
                        "confidence_scores_samples": samples_dict["confidence_scores"],
                        "confidence_scores_single_sample": predictions_dict["confidence_scores"],
                        "training_assignation_samples": samples_dict["training_assignation"],
                        "training_assignation_single_sample": predictions_dict["training_assignation"],
                        "attention_weights_single_sample":predictions_dict["attention_weights"],
                        "attention_weights_samples": samples_dict["attention_weights"],
                        "encoder_hidden_states_single_sample":predictions_dict["encoder_hidden_states"],
                        "encoder_hidden_states_samples":samples_dict["encoder_hidden_states"],
                        "decoder_hidden_states_single_sample": predictions_dict["decoder_hidden_states"],
                        "decoder_hidden_states_samples": samples_dict["decoder_hidden_states"],
                        "encoder_final_hidden_state_single_sample": predictions_dict["encoder_final_hidden_state"],
                        "encoder_final_hidden_state_samples": samples_dict["encoder_final_hidden_state"],
                        "decoder_final_hidden_state_single_sample": predictions_dict["decoder_final_hidden_state"],
                        "decoder_final_hidden_state_samples": samples_dict["decoder_final_hidden_state"],

                        }
    else:
        summary_dict = {"data_int_single_sample":None,
                        "data_int_samples": samples_dict["data_int"],
                        "data_mask_single_sample": None,
                        "data_mask_samples": samples_dict["data_mask"],
                        "class_binary_predictions_samples": binary_predictions_samples,
                        "class_binary_predictions_samples_mode": binary_predictions_samples_mode,
                        "class_binary_predictions_samples_frequencies": binary_frequencies,
                        "class_logits_predictions_samples": logits_predictions_samples,
                        "class_logits_predictions_samples_argmax": class_logits_predictions_samples_argmax,
                        "class_logits_predictions_samples_argmax_frequencies": argmax_frequencies,
                        "class_logits_predictions_samples_argmax_mode": class_logits_predictions_samples_argmax_mode,
                        "class_probs_predictionss_samples": probs_predictions_samples,
                        "class_probs_predictions_samples_average": np.mean(probs_predictions_samples, axis=1),
                        "class_probs_predictions_samples_5%CI_class_0": probs_5_class_0,
                        "class_probs_predictions_samples_95%CI_class_0": probs_95_class_0,
                        "class_probs_predictions_samples_5%CI_class_1": probs_5_class_1,
                        "class_probs_predictions_samples_95%CI_class_1": probs_95_class_1,
                        "class_binary_predictions_samples_logits_average_argmax": np.argmax(np.mean(probs_predictions_samples, axis=1),axis=1),
                        "class_binary_predictions_single_sample": None,
                        "class_logits_predictions_single_sample": None,
                        "class_logits_predictions_single_sample_argmax": None,
                        "class_probs_predictions_single_sample_true": None,
                        "class_probs_predictions_single_sample": None,
                        "samples_average_accuracy": samples_dict["accuracy"],
                        "true_samples": true_labels_samples,
                        "true_onehot_samples": samples_dict["true_onehot"],
                        "true_single_sample": None,
                        "true_onehot_single_sample": None,
                        "confidence_scores_samples": samples_dict["confidence_scores"],
                        "confidence_scores_single_sample": None,
                        "training_assignation_samples": samples_dict["training_assignation"],
                        "training_assignation_single_sample": None,
                        "attention_weights_single_sample": None,
                        "attention_weights_samples": samples_dict["attention_weights"],
                        "encoder_hidden_states_single_sample": None,
                        "encoder_hidden_states_samples": samples_dict["encoder_hidden_states"],
                        "decoder_hidden_states_single_sample": None,
                        "decoder_hidden_states_samples": samples_dict["decoder_hidden_states"],
                        "encoder_final_hidden_state_single_sample": None,
                        "encoder_final_hidden_state_samples": samples_dict["encoder_final_hidden_state"],
                        "decoder_final_hidden_state_single_sample": None,
                        "decoder_final_hidden_state_samples": samples_dict["decoder_final_hidden_state"],
                        }

    return summary_dict

def save_results_table(predictions_dict,latent_space, args,dataset_info,results_dir,method="Train",merge_netmhc=False,save_df=True):
    """
    """


    class_probabilities = predictions_dict["class_probs_predictions_samples_average"]
    onehot_labels = predictions_dict["true_onehot_samples"]

    probs_5_class_0 = predictions_dict["class_probs_predictions_samples_5%CI_class_0"]
    probs_95_class_0 = predictions_dict["class_probs_predictions_samples_95%CI_class_0"]

    probs_5_class_1 = predictions_dict["class_probs_predictions_samples_5%CI_class_1"]
    probs_95_class_1 = predictions_dict["class_probs_predictions_samples_95%CI_class_1"]

    data_int = predictions_dict["data_int_samples"]
    sequences = data_int[:, 1:].squeeze(1)
    # if dataset_info.corrected_aa_types == 20:
    #     sequences_mask = np.zeros_like(sequences).astype(bool)
    # else:
    #     sequences_mask = np.array((sequences == 0))

    #sequences_lens = np.sum(~sequences_mask, axis=1)

    #Highlight: Load pre-computed features dictionaries
    custom_features_dicts = build_features_dicts(dataset_info)
    aminoacids_dict_reversed = custom_features_dicts["aminoacids_dict_reversed"]
    volume_dict = custom_features_dicts["volume_dict"]
    radius_dict = custom_features_dicts["radius_dict"]
    side_chain_pka_dict = custom_features_dicts["side_chain_pka_dict"]
    bulkiness_dict = custom_features_dicts["bulkiness_dict"]

    sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(sequences)

    sequences_list = list(map(lambda seq: "{}".format("".join(seq).replace("#","-")), sequences_raw.tolist()))
    if dataset_info.corrected_aa_types == 20:
        sequences_mask=np.zeros_like(sequences).astype(bool)
    else:
        sequences_mask = np.array((sequences == 0))

    bulkiness_scores = np.vectorize(bulkiness_dict.get)(sequences)
    bulkiness_scores = np.ma.masked_array(bulkiness_scores, mask=sequences_mask, fill_value=0)
    bulkiness_scores = np.ma.sum(bulkiness_scores, axis=1)

    volume_scores = np.vectorize(volume_dict.get)(sequences)
    volume_scores = np.ma.masked_array(volume_scores, mask=sequences_mask, fill_value=0)
    volume_scores = np.ma.sum(volume_scores, axis=1)  # TODO: Mean? or sum?

    radius_scores = np.vectorize(radius_dict.get)(sequences)
    radius_scores = np.ma.masked_array(radius_scores, mask=sequences_mask, fill_value=0)
    radius_scores = np.ma.sum(radius_scores, axis=1)

    side_chain_pka_scores = np.vectorize(side_chain_pka_dict.get)(sequences)
    side_chain_pka_scores = np.ma.masked_array(side_chain_pka_scores, mask=sequences_mask, fill_value=0)
    side_chain_pka_scores = np.ma.mean(side_chain_pka_scores, axis=1)  # Highlight: before I was doing just the sum

    sequences_list = list(map(lambda seq:seq.replace("-",""),sequences_list))

    isoelectric_scores = np.array(list(map(lambda seq: calculate_isoelectric(seq), sequences_list)))
    aromaticity_scores = np.array(list(map(lambda seq: calculate_aromaticity(seq), sequences_list)))
    gravy_scores = np.array(list(map(lambda seq: calculate_gravy(seq), sequences_list)))
    molecular_weight_scores = np.array(list(map(lambda seq: calculate_molecular_weight(seq), sequences_list)))
    extintion_coefficient_scores_cysteines = np.array(list(map(lambda seq: calculate_extintioncoefficient(seq)[0], sequences_list)))
    extintion_coefficient_scores_cystines = np.array(list(map(lambda seq: calculate_extintioncoefficient(seq)[1], sequences_list)))

    results_df = pd.DataFrame({"Icore":sequences_list,
                       "Latent_vector":latent_space[:,6:].tolist(),
                       "Target_corrected":latent_space[:,0].tolist(),
                       "Immunoprevalence":latent_space[:,3].tolist(),
                       "Bulkiness_score":bulkiness_scores,
                       "Volume_score":volume_scores,
                       "Radius_score":radius_scores,
                       "Side_chain_pka":side_chain_pka_scores,
                       "Isoelectric_scores":isoelectric_scores,
                       "Aromaticity_scores":aromaticity_scores,
                       "Gravy_scores":gravy_scores,
                       "Molecular_weight_scores":molecular_weight_scores,
                       "Extintion_coefficient_scores(Cysteines)":extintion_coefficient_scores_cysteines,
                       "Extintion_coefficient_scores(Cystines)":extintion_coefficient_scores_cystines,
                       "Vegvisir_negative_prob":class_probabilities[:,0].tolist(),
                       "Vegvisir_positive_prob":class_probabilities[:,1].tolist(),
                       "Vegvisir_negative_5CI":probs_5_class_0.tolist(),
                       "Vegvisir_negative_95CI":probs_95_class_0.tolist(),
                       "Vegvisir_positive_5CI": probs_5_class_1.tolist(),
                       "Vegvisir_positive_95CI": probs_95_class_1.tolist(),
                       })


    if merge_netmhc and args.dataset_name in ["viral_dataset9","viral_dataset14"]:
        storage_folder = custom_features_dicts["storage_folder"]
        train_data_info = pd.read_csv("{}/common_files/Viruses_db_partitions_notest.tsv".format(storage_folder), sep="\t")
        train_data_info = train_data_info[["Icore","allele","Rnk_EL"]]
        test_data_info = pd.read_csv("{}/common_files/new_test_nonanchor_immunodominance.csv".format(storage_folder),sep=",")
        test_data_info = test_data_info[["Icore","allele","Rnk_EL"]]
        #data_info = pd.concat([train_data_info,test_data_info],axis=0)
        data_info = pd.merge(train_data_info, test_data_info, on=['Icore'], how='outer',suffixes=('_a', '_b'))
        data_info["allele"] = data_info["allele_a"].fillna(data_info["allele_b"])
        data_info["Rnk_EL"] = data_info["Rnk_EL_a"].fillna(data_info["Rnk_EL_b"])
        data_info.drop(["allele_a","allele_b","Rnk_EL_a","Rnk_EL_b"],axis=1,inplace=True)

        data_info = data_info.groupby('Icore', as_index=False)[["allele","Rnk_EL"]].agg(list) #aggregate as a list --> contains more Icore sequences than used for training & testing, because seqs with no patients were discarded
        results_df = results_df.merge(data_info,how="left",on="Icore")

    if save_df:
        results_df.to_csv("{}/{}/Epitopes_predictions_{}.tsv".format(results_dir,method,method),sep="\t",index=False)

    return results_df

def extract_group_old_test(train_summary_dict,valid_summary_dict,args):
    """"""
    test_train_summary_dict = defaultdict() #old test data points localizados en el train dataset
    test_valid_summary_dict = defaultdict() #old test data points localizados en el train dataset
    test_all_summary_dict = defaultdict()

    for train_key,valid_key in zip(train_summary_dict,valid_summary_dict):
        train_val = train_summary_dict[train_key]
        valid_val = valid_summary_dict[valid_key]
        for sample_mode in ["single_sample","samples"]:
            train_training_assignation = np.invert(train_summary_dict["training_assignation_{}".format(sample_mode)].astype(bool))
            valid_training_assignation = np.invert(valid_summary_dict["training_assignation_{}".format(sample_mode)].astype(bool))
            train_ndata = train_training_assignation.shape[0]
            valid_ndata = valid_training_assignation.shape[0]

            if sample_mode in train_key:
                if train_val is not None:
                    if train_val.ndim != 0:
                        if train_val.shape[0] == train_ndata:
                            test_train_summary_dict[train_key] = train_val[train_training_assignation]
                            test_valid_summary_dict[train_key] = valid_val[valid_training_assignation]
                            test_all_summary_dict[train_key] = np.concatenate([train_val[train_training_assignation],valid_val[valid_training_assignation]],axis=0)
                        else:
                            test_train_summary_dict[train_key] = train_val[:,train_training_assignation]
                            test_valid_summary_dict[train_key] = valid_val[:,valid_training_assignation]
                            test_all_summary_dict[train_key] = np.concatenate(
                                [train_val[:,train_training_assignation], valid_val[:,valid_training_assignation]], axis=1)
                else:
                    test_train_summary_dict[train_key] = train_val
                    test_valid_summary_dict[train_key] = valid_val
                    test_all_summary_dict[train_key] = None

    return test_train_summary_dict,test_valid_summary_dict,test_all_summary_dict

def information_shift(arr,arr_mask,diag_idx_maxlen,max_len):
    """
    Assuming that the RNN hidden states are obtained as stated here: https://github.com/pytorch/pytorch/issues/3587
    Calculates the amount of vector similarity/distance change between the hidden representations of the positions in the sequence for both backward and forward RNN hidden states.
    1) For a given sequence with 2 sequences of hidden states [2,L,Hidden_dim]

        A) Calculate the cosine similarities between all the hidden states of the forward and backward networks of an RNN
        Forward = Cos_sim([Hidden_states[0],Hidden_states[0]]
        Backward = Cos_sim([Hidden_states[1],Hidden_states[1]]

        b) Retrieve from the cosine similarity matrix the offset +1 diagonal which contains the following information about contigous states

        Forward states:        [0->1][1->2][2->3][3->4]
        Backward states: [0<-1][1<-2][2<-3][3<-4]
        ------------------------------------

    2) Make the average among the information gains of the forward and backward states (overlapping)
        Pos 0 : [0<-1]
        Pos 1 : ([0->1] + [1<-2])/2
        Pos 2 : ([1->2] + [2<-3])/2
        Pos 3 : ([2->3] + [3<-4])/2
        Pos 4 : [3->4]


    :param arr: Hidden states of one sequence
    :param arr_mask: Boolean indicating the paddings in the sequence
    :param diag_idx_maxlen:
    :param max_len:
    :return:
    """
    forward = None
    backward = None
    for idx in [0,1]:#0 is the forward state, 1 is the backward
        cos_sim_arr = cosine_similarity(arr[idx],arr[idx],correlation_matrix=False) #cosine similarity among all the vectors in the <forward/backward> hidden states
        cos_sim_diag = cos_sim_arr[diag_idx_maxlen[0][:-1],diag_idx_maxlen[1][1:]] #k=1 offset diagonal
        #Highlight: ignore the positions that have paddings
        n_paddings = (arr_mask.shape[0] - arr_mask.sum()) # max_len - true_len
        keep = cos_sim_diag.shape[0] - n_paddings #number of elements in the offset diagonal - number of "False" or paddings along the sequence
        if keep <= 0: #when all the sequence is made of paddings or only one position is not a padding, every position gets value 0
            if idx == 0:
                forward = np.zeros((max_len-1))
            else:
                backward = np.zeros((max_len-1))
        else:
            information_shift = 1-cos_sim_diag[:keep] #or cosine distance , the cosine distancevaries between 0 and 2
            #information_shift = np.abs(cos_sim_diag[:-1] -cos_sim_diag[1:])
            #Highlight: Set to 0 the information gain in the padding positions
            information_shift = np.concatenate([information_shift,np.zeros((n_paddings,))])
            if idx == 0:
                forward = information_shift
            else:
                backward = information_shift
            assert information_shift.shape[0] == max_len-1

    #Highlight: Make the arrays overlap
    forward = np.insert(forward,obj=0,values=0,axis=0)

    backward = np.append(backward,np.zeros((1,)),axis=0)
    weights = np.concatenate([forward[None,:],backward[None,:]],axis=0)
    weights = np.mean(weights,axis=0)
    #weights = np.exp(weights - np.max(weights)) / np.exp(weights - np.max(weights)).sum() #softmax
    #Highlight: Minmax scaling
    weights = (weights - weights.min()) / (weights - weights.min()).sum()
    weights*= arr_mask
    return weights[None,:]

def information_shift_samples(hidden_states,data_mask_seq,diag_idx_maxlen,seq_max_len):
    # Highlight: Encoder
    encoder_information_shift_weights_seq = Parallel(n_jobs=MAX_WORKERs)(
        delayed(information_shift)(seq, seq_mask, diag_idx_maxlen,
                                                seq_max_len) for seq, seq_mask in
        zip(hidden_states, data_mask_seq))
    encoder_information_shift_weights_sample = np.concatenate(encoder_information_shift_weights_seq, axis=0)

    return encoder_information_shift_weights_sample[:, None]

def compute_sites_entropies(logits, node_names):
    """
    Calculate the Shannon entropy of a sequence
    :param tensor logits = [n_seq, L, 21]
    :param tensor node_names: tensor with the nodes tree level order indexes ("names")
    observed = [n_seq,L]
    Pick the aa with the highest logit,
    logits = log(prob/1-prob)
    prob = exp(logit)/(1+exp(logit))
    entropy = prob.log(prob) per position in the sequence
    The entropy H is maximal when each of the symbols in the position has equal probability
    The entropy H is minimal when one of the symbols has probability 1 and the rest 0. H = 0"""
    #probs = torch.exp(logits)  # torch.sum(probs,dim=2) returns 1's so , it's correct

    prob = np.exp(logits) / (1 + np.exp(logits))
    seq_entropies = -np.sum(prob*np.log(prob),axis=2)

    seq_entropies = np.concatenate((node_names[:,None],seq_entropies),axis=1)
    return seq_entropies

def convert_to_pandas_dataframe(epitopes_padded,data,storage_folder,args,use_test=True):
    """"""
    epitopes_padded = list(map(''.join, epitopes_padded))
    data["Icore"] = epitopes_padded
    data["Icore"] = data["Icore"].str.replace('#','')

    column_names = ["Icore","target_corrected","partition"]
    shift_proportions =False
    if use_test:
        data_train = data[data["training"] == True][column_names]
        data_test = data[data["training"] == False][column_names]

        labels_counts = data_test["target_corrected"].value_counts()
        n_positives = labels_counts[1.0]
        n_negatives = labels_counts[0.0]
        positives_proportion = (n_positives * 100)/data_test.shape[0]
        negatives_proportion = (n_negatives * 100)/data_test.shape[0]

        if shift_proportions:
            VegvisirLoadUtils.redefine_class_proportions(data_test,n_positives,n_negatives,positives_proportion,negatives_proportion,drop="positives")
            shifted = "shifted_proportions"
        else:
            shifted = ""

        data_train = data_train.astype({'partition': 'int'})
        data_test.drop("partition", axis=1, inplace=True)
        data_train["Icore"].to_csv("{}/{}/viral_seq2logo.tsv".format(storage_folder, args.dataset_name), sep="\t",
                                   index=False, header=None)
        shuffled = ["shuffled" if args.shuffle_sequence else "non_shuffled"][0]
        data_train.to_csv("{}/{}/viral_nnalign_input_train_{}.tsv".format(storage_folder, args.dataset_name,shuffled), sep="\t",
                          index=False, header=None)
        data_test.to_csv("{}/{}/viral_nnalign_input_valid_{}_{}.tsv".format(storage_folder, args.dataset_name,shuffled,shifted), sep="\t",
                          index=False, header=None)  # TODO: Header None?

    else:
        data = data[data["training"] == True]
        data_train = data[data["partition"] != 4][column_names]
        data_valid = data[data["partition"] == 4][column_names]

        labels_counts = data_valid["target_corrected"].value_counts()
        n_positives = labels_counts[1.0]
        n_negatives = labels_counts[0.0]
        positives_proportion = (n_positives * 100)/data_valid.shape[0]
        negatives_proportion = (n_negatives * 100)/data_valid.shape[0]

        if shift_proportions:
            VegvisirLoadUtils.redefine_class_proportions(data_valid,n_positives,n_negatives,positives_proportion,negatives_proportion,drop="negatives")
            shifted = "shifted_proportions"
        else:
            shifted = ""
        data_train = data_train.astype({'partition': 'int'})
        data_valid.drop("partition", axis=1, inplace=True)
        data_train["Icore"].to_csv("{}/{}/viral_seq2logo.tsv".format(storage_folder, args.dataset_name), sep="\t",
                                   index=False, header=None)
        shuffled = ["shuffled" if args.shuffle_sequence else "non_shuffled"][0]
        data_train.to_csv("{}/{}/viral_nnalign_input_train_{}_no_test.tsv".format(storage_folder, args.dataset_name,shuffled),
                          sep="\t",
                          index=False, header=None)
        data_valid.to_csv("{}/{}/viral_nnalign_input_valid_{}_no_test_partition_4_{}.tsv".format(storage_folder, args.dataset_name,shuffled,shifted),
                          sep="\t",
                          index=False, header=None)  # TODO: Header None?

def calculate_isoelectric(seq):
    seq = "".join(seq).replace("#","").replace("-","")
    if seq:
        isoelectric = IP(seq).pi()
    else:
        isoelectric = 0
    return isoelectric

def calculate_molecular_weight(seq):
    seq = "".join(seq).replace("#","").replace("-","")
    if seq:
        molecular_weight = ProteinAnalysis(seq).molecular_weight()
    else:
        molecular_weight = 0
    return molecular_weight

def calculate_aromaticity(seq):
    seq = "".join(seq).replace("#","").replace("-","")
    if seq:
        aromaticity = ProteinAnalysis(seq).aromaticity()
    else:
        aromaticity = 0
    return aromaticity

def calculate_gravy(seq):
    "GRAVY (grand average of hydropathy)"
    seq = "".join(seq).replace("#","").replace("-","")
    if seq:
        gravy = ProteinAnalysis(seq).gravy()
    else:
        gravy = 0
    return gravy

def calculate_extintioncoefficient(seq):
    """Calculates the molar extinction coefficient assuming cysteines (reduced) and cystines residues (Cys-Cys-bond)
    :param str seq"""
    seq = "".join(seq).replace("#","").replace("-","")
    if seq:
        excoef_cysteines, excoef_cystines = ProteinAnalysis(seq).molar_extinction_coefficient()
    else:
        excoef_cysteines, excoef_cystines = 0,0
    return excoef_cysteines,excoef_cystines


def aa_dict_1letter_full():
    amino_acid_dict = {
        'A': 'Alanine',
        'C': 'Cysteine',
        'D': 'Aspartic Acid',
        'E': 'Glutamic Acid',
        'F': 'Phenylalanine',
        'G': 'Glycine',
        'H': 'Histidine',
        'I': 'Isoleucine',
        'K': 'Lysine',
        'L': 'Leucine',
        'M': 'Methionine',
        'N': 'Asparagine',
        'P': 'Proline',
        'Q': 'Glutamine',
        'R': 'Arginine',
        'S': 'Serine',
        'T': 'Threonine',
        'V': 'Valine',
        'W': 'Tryptophan',
        'Y': 'Tyrosine'
    }
    return  amino_acid_dict


class CalculatePeptideFeatures(object):
    """Properties table (radius etc) from https://www.researchgate.net/publication/15556561_Global_Fold_Determination_from_a_Small_Number_of_Distance_Restraints"""
    def __init__(self,seq_max_len,list_sequences,storage_folder,return_aa_freqs=False,only_w=True):
        self.storage_folder = storage_folder
        self.seq_max_len = seq_max_len
        self.aminoacid_properties = pd.read_csv("{}/aminoacid_properties.txt".format(storage_folder),sep = "\s+")
        self.list_sequences = list_sequences
        self.return_aa_freqs = return_aa_freqs
        self.only_w = only_w
        self.corrected_aa_types = len(set().union(*self.list_sequences))
        self.aminoacids_dict = aminoacid_names_dict(self.corrected_aa_types)
        self.aminoacids_list = list(self.aminoacids_dict.keys())
        self.gravy_dict = dict(zip(self.aminoacid_properties["1letter"].values.tolist(),self.aminoacid_properties["Hydropathy_index"].values.tolist()))
        self.volume_dict = dict(zip(self.aminoacid_properties["1letter"].values.tolist(), self.aminoacid_properties["Volume(A3)"].values.tolist()))
        self.radius_dict = dict(zip(self.aminoacid_properties["1letter"].values.tolist(), self.aminoacid_properties["Radius"].values.tolist()))
        self.side_chain_pka_dict = dict(zip(self.aminoacid_properties["1letter"].values.tolist(),self.aminoacid_properties["side_chain_pka"].values.tolist()))
        self.isoelectric_dict = dict(zip(self.aminoacid_properties["1letter"].values.tolist(),self.aminoacid_properties["isoelectric_point"].values.tolist()))
        self.bulkiness_dict = dict(zip(self.aminoacid_properties["1letter"].values.tolist(), self.aminoacid_properties["bulkiness"].values.tolist()))
    def return_dicts(self):

        return PeptideFeatures(gravy_dict=self.gravy_dict,
                               volume_dict = self.volume_dict,
                               radius_dict= self.radius_dict,
                               side_chain_pka_dict=self.side_chain_pka_dict,
                               isoelectric_dict= self.isoelectric_dict,
                               bulkiness_dict=self.bulkiness_dict)

    def calculate_volumetrics(self,seq,seq_max_len):
        """Calculates molecular weight, volume, radius of each residue in a protein sequence"""
        seq = "".join(seq).replace("#", "")

        pads = [0] *(seq_max_len-len(seq))
        molecular_weight = list( map(lambda aa: ProteinAnalysis(aa).molecular_weight(), list(seq))) + pads
        volume = list( map(lambda aa: self.volume_dict[aa], list(seq))) +  pads
        radius = list( map(lambda aa: self.radius_dict[aa], list(seq))) + pads
        bulkiness = list( map(lambda aa: self.bulkiness_dict[aa], list(seq))) + pads


        return molecular_weight,volume,radius,bulkiness

    def calculate_features(self,seq,seq_max_len):
        """Calculates molecular weight, volume, radius, isoelectric point, side chain pka, gravy of each residue in a protein sequence"""
        seq = "".join(seq).replace("#", "")

        pads = [0] *(seq_max_len-len(seq))
        molecular_weight = calculate_molecular_weight(seq)
        volume = sum(list( map(lambda aa: self.volume_dict[aa], list(seq))) +  pads)
        radius = sum(list( map(lambda aa: self.radius_dict[aa], list(seq))) + pads)
        bulkiness = sum(list( map(lambda aa: self.bulkiness_dict[aa], list(seq))) + pads)
        isoelectric = calculate_isoelectric(seq)
        gravy = calculate_gravy(seq)
        side_chain_pka = sum(list( map(lambda aa: self.side_chain_pka_dict[aa], list(seq))) + pads)/len(seq)
        aromaticity = calculate_aromaticity(seq)
        extintion_coefficient_reduced,extintion_coefficient_cystines = calculate_extintioncoefficient(seq)
        aminoacid_frequencies = list(map(lambda aa,seq: seq.count(aa)/len(seq),self.aminoacids_list,[seq]*len(self.aminoacids_list)))
        #aminoacid_frequencies_dict = dict(zip(self.aminoacids_list,aminoacid_frequencies))
        if self.return_aa_freqs:
            return molecular_weight,volume,radius,bulkiness,isoelectric,gravy,side_chain_pka,aromaticity,extintion_coefficient_reduced,extintion_coefficient_cystines,*aminoacid_frequencies
        else:
            return molecular_weight,volume,radius,bulkiness,isoelectric,gravy,side_chain_pka,aromaticity,extintion_coefficient_reduced,extintion_coefficient_cystines

    def calculate_aminoacid_frequencies(self,seq,seq_max_len):

        seq = "".join(seq).replace("#", "")
        aminoacid_frequencies = list(map(lambda aa,seq: seq.count(aa)/len(seq),self.aminoacids_list,[seq]*len(self.aminoacids_list)))
        aminoacid_frequencies_dict = dict(zip(self.aminoacids_list,aminoacid_frequencies))

        return aminoacid_frequencies_dict

    def volumetrics_summary(self):

        if self.list_sequences:
            results = list(map(lambda seq: self.calculate_volumetrics(seq,self.seq_max_len), self.list_sequences))
            zipped_results = list(zip(*results))
            volumetrics_dict = {"molecular_weights":np.array(zipped_results[0]),
                                "volume":np.array(zipped_results[1]),
                                "radius":np.array(zipped_results[2]),
                                "bulkiness":np.array(zipped_results[3])}
        else:
            volumetrics_dict = {"molecular_weights":None,
                                "volume":None,
                                "radius":None,
                                "bulkiness":None}

        return volumetrics_dict

    def features_summary(self):

        if self.list_sequences:
            results = list(map(lambda seq: self.calculate_features(seq,self.seq_max_len), self.list_sequences))
            zipped_results = list(zip(*results))
            if self.return_aa_freqs:
                features_dict = {"molecular_weights": np.array(zipped_results[0]),
                                 "volume": np.array(zipped_results[1]),
                                 "radius": np.array(zipped_results[2]),
                                 "bulkiness": np.array(zipped_results[3]),
                                 "isoelectric": np.array(zipped_results[4]),
                                 "gravy": np.array(zipped_results[5]),
                                 "side_chain_pka": np.array(zipped_results[6]),
                                 "aromaticity": np.array(zipped_results[7]),
                                 "extintion_coefficient_cysteines": np.array(zipped_results[8]),
                                 "extintion_coefficient_cystines": np.array(zipped_results[9]),
                                 #"aminoacid_frequencies_dict":dict(zip(self.list_sequences,zipped_results[10]))
                                 }

                frequencies_dict = dict.fromkeys(self.aminoacids_list)
                for i in range(10,len(self.aminoacids_list) + 10):
                    aa_name = self.aminoacids_list[int(i-10)]
                    aa_freq = np.array(zipped_results[i])
                    frequencies_dict[aa_name] = aa_freq

                if self.only_w :
                    features_dict["Tryptophan"] = frequencies_dict["W"]
                else:
                    features_dict = {**features_dict,**frequencies_dict}

                #features_dict[""] = frequencies_dict["W"]

            else:
                features_dict = {"molecular_weights":np.array(zipped_results[0]),
                                    "volume":np.array(zipped_results[1]),
                                    "radius":np.array(zipped_results[2]),
                                    "bulkiness":np.array(zipped_results[3]),
                                    "isoelectric":np.array(zipped_results[4]),
                                    "gravy":np.array(zipped_results[5]),
                                    "side_chain_pka":np.array(zipped_results[6]),
                                    "aromaticity":np.array(zipped_results[7]),
                                    "extintion_coefficient_cysteines":np.array(zipped_results[8]),
                                    "extintion_coefficient_cystines": np.array(zipped_results[9]),
                                     #"aminoacid_frequencies_dict":dict(zip(self.list_sequences,zipped_results[10]))
                                    }
        else:
            features_dict = {"molecular_weights":None,
                                "volume":None,
                                "radius":None,
                                "bulkiness":None,
                                "isoelectric": None,
                                "gravy": None,
                                "side_chain_pka": None,
                                "aromaticity":None,
                                "extintion_coefficient_reduced":None,
                                "extintion_coefficient_cystines": None,
                                #"aminoacid_frequencies_dict":None
                             }

        return features_dict

    def aminoacid_frequencies(self):
        if self.list_sequences:
            aminoacid_frequencies_dict= list(map(lambda seq: self.calculate_aminoacid_frequencies(seq,self.seq_max_len), self.list_sequences))

            return aminoacid_frequencies_dict
        else:
            raise ValueError("sequences list is empty")

    def aminoacid_embedding(self):
        if self.list_sequences:
            results = list(map(lambda seq: self.calculate_features(seq,self.seq_max_len), self.list_sequences))



            return results

        else:
            return None

def build_features_dicts(dataset_info):
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) #finds the /data folder of the repository
    features_dicts = CalculatePeptideFeatures(dataset_info.seq_max_len,[],storage_folder).return_dicts()
    gravy_dict = features_dicts.gravy_dict
    volume_dict = features_dicts.volume_dict
    radius_dict = features_dicts.radius_dict
    side_chain_pka_dict = features_dicts.side_chain_pka_dict
    isoelectric_dict = features_dicts.isoelectric_dict
    bulkiness_dict = features_dicts.bulkiness_dict


    if dataset_info.corrected_aa_types == 20:
        aminoacids_dict = aminoacid_names_dict(dataset_info.corrected_aa_types, zero_characters=[])
    else:
        aminoacids_dict = aminoacid_names_dict(dataset_info.corrected_aa_types, zero_characters=["#"])
        gravy_dict["#"] = 0
        volume_dict["#"] = 0
        radius_dict["#"] = 0
        side_chain_pka_dict["#"] = 0
        isoelectric_dict["#"] = 0
        bulkiness_dict["#"] = 0

    aminoacids_dict_reversed = {val:key for key,val in aminoacids_dict.items()}
    gravy_dict = {aminoacids_dict[key]:value for key,value in gravy_dict.items()}
    volume_dict = {aminoacids_dict[key]:value for key,value in volume_dict.items()}
    radius_dict = {aminoacids_dict[key]:value for key,value in radius_dict.items()}
    side_chain_pka_dict = {aminoacids_dict[key]:value for key,value in side_chain_pka_dict.items()}
    isoelectric_dict = {aminoacids_dict[key]:value for key,value in isoelectric_dict.items()}
    bulkiness_dict = {aminoacids_dict[key]:value for key,value in bulkiness_dict.items()}

    return {"aminoacids_dict_reversed":aminoacids_dict_reversed,
            "gravy_dict":gravy_dict,
            "volume_dict":volume_dict,
            "radius_dict":radius_dict,
            "side_chain_pka_dict":side_chain_pka_dict,
            "isoelectric_dict":isoelectric_dict,
            "bulkiness_dict":bulkiness_dict,
            "storage_folder":storage_folder}

def merge_in_left_order(x, y, on=None):
    x = x.copy()
    x["Order"] = np.arange(len(x))
    z = x.merge(y, how='left', on=on).set_index("Order").loc[np.arange(len(x)), :]
    return z

def calculate_correlations(feat1,feat2,method="pearson"):
    unique_vals = np.unique(feat2)
    if (unique_vals.astype(int) == unique_vals).sum() == len(unique_vals): #if the variable is categorical
        #print("found categorical variable")
        result =  scipy.stats.pointbiserialr(feat1, feat2)
    else:
        #print("continuous variable")
        if method == "pearson":
            result =  scipy.stats.pearsonr(feat1, feat2)
        else:
            result =  scipy.stats.spearmanr(feat1, feat2)
    return result

def generate_mask(max_len, length):
    seq_mask = np.array([True] * (length) + [False] * (max_len - length))
    return seq_mask[None, :]

def clean_generated_sequences(seq_int,seq_mask,zero_character,min_len,max_len,keep_truncated=False):
    """"""
    seq_mask = np.array(seq_mask)
    seq_int = np.array(seq_int)
    if zero_character is not None:
        idx = np.where(seq_int == zero_character)[0]
        if idx.size == 0:
            seq_mask = np.ones_like(seq_int).astype(bool)
            return (seq_int[None,:],seq_mask[None,:])
        else:
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

def numpy_to_fasta(aa_sequences,binary_pedictions,probabilities,results_dir,folder_name="",title_name=""):
    print("Saving generated sequences to fasta & text files ")
    f1 = open("{}/epitopes{}.fasta".format(results_dir,title_name), "a+")
    f2 = open("{}/epitopes{}.txt".format(results_dir,title_name), "a+")

    headers_list  = list(map(lambda idx,label,prob: ">Epitope_{}_class_{}_probability_{}\n".format(idx,label,prob), list(range(aa_sequences.shape[0])),binary_pedictions.tolist(),probabilities.tolist()))

    sequences_list = list(map(lambda seq: "{}\n".format("".join(seq).replace("#","-")), aa_sequences.tolist()))

    headers_sequences_list = [None]*len(headers_list) + [None]*len(sequences_list)

    headers_sequences_list[::2] = headers_list
    headers_sequences_list[1::2] = sequences_list

    f1.write("".join(headers_sequences_list))
    f2.write("".join(sequences_list))

    df = pd.DataFrame({"Epitopes":sequences_list,"Negative_score":probabilities[:,0].tolist(),"Positive_score":probabilities[:,1].tolist()})
    df["Epitopes"] = df["Epitopes"].str.replace("\n","")
    df.to_csv("{}/epitopes.tsv".format(results_dir),sep="\t",index=False)

    VegvisirPlots.plot_generated_labels_histogram(df,results_dir)

    #try:
    sequences_list2 = list(map(lambda seq: "{}".format("".join(seq).replace("#", "-")), aa_sequences.tolist()))
    VegvisirPlots.plot_logos(sequences_list2,results_dir,"ALL_generated")
    #except:
    #    pass

    positive_sequences = df[df["Positive_score"] >= 0.6]
    positive_sequences_list = positive_sequences["Epitopes"].tolist()

    if positive_sequences_list:
        VegvisirPlots.plot_logos(positive_sequences_list,results_dir,"POSITIVES_generated")

    negative_sequences = df[df["Negative_score"] < 0.4]
    negative_sequences_list = negative_sequences["Epitopes"].tolist()

    if negative_sequences_list:
        VegvisirPlots.plot_logos(negative_sequences_list,results_dir,"NEGATIVES_generated")

def squeeze_tensor(required_ndims,tensor):
    """Squeezes a tensor to match ndim"""
    size = torch.tensor(tensor.shape)
    ndims = len(size)
    idx_ones = (size == 1)
    if True in idx_ones:
        ones_pos = size.tolist().index(1)
    if ndims > required_ndims:
        while ndims > required_ndims:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
                size = torch.tensor(tensor.shape)
                ndims = len(size)
            elif True in idx_ones:
                tensor = tensor.squeeze(ones_pos)
                size = torch.tensor(tensor.shape)
                ndims = len(size)
            else:
                ndims = required_ndims

        return tensor
    else:
        return tensor

def clustering_accuracy(binary_arr):
    """Computes a clustering accuracy score"""
    maxlen = binary_arr.shape[0]
    count_ones, count_zeros = 0, 0
    max_count_ones, max_count_zeros = 0, 0
    previdx_ones, previdx_zeros = 0, 0
    groups_counts_ones, groups_counts_zeros = defaultdict(), defaultdict() #registers the start index of the 1ś clusters
    for idx,num in enumerate(binary_arr):
        if num != 1:
            max_count_ones = max(max_count_ones, count_ones)
            if idx == 0:
                groups_counts_ones[previdx_ones] = count_ones #previous index plus 1
            else:
                groups_counts_ones[previdx_ones +1] = count_ones #previous index plus 1
            previdx_ones=idx
            count_ones = 0
            count_zeros += 1
        else:
            max_count_zeros = max(max_count_zeros, count_zeros)
            if idx == 0:
                groups_counts_zeros[previdx_zeros] = count_zeros #previous index plus 1
            else:
                groups_counts_zeros[previdx_zeros + 1] = count_zeros #previous index plus 1
            count_zeros = 0
            previdx_zeros=idx
            count_ones += 1

    if previdx_zeros + 1 < maxlen:
        groups_counts_zeros[previdx_zeros + 1] = count_zeros #previous index plus 1, have to add this here
    if previdx_ones + 1 < maxlen:
        groups_counts_ones[previdx_ones + 1] = count_ones #previous index plus 1, have to add this here

    maxcountones = max(max_count_ones, count_ones)
    maxcountzeros = max(max_count_zeros, count_zeros)
    total_std_idx = np.std(np.arange(maxlen))
    total_ones = np.sum(binary_arr)
    total_zeros = binary_arr.shape[0] - total_ones


    # Highlight: Transform to array keeping only the results for the ones
    starting_idx_ones = np.array([key for key, val in groups_counts_ones.items() if val != 0])
    idx_ones = np.array(binary_arr == 1)
    idx_ones = np.arange(maxlen)[idx_ones]
    cluster_sizes_ones = np.array([val for key, val in groups_counts_ones.items() if val != 0])
    #counts_array_ones = np.concatenate([starting_idx[:, None], cluster_size[:, None]], axis=1)

    if cluster_sizes_ones.size != 0:
        std_idx_ones = np.std(idx_ones)
        max_size_ones = np.max(cluster_sizes_ones)
    else:
        std_idx_ones = total_std_idx
        max_size_ones = total_ones = 1

    starting_idx_zeros = np.array([key for key, val in groups_counts_zeros.items() if val != 0])
    idx_zeros = np.array(binary_arr == 0)
    idx_zeros = np.arange(maxlen)[idx_zeros]
    cluster_sizes_zeros = np.array([val for key, val in groups_counts_zeros.items() if val != 0])

    if cluster_sizes_zeros.size != 0:
        std_idx_zeros = np.std(idx_zeros)
        max_size_zeros = np.max(cluster_sizes_zeros)
    else:
        std_idx_zeros = total_std_idx
        max_size_zeros = total_zeros = 1

    score_a = ((max_size_ones*100/total_ones) + (max_size_zeros*100/total_zeros)) / 2
    score_b = ((std_idx_ones*100/total_std_idx) + (std_idx_zeros*100/total_std_idx) + 2*(max_size_ones*100/total_ones) + 2*(max_size_zeros*100/total_zeros)) / 6

    return {"maxcountones": maxcountones,
            "group_counts_ones": groups_counts_ones,
            "maxcountzeros": maxcountzeros,
            "group_counts_zeros": groups_counts_zeros,
            "clustering_score_a" : score_a,
            "clustering_score_b" : score_b
            }

def clustering_significance(labels):
    """Performs a permutation test on the location of the labels (idx label 0 or idx label 1) to estimate the significance of the clusters, whether they are due to random or not ...
    NOTES:
        For a given array of clustered labels of size N, calculate the average rank of one of the two labels, say label 1. Call this value <R_1>

        Next repeat 10000 times:permute the order of the array of clustered labels (each time with a new seed)
                calculate the average rank of the entries with label 1, <R_1_perm>
        Next,
        if <R_1>  < N/2:
                the p-value for the rank of label 1 entries in the original data is random will be equal to the proportion of <R_1_perm> values that are lower than <R_1>
        else
                the p-value for the rank of label 1 entries in the original data is random will be equal to the proportion of <R_1_perm> values that are higher than <R_1>
    """

    idx_ones = np.where(labels==1)[0].astype(float)
    ndata = labels.shape[0]
    r1_avg = np.average(idx_ones)
    def calculate_r1(i,labels):
        np.random.seed(i)
        shuffled_labels = np.array(labels).copy()
        np.random.shuffle(shuffled_labels)
        idx_ones_shuffled = np.where(shuffled_labels==1)[0].astype(float)
        r1_permuted = np.average(idx_ones_shuffled)
        return r1_permuted

    r1_permuted_list = list(map(lambda i: calculate_r1(i,labels), list(range(10000))))

    r1_permuted_arr = np.array(r1_permuted_list)
    if r1_avg < ndata/2:
        lower_idx = np.where(r1_permuted_arr < r1_avg)[0]

        pval = len(lower_idx)/ndata
    else:
        higher_idx = np.where(r1_permuted_arr > r1_avg)[0]
        pval = len(higher_idx)/ndata
    return pval


#TODO: Put into new plots_utils.py, however right now it is annoying to change the structure because of dill
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns;sns.set()
class SeabornFig2Grid():
    """Class from https://stackoverflow.com/questions/47535866/how-to-iteratively-populate-matplotlib-gridspec-with-a-multipart-seaborn-plot/47624348#47624348"""
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        elif isinstance(self.sg, sns.matrix.ClusterGrid):#https://github1s.com/mwaskom/seaborn/blob/master/seaborn/matrix.py#L696
            # print(dir(self.sg))
            # print(dir(self.sg.figure))
            self._moveclustergrid()
        else:
            print("what am i")

        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveclustergrid(self):
        """Move cluster grid"""
        r = len(self.sg.figure.axes)
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r, r + 10, subplot_spec=self.subplot)
        subplots_axes = self.sg.figure.axes
        self._resize()
        self._moveaxes(subplots_axes[0], self.subgrid[1:, 0:3]) #left cladogram #ax_row_dendrogram
        self._moveaxes(subplots_axes[1], self.subgrid[0, 4:-2]) #top cladogram #ax_col_dendrogram
        self._moveaxes(subplots_axes[2], self.subgrid[1:, 3]) #labels bar
        self._moveaxes(subplots_axes[3], self.subgrid[1:, 4:-2]) #heatmap #ax_heatmap
        self._moveaxes(subplots_axes[4], self.subgrid[1:, -1]) #colorbar


    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())




