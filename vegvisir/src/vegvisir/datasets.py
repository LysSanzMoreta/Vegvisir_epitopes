import os
import time,datetime
import dill
import pandas as pd
import operator,functools
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
import dataframe_image as dfi
import vegvisir.nnalign as VegvisirNNalign
import vegvisir.utils as VegvisirUtils
def available_datasets():
    """Prints the available datasets"""
    datasets = {0:"viral_dataset",
                1:"viral_dataset2"}
    return datasets
def select_dataset(dataset_name,script_dir,args,update=True):
    """Selects from available datasets
    :param dataset_name: dataset of choice
    :param script_dir: Path from where the scriptis being executed
    :param update: If true it will download and update the most recent version of the dataset
    """
    func_dict = {"viral_dataset": viral_dataset,
                 "viral_dataset2":viral_dataset2}
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) #finds the /data folder of the repository

    dataset_load_fx = lambda f,dataset_name,current_path,storage_folder,args,update: lambda dataset_name,current_path,storage_folder,args,update: f(dataset_name,current_path,storage_folder,args,update)
    data_load_function = dataset_load_fx(func_dict[dataset_name],dataset_name,script_dir,storage_folder,args,update)
    dataset = data_load_function(dataset_name,script_dir,storage_folder,args,update)
    print("Data retrieved")

    return dataset

def viral_dataset(dataset_name,current_path,storage_folder,args,update):
    """Loads the viral dataset generated from **IEDB** using parameters:
           -Epitope: Linear peptide
           -Assay: T-cell
           -Epitope source: Organism: Virus
           -MHC Restriction: Class I
           -Host: Human
           -Disease: Any

    The dataset is organized as follows:
    ####################
    #HEADER DESCRIPTIONS#
    ####################
    #id: unique id for each datapoint in the database.
    #Icore: Interaction core. This is the sequence of the binding core including eventual insertions of deletions (derived from the prediction of the likelihood of binding of the peptide to the reported MHC-I with NetMHCpan-4.1).
    #Allele: MHC class I allele reported in IEDB.
    #protein_sequence: Protein sequence recovered from Entrez given the ENSP identifier reported in the IEDB. This is the so-called "source protein" of the peptide.
    #Core: The minimal 9 amino acid binding core directly in contact with the MHC (derived from the prediction with NetMHCpan-4.1).
    #Of: The starting position of the Core within the Peptide (if > 0, the method predicts a N-terminal protrusion) (derived from the prediction with NetMHCpan-4.1).
    #Gp: Position of the deletion, if any (derived from the prediction with NetMHCpan-4.1).
    #Gl: Length of the deletion, if any (derived from the prediction with NetMHCpan-4.1).
    #Ip: Position of the insertion, if any (derived from the prediction with NetMHCpan-4.1).
    #Il: Length of the insertion, if any (derived from the prediction with NetMHCpan-4.1).
    #Rnk_EL: The %rank value reflects the likelihood of binding of the peptide to the reported MHC-I, computed with NetMHCpan-4.1.
    The lower the rank the stronger the binding of the peptide with the reported MHC.
    #org_id: id of the organism the peptide derives from, reported by the IEDB.
    #prot_name: protein name (reported by the IEDB).
    #uniprot_id: UniProt ID (reported by the IEDB).
    #number_of_papers_positive: number of papers where the peptide-MHC was reported positive.
    #number_of_papers_negative: number of papers where the peptide-MHC was reported negative.
    #target: target value (1: immunogenic/positive, 0:non-immunogenic/negative).
    #target_bin_2: corrected target value, where positives are considered as "1" only if they are reported as positives in 2 or more papers.
    #start_prot: aa position (index) where the peptide starts within its source protein.
    #start_prot: aa position where the peptide ends within its source protein.
    #filter_register: dismiss this field.
    #training: "1" if the datapoint is considered part of the training set and "0" is it belongs to the validation set.
    #partition: number of training partition the datapoint is assigned to (0 to 4). The training is done in a 5-fold cross-validation scheme.
    """
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    sequence_column = ["Core","Icore"][1]
    score_column = ["Rnk_EL","target"][1]
    data = pd.read_csv("{}/viral_dataset/Viruses_db_partitions.tsv".format(storage_folder),sep="\t")
    nnalign_input = data[[sequence_column,score_column,"training","partition"]]
    # print(nnalign_input.shape)
    # peptides = nnalign_input[["Icore"]].values.tolist()
    # peptides = functools.reduce(operator.iconcat, peptides, []) #flatten list of lists
    # aas = [list(pep) for pep in peptides]
    # aas = functools.reduce(operator.iconcat, aas, [])
    # aas_unique = list(set((aas)))
    # for aa in aas_unique:
    #     if aa not in alphabet:
    #         print(aa)
    # print(aas_unique)
    # exit()
    # peptides_unique = list(set((peptides)))
    # print(nnalign_input.shape)
    # exit()
    nnalign_input_train = nnalign_input.loc[nnalign_input['training'] == 1]
    nnalign_input_eval = nnalign_input.loc[nnalign_input['training'] == 0]
    # peptides = nnalign_input_train[["Core"]].values.tolist()
    # peptides = functools.reduce(operator.iconcat, peptides, []) #flatten list of lists
    # peptides_unique = list(set((peptides)))
    nnalign_input_train = nnalign_input_train.drop_duplicates(sequence_column,keep="first")
    nnalign_input_eval = nnalign_input_eval.drop_duplicates(sequence_column, keep="first")
    nnalign_input_train.drop('training',inplace=True,axis=1)
    nnalign_input_eval.drop('training', inplace=True,axis=1)
    nnalign_input_eval.drop('partition', inplace=True, axis=1)
    nnalign_input_train.to_csv("{}/viral_dataset/viral_nnalign_input_train.tsv".format(storage_folder),sep="\t",index=False)
    nnalign_input_eval.to_csv("{}/viral_dataset/viral_nnalign_input_eval.tsv".format(storage_folder), sep="\t",index=False) #TODO: Header None?

    if args.run_nnalign:
        VegvisirNNalign.run_nnalign(args,storage_folder)



def viral_dataset2(dataset_name,current_path,storage_folder,args,update):
    """Loads the viral dataset generated from **IEDB** database using parameters:
           -Epitope: Linear peptide
           -Assay: T-cell
           -Epitope source: Organism: Virus
           -MHC Restriction: Class I
           -Host: Human
           -Disease: Any

    The dataset is organized as follows:
    ####################
    #HEADER DESCRIPTIONS#
    ####################
    ('Reference', 'Date')
    ('Epitope', 'Description')
    ('Epitope', 'Organism Name')
    ('Epitope', 'Parent Species ID')
    ('Epitope', 'Parent Protein Accession')
    ('Assay', 'Qualitative Measure')
    ('Assay', 'Number of Subjects Tested'):Number of people in the study evaluated for immunogenicity against the epitope
    ('Assay', 'Number of Subjects Responded')
    ('Assay', 'Response Frequency')
    ('MHC', 'Allele Name'): HLA alelle name (MHC allele)
    Icore: Interaction core. This is the sequence of the binding core including eventual insertions of deletions (derived from the prediction of the likelihood of binding of the peptide to the reported MHC-I with NetMHCpan-4.1)
    Rnk_EL: The %rank value reflects the likelihood of binding of the peptide to the reported MHC-I, computed with NetMHCpan-4.1.

    return
      Icore:Interaction peptide core
      Confidence_score: Number of + / Number of tested ---> Normalized to 0-1
      Rnk_EL: Average rank score per peptide by NetMHC ---> Normalized to 0-1
    """
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    data = pd.read_csv("{}/{}/Viruses_predict_hla.csv".format(storage_folder,args.dataset_name),sep="\t")
    columns = ["Reference_data","Epitope_description","Epitope_organism_name","Parent_species_id","Parent_protein_accession",
               "Assay_qualitative_measure","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded",
               "Assay_response_frequency","MHC_allele_name","Icore","Rnk_EL"]
    data.columns = columns
    data_a = data.groupby('Icore',as_index=False)[["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
    data_b = data.groupby('Icore',as_index=False)[["Rnk_EL"]].agg(lambda x: sum(list(x))/len(list(x)))
    #max_number_subjects = data["Assay_number_of_subjects_tested"].max()
    data_a["confidence_score"] = data_a["Assay_number_of_subjects_responded"]/data_a["Assay_number_of_subjects_tested"]
    #data["confidence_score"] = data["Assay_number_of_subjects_responded"] / max_number_subjects
    data_a.fillna(0,inplace=True)
    max_conf = data_a["confidence_score"].max()
    min_conf = data_a["confidence_score"].min()
    data_a["confidence_score"] = (data_a["confidence_score"] - min_conf)/max_conf-min_conf
    #Likelihood rank
    max_conf = data_b["Rnk_EL"].max()
    min_conf = data_b["Rnk_EL"].min()
    data_b["Rnk_EL"] = (data_b["Rnk_EL"] - min_conf) / max_conf - min_conf
    data_a["Rnk_EL"] = data_b["Rnk_EL"]

    data_b.fillna(0, inplace=True)
    fig, ax = plt.subplots()
    # the histogram of the data
    num_bins = 50
    ax.hist(data_a["confidence_score"].to_numpy() , num_bins, density=True)
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of confidence scores ')
    fig.tight_layout()
    plt.savefig("{}/{}/Viruses_histogram_dataset_labels".format(storage_folder,args.dataset_name),dpi=300)
    plt.clf()
    fig, ax = plt.subplots()
    # the histogram of the data
    num_bins = 50
    ax.hist(data_a["Rnk_EL"].to_numpy(), num_bins, density=True)
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of Rnk_EL scores')
    fig.tight_layout()
    plt.savefig("{}/{}/Viruses_histogram_dataset_rank".format(storage_folder,args.dataset_name), dpi=300)
    plt.clf()
    data = process_data(data_a,args,storage_folder)

    return data

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


def store_dataset():
    """"""


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
    #array = array[:5000]
    n_data = array.shape[0]
    array_mask = array_mask[:n_data]
    split_size = [int(array.shape[0] / batch_size) if not batch_size > array.shape[0] else 1][0]
    splits = np.array_split(array, split_size)
    mask_splits = np.array_split(array_mask, split_size)
    print("Generated {} splits".format(len(splits)))
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
            start_store_point_i = end_store_point_i
            if j + 1 != len(rest_splits):
                end_store_point_i += rest_splits[j + 1].shape[0]  # it has to be the next r_j
        end_i = time.time()
        print("Time for finishing loop (i vs j) {}".format(str(datetime.timedelta(seconds=end_i - start_i))))
        percent_identity_mean[start_store_point:end_store_point] = percent_identity_mean_i
        cosine_similarity_mean[start_store_point:end_store_point] = cosine_similarity_mean_i
        kmers_cosine_similarity[start_store_point:end_store_point] = kmers_cosine_similarity_i
        kmers_pid_similarity[start_store_point:end_store_point] = kmers_pid_similarity_i
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


def calculate_similarity_matrix_old(array,max_len,array_mask,batch_size=200,ksize=3):
    """Batched method to calculate the cosine similarity and percent identity/pairwise distance between the blosum encoded sequences.
    :param numpy array: Integer representation [n,max_len] or Blosum encoded [n,max_len,aa_types]
    :param numpy array_nan: Integer representation as [n,max_len] or Blosum encoded as [n,max_len,aa_types]. The values of the padding values (#) are represented as np.nan
    NOTE: Use smaller batches for faster results ( obviously to certain extent, check into balancing the batch size and the number of for loops)
    returns
        pairwise_similarity_matrices: [n,n,max-len,max_len] : Per sequence compare all amino acids from one sequence compared against all amino acids of the other sequence ---> useful for k-mers calculation
        percent_identity: [n,n,max_len] ---> Percent identity
        cosine_similarities: [n,n,max-len,max_len] ---> Per sequence calculate the cosine similarity among all the "amino acids blosum vectors" from one sequence compared against all "amino acids blosum vectors" of the other sequence ---> Useful for k-mers calculation
                            1 means the two aa are identical and −1 means the two aa are not similar."""
    #TODO: Make it run with Cython (faster indexing): https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
    array = array[:5]
    n_data = array.shape[0]
    array_mask = array_mask[:n_data]
    #print(array.shape)
    #array = np.ma.masked_array(array,array_mask[:300][:,:,None].repeat(array.shape[-1],axis= -1))
    #batch_size = array.shape[0]
    split_size = [int(array.shape[0]/batch_size) if not batch_size > array.shape[0] else 1][0]
    splits = np.array_split(array,split_size)
    mask_splits = np.array_split(array_mask,split_size)
    print("Generated {} splits".format(len(splits)))
    idx = list(range(len(splits)))
    if ksize >= max_len:
        ksize = max_len
    overlapping_kmers = extract_windows_vectorized(splits[0],1,max_len-ksize,ksize,only_windows=True) #TODO:Might not be necessary o yes

    diag_idx = np.diag_indices(ksize)
    nkmers = overlapping_kmers.shape[0]
    diag_idx_nkmers = np.diag_indices(nkmers)
    diag_idx_maxlen = np.diag_indices(max_len)

    #results_dict = defaultdict(lambda : defaultdict())
    if np.ndim(array) == 2: #TODO: this approach might be limited by memory
        pairwise_similarity_matrices = np.zeros((array.shape[0],) + array.shape[-2:] + (array.shape[-1],))  # [n,n,max_len,max_len]
        percent_identity = np.zeros(((array.shape[0],) + array.shape[-2:]))  # [n,n,max_len]
        cosine_similarities = np.zeros((( (array.shape[0],) + array.shape[-2:] + (array.shape[-1],))))  # [n,n,max_len,max_len]
    else:
        pairwise_similarity_matrices = np.zeros((( (array.shape[0],) + array.shape[-3:-1] + (array.shape[-2],))) ) # [n,n,max_len,max_len]
        percent_identity = np.zeros((( (array.shape[0],) + array.shape[-3:-1])))  # [n,n,max_len]
        cosine_similarities = np.zeros(((array.shape[0],) + array.shape[-3:-1] + (array.shape[-2],)))  # [n,n,max_len.max_len]



    #TODO: Avoid storing these matrices
    kmers_matrix = np.zeros((n_data,n_data,nkmers,nkmers,ksize,ksize )) # [n,n,nkmers,nkmers,ksize,ksize]
    kmers_matrix_cosine = np.zeros((n_data,n_data,nkmers,nkmers,ksize,ksize )) # [n,n,nkmers,nkmers,ksize,ksize]

    start_store_point = 0
    end_store_point = splits[0].shape[0]
    start = time.time()
    for i in idx:
        print("i ------------ {}".format(i))
        curr_array = splits[i] #TODO:  Plot correlation %ID and cosine sim correlations
        curr_mask = mask_splits[i]
        rest_splits = splits.copy()
        #del rest_splits[i] #remove the current split from the list?
        #Highlight: Define storing arrays
        if np.ndim(curr_array) == 2:
            pairwise_similarity_matrices_i = np.zeros(((curr_array.shape[0],) + array.shape[-2:] + (array.shape[-1],))) # [m,n,max_len,max_len]
            percent_identity_i = np.zeros(((curr_array.shape[0],) + array.shape[-2:]))  # [m,n,max_len]
            cosine_similarities_i = np.zeros(((curr_array.shape[0],) + array.shape[-2:] + (array.shape[-1],)))  # [m,n,max_len,max_len]
        else:
            pairwise_similarity_matrices_i = np.zeros(((curr_array.shape[0],) + array.shape[-3:-1] + (array.shape[-2],))) #[m,n,max_len,max_len]
            percent_identity_i = np.zeros(((curr_array.shape[0],) + array.shape[-3:-1])) #[m,n,max_len]
            cosine_similarities_i = np.zeros(((curr_array.shape[0],) + array.shape[-3:-1] + (array.shape[-2],))) #[m,n,max_len.max_len]
        kmers_matrix_i = np.zeros((curr_array.shape[0], n_data, nkmers, nkmers, ksize, ksize))  # [n,n,nkmers,nkmers,ksize,ksize]
        kmers_matrix_cosine_i = np.zeros((curr_array.shape[0], n_data, nkmers, nkmers, ksize, ksize))  # [n,n,nkmers,nkmers,ksize,ksize]
        start_store_point_i = 0
        end_store_point_i = rest_splits[0].shape[0] #initialize
        start_i = time.time()
        for j,r_j in enumerate(rest_splits): #calculate distance among all kmers per sequence in the block (n, n_kmers,n_kmers)
            #print("j {}".format(j))
            r_j_mask = mask_splits[j]
            cosine_sim = cosine_similarity(curr_array,r_j, correlation_matrix=False)
            if np.ndim(curr_array) == 2:
                pairwise_sim = (curr_array[None, :] == r_j[:, None]).astype(int)
            else:
                pairwise_sim = (curr_array[:, None] == r_j[None, :]).all((-1)).astype(int)  # .all((-2,-1))
            if np.ndim(curr_array) == 2:
                pairwise_matrix = (curr_array[:, None, :, None] == r_j[None, :, None, :]).astype(int)
            else:
                # pairwise_matrix = (curr_array[:, None,:,None] == r_j[None, :,None,:]).astype(float).sum(-1)  # .all((-2,-1))
                # pairwise_matrix /= curr_array.shape[-1]
                # pairwise_matrix[pairwise_matrix != 1.] = 0
                #TODO: Problem when calculating self.similarity because np.nan == np.nan is False
                pairwise_matrix = (curr_array[:, None, :, None] == r_j[None, :, None, :]).all((-1)).astype(float)  # .all((-2,-1))

            percent_identity_i[:,start_store_point_i:end_store_point_i] = pairwise_sim
            pairwise_similarity_matrices_i[:,start_store_point_i:end_store_point_i] = pairwise_matrix
            cosine_similarities_i[:,start_store_point_i:end_store_point_i] = cosine_sim
            #Highlight: further transformations: Basically slice the overlapping kmers and organize them to have shape [m,n,kmers,nkmers,ksize,ksize], where the diagonal contains the pairwise values between the kmers

            #kmers_matrix_ = pairwise_matrix[:,:,:,overlapping_kmers][:,:,overlapping_kmers].reshape((curr_array.shape[0],curr_array.shape[0],nkmers,ksize,nkmers,ksize),order="A").transpose(0,1,4,2,3,5)
            kmers_matrix_ = pairwise_matrix[:,:,:,overlapping_kmers][:,:,overlapping_kmers].transpose(0,1,4,2,3,5)
            kmers_matrix_i[:,start_store_point_i:end_store_point_i] = kmers_matrix_
            kmers_matrix_cosine_ = cosine_sim[:,:,:,overlapping_kmers][:,:,overlapping_kmers].transpose(0,1,4,2,3,5)
            #kmers_matrix_cosine_ = cosine_sim[:,:,:,overlapping_kmers][:,:,overlapping_kmers].reshape((curr_array.shape[0],curr_array.shape[0],nkmers,ksize,nkmers,ksize),order="A").transpose(0,1,4,2,3,5)
            kmers_matrix_cosine_i[:, start_store_point_i:end_store_point_i] = kmers_matrix_cosine_

            start_store_point_i = end_store_point_i
            if j +1  != len(rest_splits):
                end_store_point_i += rest_splits[j+1].shape[0] #it has to be the next r_j
        end_i = time.time()
        print("Time for finishing loop (j) {}".format(str(datetime.timedelta(seconds=end_i-start_i))))
        pairwise_similarity_matrices[start_store_point:end_store_point] = pairwise_similarity_matrices_i
        percent_identity[start_store_point:end_store_point] = percent_identity_i
        cosine_similarities[start_store_point:end_store_point] = cosine_similarities_i
        # Highlight: further transformations
        kmers_matrix[start_store_point:end_store_point] = kmers_matrix_i
        kmers_matrix_cosine[start_store_point:end_store_point] = kmers_matrix_cosine_i
        start_store_point = end_store_point

        if i + 1 != len(splits):
            end_store_point += splits[i + 1].shape[0]  # it has to be the next curr_array
    end = time.time()
    print("Overall calculation time {}".format(str(datetime.timedelta(seconds=end - start))))

    #kmers_matrix = pairwise_similarity_matrices[:,:,:,overlapping_kmers][:,:,overlapping_kmers].reshape((n_data,n_data,overlapping_kmers.shape[0],ksize,overlapping_kmers.shape[0],ksize),order="A").transpose(0,1,4,2,3,5)
    #Highlight: Create masks to ignore the paddings of the sequences
    kmers_mask = array_mask[:,overlapping_kmers] #TODO also in the for loop?
    kmers_mask = (kmers_mask[:,None]*kmers_mask[None,:]).mean(-1) #TODO: not 100% this is correct
    kmers_mask[kmers_mask!=1.] = 0.
    kmers_mask = kmers_mask.astype(bool)
    pid_mask = array_mask[:,None]*array_mask[None,:]

    #Highlight: Apply masks to calculate the similarities. NOTE: To get the data with the filled value use k = np.ma.getdata(kmers_matrix_diag_masked)
    ##PERCENT IDENTITY (binary pairwise comparison) ###############
    percent_identity_mean = np.ma.masked_array(percent_identity,mask=~pid_mask,fill_value=0.).mean(-1) #Highlight: In the mask if True means to mask and ignore!!!!
    ##COSINE SIMILARITY (pairwise comparison)########################
    cosine_similarity_mean = np.ma.masked_array(cosine_similarities[:,:,diag_idx_maxlen[0],diag_idx_maxlen[1]],mask=~pid_mask,fill_value=0.).mean(-1) #Highlight: In the mask if True means to mask and ignore!!!!
    #KMERS PERCENT IDENTITY ############
    kmers_matrix_diag = kmers_matrix[:,:,:,:,diag_idx[0],diag_idx[1]] #does not seem expensive
    kmers_matrix_diag_2 = np.mean(kmers_matrix_diag,axis=4)[:,:,diag_idx_nkmers[0],diag_idx_nkmers[1]] #if we mask this only it should be fine
    kmers_pid_similarity= np.ma.masked_array(kmers_matrix_diag_2,mask=~kmers_mask,fill_value=0.).mean(axis=2)
    #KMERS COSINE SIMILARITY
    kmers_matrix_cosine_diag = kmers_matrix_cosine[:,:,:,:,diag_idx[0],diag_idx[1]] #does not seem expensive
    kmers_matrix_cosine_diag_2 = np.nanmean(kmers_matrix_cosine_diag,axis=4)[:,:,diag_idx_nkmers[0],diag_idx_nkmers[1]]
    kmers_cosine_similarity = np.ma.masked_array(kmers_matrix_cosine_diag_2,mask=~kmers_mask,fill_value=0.).mean(axis=2)
    # print("Kmers % ID")
    # print(kmers_similarity[0][0:4])
    # print("Kmers Cosine similarity")
    # print(kmers_similarity_cosine[0][0:4])
    # print("Percent ID")
    # print(percent_identity_mean[0][0:4])
    # print("Cosine similarity")
    # print(cosine_similarities_mean[0][0:4])
    # print("--------------------")
    # print("Kmers % ID")
    # print(kmers_similarity[1][0:4])
    # print("Kmers Cosine similarity")
    # print(kmers_similarity_cosine[1][0:4])
    # print("Percent ID")
    # print(percent_identity_mean[1][0:4])
    # print("Cosine similarity")
    # print(cosine_similarities_mean[1][0:4])

    return np.ma.getdata(percent_identity_mean),np.ma.getdata(cosine_similarity_mean),np.ma.getdata(kmers_pid_similarity),np.ma.getdata(kmers_cosine_similarity)

def process_data(data,args,storage_folder,plot_blosum=False):
    """
    :param pandas dataframe data: Contains Icore, Confidence_score and Rnk_EL
    :param args
    :param storage_folder
    """
    blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(args.aa_types, args.subs_matrix)
    if plot_blosum:
        blosum_cosine = cosine_similarity(blosum_array[1:, 1:], blosum_array[1:, 1:])
        aa_dict = VegvisirUtils.aminoacid_names_dict(args.aa_types,zero_characters=["#"])
        aa_list =[key for key,val in aa_dict.items() if val in list(blosum_array[:,0])]
        blosum_cosine_df = pd.DataFrame(blosum_cosine,columns=aa_list,index=aa_list)
        sns.heatmap(blosum_cosine_df.to_numpy(),
                    xticklabels=blosum_cosine_df.columns.values,
                    yticklabels=blosum_cosine_df.columns.values,annot=True,annot_kws={"size": 4},fmt=".2f")
        plt.savefig('{}/{}/blosum_cosine.png'.format(storage_folder,args.dataset_name),dpi=600)

    aa_dict = VegvisirUtils.aminoacid_names_dict(args.aa_types, zero_characters=["#"])
    epitopes = data[["Icore"]].values.tolist()
    epitopes = functools.reduce(operator.iconcat, epitopes, [])  # flatten list of lists
    epitopes_max_len = len(max(epitopes, key=len))
    epitopes_lens = np.array(list(map(len, epitopes)))

    #Pad the sequences
    epitopes = [list(seq.ljust(epitopes_max_len, "#")) for seq in epitopes]
    print(epitopes[0])
    print(epitopes[1])
    print(epitopes[2])
    print(epitopes[3])
    #print(epitopes[1])
    epitopes_array = np.array(epitopes)
    epitopes_array_int = np.vectorize(aa_dict.get)(epitopes_array)
    epitopes_mask = epitopes_array_int.astype(bool)
    epitopes_array_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(epitopes_array_int)
    ksize = 3
    if not os.path.exists("{}/{}/pairwise_similarity_matrices.npy".format(storage_folder,args.dataset_name)):
        print("Epitopes similarity matrices not existing, calculating ....")
        percent_identity_mean,cosine_similarity_mean,kmers_pid_similarity,kmers_cosine_similarity = calculate_similarity_matrix(epitopes_array_blosum,epitopes_max_len,epitopes_mask,ksize=ksize)
        np.save("{}/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name), percent_identity_mean)
        np.save("{}/{}/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name), cosine_similarity_mean)
        np.save("{}/{}/kmers_pid_similarity_{}ksize.npy".format(storage_folder,args.dataset_name,ksize), kmers_pid_similarity)
        np.save("{}/{}/kmers_cosine_similarity_{}ksize.npy".format(storage_folder,args.dataset_name,ksize), kmers_cosine_similarity)

    else:
        print("Loading pre-calculated epitopes similarity matrices")
        percent_identity_mean = np.load("{}/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name))
        cosine_similarity_mean = np.load("{}/{}/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name))
        kmers_pid_similarity = np.load("{}/{}/kmers_pid_similarity_{}ksize.npy".format(storage_folder, args.dataset_name,ksize))
        kmers_cosine_similarity = np.save("{}/{}/kmers_cosine_similarity_{}ksize.npy".format(storage_folder, args.dataset_name,ksize))

    #comparison = np.char.add(epitopes_array [:,None], epitopes_array [None,:]) #a = np.array([["A","R","T","#"],["Y","M","T","P"],["I","R","T","#"]])

