import os
import pandas as pd
import operator,functools
import matplotlib.pyplot as plt
import numpy as np
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
    data = pd.read_csv("{}/viral_dataset/Viruses_predict_hla.csv".format(storage_folder),sep="\t")
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
    plt.savefig("{}/viral_dataset/Viruses_histogram_dataset_labels".format(storage_folder),dpi=300)

    fig, ax = plt.subplots()
    # the histogram of the data
    num_bins = 50
    ax.hist(data_a["Rnk_EL"].to_numpy(), num_bins, density=True)
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of Rnk_EL scores')
    fig.tight_layout()
    plt.savefig("{}/viral_dataset/Viruses_histogram_dataset_rank".format(storage_folder), dpi=300)
    data = process_data(data_a,args)

    return data


def cosine_similarity(a,b,correlation_matrix=False):
    """Calculates the cosine similarity between matrices of k-mers.
    :param numpy array a: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param numpy array b: (max_len,aa_types) or (num_seq,max_len, aa_types)
    :param bool:Calculate matrix correlation(as in numpy coorcoef)"""
    if np.ndim(a) == 2:
        if correlation_matrix:
            b = b - b.mean(axis=1)[:, None]
            a = a - a.mean(axis=1)[:, None]

        num = np.dot(a, b.T)
        p1 =np.sqrt(np.sum(a**2,axis=1))[:,None] #[n,1]
        p2 = np.sqrt(np.sum(b ** 2, axis=1))[None, :] #[1,n]

        cosine_sim = num / (p1 * p2)
        #patristic_distances = (mutant_sequences.shape[1]-np.sum((mutant_sequences[:,None,:] == mutant_sequences[None,:,:]),axis=-1))/mutant_sequences.shape[1]
        #np.matmul(v[:,:,None,:],w[:,:,:,None])
        return cosine_sim
    else:
        if correlation_matrix:
            b = b - b.mean(axis=2)[:,:, None]
            a = a - a.mean(axis=2)[:,:, None]
        num = np.matmul(a[:,None], np.transpose(b,(0,2,1))[:,None])

        p1 = np.sqrt(np.sum(a ** 2, axis=2))[:, :,None]
        p2 = np.sqrt(np.sum(b ** 2, axis=2))[:,None, :]

        cosine_sim = num / (p1 * p2)


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

def calculate_similarity_matrix(array,batch_size=300,ksize=3):
    """Batched method to calculate the cosine similarity between the blosum encoded sequences"""

    print(array.shape)
    split_size = int(array.shape[0]/batch_size)
    splits = np.array_split(array,split_size)
    print("Generated {} splits".format(len(splits)))
    idx = list(range(len(splits)))
    overlapping_kmers = extract_windows_vectorized(splits[0],1,array.shape[1]-ksize,ksize,only_windows=True) #TODO:Might not be necessary
    for i in idx:
        curr_array = splits[i] #TODO: Calculate distance to itself for sanity check---> Plot correlation %ID and cosine sim
        rest_splits = splits.copy()
        del rest_splits[i]
        distances = []
        for r_j in rest_splits: #calculate distance among all kmers per sequence in the block (n, n_kmers,n_kmers)
            # kmers_i = curr_array[:,overlapping_kmers]
            # kmers_j = r_j[:,overlapping_kmers]
            # pairwise_comparison = (kmers_i[None,:] == kmers_j[:,None]).astype(int)
            cosine_sim = cosine_similarity(curr_array,r_j)
            print(cosine_sim[0][0])
            print("first sequence")
            print(curr_array[0][0])
            print("second sequence")
            print(r_j[0][1])
            exit()



    print(splits)
    exit()

def process_data(data,args):
    """
    :param pandas dataframe data: Contains Icore, Confidence_score and Rnk_EL
    """
    blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(args.aa_types, args.subs_matrix)
    print(blosum_array_dict)
    aa_dict = VegvisirUtils.aminoacid_names_dict(args.aa_types, zero_characters=["#"])
    epitopes = data[["Icore"]].values.tolist()
    epitopes = functools.reduce(operator.iconcat, epitopes, [])  # flatten list of lists
    epitopes_max_len = len(max(epitopes, key=len))
    epitopes_lens = np.array(list(map(len, epitopes)))

    #Pad the sequences
    epitopes = [list(seq.ljust(epitopes_max_len, "#")) for seq in epitopes]

    epitopes_array = np.array(epitopes)
    print(epitopes_array[0])
    epitopes_array_int = np.vectorize(aa_dict.get)(epitopes_array)
    epitopes_array_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(epitopes_array_int)
    calculate_similarity_matrix(epitopes_array_blosum)

    #comparison = np.char.add(epitopes_array [:,None], epitopes_array [None,:]) #a = np.array([["A","R","T","#"],["Y","M","T","P"],["I","R","T","#"]])

