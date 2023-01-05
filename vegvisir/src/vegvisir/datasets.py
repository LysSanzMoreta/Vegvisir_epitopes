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
    The dataset (Viruses_db_partitions.tsv) is organized as follows:
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

    The dataset (Viruses_predict_hla) is organized as follows:
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
    dataset_info_file = open("{}/{}/dataset_info.txt".format(storage_folder,args.dataset_name), 'w')

    sequence_column = ["Core","Icore"][1]
    score_column = ["Rnk_EL","target"][1]
    ###### READ THE DATASET CONTAINING THE NUMBER OF SUBJECTS #######################################

    data_partitions = pd.read_csv("{}/viral_dataset/Viruses_db_partitions.tsv".format(storage_folder,args.dataset_name),sep="\t")
    nnalign_input = data_partitions[[sequence_column,score_column,"training","partition"]]
    nnalign_input_train = nnalign_input.loc[nnalign_input['training'] == 1]
    nnalign_input_eval = nnalign_input.loc[nnalign_input['training'] == 0]
    nnalign_input_train = nnalign_input_train.drop_duplicates(sequence_column,keep="first")
    nnalign_input_eval = nnalign_input_eval.drop_duplicates(sequence_column, keep="first")
    nnalign_input_train.drop('training',inplace=True,axis=1)
    nnalign_input_eval.drop('training', inplace=True,axis=1)
    nnalign_input_eval.drop('partition', inplace=True, axis=1)
    nnalign_input_train.to_csv("{}/{}/viral_nnalign_input_train.tsv".format(storage_folder,args.dataset_name),sep="\t",index=False)
    nnalign_input_eval.to_csv("{}/{}/viral_nnalign_input_eval.tsv".format(storage_folder,args.dataset_name), sep="\t",index=False) #TODO: Header None?

    if args.run_nnalign:
        VegvisirNNalign.run_nnalign(args,storage_folder)
    ###### READ THE DATASET CONTAINING THE NUMBER OF SUBJECTS #######################################
    data_subjects = pd.read_csv("{}/{}/Viruses_predict_hla.csv".format(storage_folder,args.dataset_name),sep="\t")
    columns = ["Reference_data","Epitope_description","Epitope_organism_name","Parent_species_id","Parent_protein_accession",
               "Assay_qualitative_measure","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded",
               "Assay_response_frequency","MHC_allele_name","Icore","Rnk_EL"]
    data_subjects.columns = columns
    #Reattach partition, training ... information
    data = pd.merge(data_subjects,data_partitions[["Icore","partition","target","training"]], on='Icore', how='outer')
    #Clean missing data
    data = data.dropna(subset=['partition', 'target','training'],how="all")
    data = data.dropna(subset=["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"],how="all")
    #Group data by Icore
    data_a = data.groupby('Icore',as_index=False)[["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
    data_b = data.groupby('Icore',as_index=False)[["Rnk_EL"]].agg(lambda x: sum(list(x))/len(list(x)))
    data_part_info = data.groupby('Icore',as_index=False)[["partition","target","training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
    #Reattach missing info
    data_a = pd.merge(data_a,data_part_info, on='Icore', how='outer')
    nprefilter = data_a.shape[0]
    dataset_info_file.write("Initial data size is {} (with data points with missing information eliminated) \n".format(nprefilter))
    #Highlight: Grab only 9-mers
    data_a = data_a[data_a["Icore"].apply(lambda x: len(x) == 9)]
    nfiltered = data_a.shape[0]
    dataset_info_file.write("Filter 1: Filtered {} Icores whose length is different than 9. Remaining {} \n".format(nprefilter-nfiltered,nfiltered))
    #Highlight: Filter the low subject count only if all "negative"
    nprefilter = nfiltered
    data_a = data_a[(data_a["Assay_number_of_subjects_tested"] > 5) & (data_a["Assay_number_of_subjects_responded"].apply(lambda x: x >= 1))]
    nfiltered = data_a.shape[0]
    dataset_info_file.write("Filter 2: Filtered {} Icores with number of subjects lower than 5 and all negatives. Remaining {} \n".format(nprefilter-nfiltered,nfiltered))
    #max_number_subjects = data["Assay_number_of_subjects_tested"].max()
    data_a["confidence_score"] = data_a["Assay_number_of_subjects_responded"]/data_a["Assay_number_of_subjects_tested"]
    data_a["Rnk_EL"] =data_b["Rnk_EL"]
    data_a.fillna(0,inplace=True)
    #Highlight: Scale-standarize values #TODO: Do separately for train, eval and test
    #TODO: Why small negative values appear?
    data_a = VegvisirUtils.minmax_scale(data_a,"confidence_score")
    data_a = VegvisirUtils.minmax_scale(data_a,"Rnk_EL") #Likelihood rank
    data_b.fillna(0, inplace=True)
    #print(data_a.sort_values(by="confidence_score",ascending=True)[["confidence_score","target"]])
    data_a.loc[data_a["confidence_score"] <= 0.,"target"] = 0 #["target"] = 0. #Strict target reassignment
    #print(data_a.sort_values(by="confidence_score", ascending=True)[["confidence_score","target"]])
    data_a.loc[data_a["confidence_score"] > 0.,"target"] = 1.
    print(data_a["target"].value_counts())
    ndata = data_a.shape[0]
    fig, ax = plt.subplots(3, figsize=(7, 10))
    num_bins = 50
    ############LABELS #############
    freq, bins, patches = ax[0].hist(data_a["target"].to_numpy() , bins=2, density=True)
    ax[0].set_xlabel('Normalized confidence score (N_+ / Subjects)')
    ax[0].set_title(r'Histogram of targets/labels')
    # Annotate the bars.
    # Iterating over the bars one-by-one
    for bar in patches:
        # Using Matplotlib's annotate function and
        # passing the coordinates where the annotation shall be done
        # x-coordinate: bar.get_x() + bar.get_width() / 2
        # y-coordinate: bar.get_height()
        # free space to be left to make graph pleasing: (0, 8)
        # ha and va stand for the horizontal and vertical alignment
        n_data_bin = (bar.get_height()*ndata)/2
        ax[0].annotate(format(n_data_bin, '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
    #######CONFIDENCE SCORES
    #fig, ax = plt.subplots()
    #num_bins = 50
    ax[1].hist(data_a["confidence_score"].to_numpy() , num_bins, density=True)
    ax[1].set_xlabel('Normalized confidence score (N_+ / Subjects)')
    ax[1].set_title(r'Histogram of confidence scores')
    ##########RANK###################
    #fig, ax = plt.subplots()
    #num_bins = 50
    ax[2].hist(data_a["Rnk_EL"].to_numpy(), num_bins, density=True)
    ax[2].set_xlabel("Binding rank estimated by NetMHCpan-4.1")
    ax[2].set_title(r'Histogram of Rnk_EL scores')
    plt.ylabel("Counts")
    fig.tight_layout()
    plt.savefig("{}/{}/Viruses_histograms".format(storage_folder,args.dataset_name), dpi=300)
    plt.clf()

    exit()

    data = process_data(data_a,args,storage_folder)

    return data

def process_data(data,args,storage_folder,plot_blosum=False):
    """
    :param pandas dataframe data: Contains Icore, Confidence_score and Rnk_EL
    :param args
    :param storage_folder
    """
    blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(args.aa_types, args.subs_matrix)
    if plot_blosum:
        blosum_cosine = VegvisirUtils.cosine_similarity(blosum_array[1:, 1:], blosum_array[1:, 1:])
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
    ksize = 3 #TODO: manage in args
    if not os.path.exists("{}/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name)):
        print("Epitopes similarity matrices not existing, calculating ....")
        percent_identity_mean,cosine_similarity_mean,kmers_pid_similarity,kmers_cosine_similarity = VegvisirUtils.calculate_similarity_matrix(epitopes_array_blosum,epitopes_max_len,epitopes_mask,ksize=ksize)
        np.save("{}/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name), percent_identity_mean)
        np.save("{}/{}/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name), cosine_similarity_mean)
        np.save("{}/{}/kmers_pid_similarity_{}ksize.npy".format(storage_folder,args.dataset_name,ksize), kmers_pid_similarity)
        np.save("{}/{}/kmers_cosine_similarity_{}ksize.npy".format(storage_folder,args.dataset_name,ksize), kmers_cosine_similarity)

    else:
        print("Loading pre-calculated epitopes similarity matrices")
        percent_identity_mean = np.load("{}/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name))
        cosine_similarity_mean = np.load("{}/{}/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name))
        kmers_pid_similarity = np.load("{}/{}/kmers_pid_similarity_{}ksize.npy".format(storage_folder, args.dataset_name,ksize))
        kmers_cosine_similarity = np.load("{}/{}/kmers_cosine_similarity_{}ksize.npy".format(storage_folder, args.dataset_name,ksize))



