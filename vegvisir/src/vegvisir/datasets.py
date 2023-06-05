"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import json
import os,random
import time,datetime
import warnings
import dill
import pandas as pd
import operator,functools
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict,namedtuple
try:
    import umap
except:
    print("Could not import UMAP because of numpy incompatibility")
    pass
import hdbscan


import scipy
import seaborn as sns
import dataframe_image as dfi
import torch
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches
import vegvisir.nnalign as VegvisirNNalign
import vegvisir.utils as VegvisirUtils
import vegvisir.similarities as VegvisirSimilarities
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.plots as VegvisirPlots
import vegvisir.mutual_information as VegvisirMI
plt.style.use('ggplot')
DatasetInfo = namedtuple("DatasetInfo",["script_dir","storage_folder","data_array_raw","data_array_int","data_array_int_mask",
                                        "data_array_blosum_encoding","data_array_blosum_encoding_mask","data_array_onehot_encoding","data_array_blosum_norm","blosum",
                                        "n_data","seq_max_len","max_len","corrected_aa_types","input_dim","positional_weights","positional_weights_mask","percent_identity_mean","cosine_similarity_mean","kmers_pid_similarity","kmers_cosine_similarity","features_names"])
DatasetDivision = namedtuple("DatasetDivision",["all","all_mask","positives","positives_mask","negatives","negatives_mask","high_confidence_negatives","high_confidence_negatives_mask"])
SimilarityResults = namedtuple("SimilarityResults",["positional_weights","percent_identity_mean","cosine_similarity_mean","kmers_pid_similarity","kmers_cosine_similarity"])

def available_datasets():
    """Prints the available datasets"""
    datasets = {0:"viral_dataset",
                1:"viral_dataset2",
                2:"viral_dataset3",
                3:"viral_dataset4",
                4:"viral_dataset5",
                5:"viral_dataset6",
                6:"viral_dataset7"}
    return datasets

def select_dataset(dataset_name,script_dir,args,results_dir,update=True):
    """Selects from available datasets
    :param dataset_name: dataset of choice
    :param script_dir: Path from where the scriptis being executed
    :param update: If true it will download and update the most recent version of the dataset
    """
    func_dict = {"viral_dataset": viral_dataset,
                 "viral_dataset2":viral_dataset2,
                 "viral_dataset3":viral_dataset3,
                 "viral_dataset4":viral_dataset4,
                 "viral_dataset5":viral_dataset5,
                 "viral_dataset6": viral_dataset6,
                 "viral_dataset7":viral_dataset7}
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) #finds the /data folder of the repository

    dataset_load_fx = lambda f,dataset_name,current_path,storage_folder,args,results_dir,update: lambda dataset_name,current_path,storage_folder,args,results_dir,update: f(dataset_name,current_path,storage_folder,args,results_dir,update)
    data_load_function = dataset_load_fx(func_dict[dataset_name],dataset_name,script_dir,storage_folder,args,results_dir,update)
    dataset = data_load_function(dataset_name,script_dir,storage_folder,args,results_dir,update)
    print("Data retrieved")

    return dataset

def viral_dataset(dataset_name,current_path,storage_folder,args,results_dir,update): #TODO: Remove?
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
    nnalign_input_train = nnalign_input.loc[nnalign_input['training'] == 1]
    nnalign_input_eval = nnalign_input.loc[nnalign_input['training'] == 0]
    nnalign_input_train = nnalign_input_train.drop_duplicates(sequence_column,keep="first")
    nnalign_input_eval = nnalign_input_eval.drop_duplicates(sequence_column, keep="first")
    nnalign_input_train.drop('training',inplace=True,axis=1)
    nnalign_input_eval.drop('training', inplace=True,axis=1)
    nnalign_input_eval.drop('partition', inplace=True, axis=1)
    nnalign_input_train.to_csv("{}/{}/viral_nnalign_input_train.tsv".format(args.dataset_name,storage_folder),sep="\t",index=False)
    nnalign_input_eval.to_csv("{}/{}/viral_nnalign_input_eval.tsv".format(args.dataset_name,storage_folder), sep="\t",index=False) #TODO: Header None?

    if args.run_nnalign:
        VegvisirNNalign.run_nnalign(args,storage_folder)

def viral_dataset2(dataset_name,script_dir,storage_folder,args,results_dir,update): #TODO: Remove?
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
    #number_of_papers_positive: number of papers where the peptide-MHC was reported to have a positive interaction with the TCR.
    #number_of_papers_negative: number of papers where the peptide-MHC was reported to have a negative interaction with the TCR.
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
      :param pandas dataframe: Results pandas dataframe with the following structure
          Icore:Interaction peptide core
          immunodominance_score: Number of + / Number of tested
          onfidence_score_scaled: Number of + / Number of tested ---> Minmax scaled to 0-1 range
          training: True assign data point to train , else assign to Test
          partition: Indicates partition assignment within 5-fold cross validation
          Rnk_EL: Average rank score per peptide by NetMHC ---> Normalized to 0-1
    """
    dataset_info_file = open("{}/{}/dataset_info.txt".format(storage_folder,args.dataset_name), 'w')

    sequence_column = ["Core","Icore"][1]
    score_column = ["Rnk_EL","target"][1]
    filters_dict,analysis_mode = select_filters(args)
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
    #Group data by Icore
    data_a = data.groupby('Icore',as_index=False)[["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
    data_b = data.groupby('Icore',as_index=False)[["Rnk_EL"]].agg(lambda x: sum(list(x))/len(list(x)))
    data_part_info = data.groupby('Icore',as_index=False)[["partition","target","training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
    #Reattach info on training
    data_a = pd.merge(data_a,data_part_info, on='Icore', how='outer')
    nprefilter = data_a.shape[0]
    dataset_info_file.write("Pre-filtering data size {} \n".format(nprefilter))
    #Clean missing data
    nprefilter = data_a.shape[0]
    data_a = data_a.dropna(subset=["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"],how="all")
    data_a = data_a.dropna(subset=['partition', 'target','training'],how="all")
    nfiltered = data_a.shape[0]
    dataset_info_file.write("Filter 1: Missing data (partition, target,training, n_subjects). Drops {} data points, remaining {} \n".format(nprefilter-nfiltered,nfiltered))
    #Highlight: Grab only 9-mers
    data_a = data_a[data_a["Icore"].apply(lambda x: len(x) == 9)]
    nfiltered = data_a.shape[0]
    dataset_info_file.write("Filter 2: Icores whose length is different than 9. Drops {} data points, remaining {} \n".format(nprefilter-nfiltered,nfiltered))
    #Highlight: Filter the points with low subject count and only keep if all "negative"
    nprefilter = data_a.shape[0]
    data_a = data_a[(data_a["Assay_number_of_subjects_tested"] > 10)] #& (data_a["Assay_number_of_subjects_responded"].apply(lambda x: x >= 1))
    nfiltered = data_a.shape[0]
    dataset_info_file.write("Filter 3: Icores with number of subjects lower than 10. Drops {} data points, remaining {} \n".format(nprefilter-nfiltered,nfiltered))
    #max_number_subjects = data["Assay_number_of_subjects_tested"].max()
    data_a["immunodominance_score"] = data_a["Assay_number_of_subjects_responded"]/data_a["Assay_number_of_subjects_tested"]
    data_a["Rnk_EL"] =data_b["Rnk_EL"]
    data_a.fillna(0,inplace=True)
    #Highlight: Scale-standarize values . This is done here for visualization purposes, it is done afterwards separately for train, eval and test
    data_a = VegvisirUtils.minmax_scale(data_a,column_name ="immunodominance_score",suffix="_scaled")
    data_a = VegvisirUtils.minmax_scale(data_a,column_name="Rnk_EL",suffix="_scaled") #Likelihood rank
    data_b.fillna(0, inplace=True)
    # print(data_a["target"].value_counts())
    # print(data_a.sort_values(by="immunodominance_score",ascending=True)[["immunodominance_score","target"]])
    #Highlight: Strict target reassignment
    data_a.loc[data_a["immunodominance_score_scaled"] <= 0.,"target_corrected"] = 0 #["target"] = 0. #Strict target reassignment
    #print(data_a.sort_values(by="immunodominance_score", ascending=True)[["immunodominance_score","target"]])
    data_a.loc[data_a["immunodominance_score_scaled"] > 0.,"target_corrected"] = 1.
    # print(data_a["target"].value_counts())
    # print("--------------------")
    # print(data_a["partition"].value_counts())
    # print("--------------------")
    # print(data_a["training"].value_counts())
    ndata = data_a.shape[0]
    fig, ax = plt.subplots(2,2, figsize=(7, 10))
    num_bins = 50
    ############LABELS #############
    freq, bins, patches = ax[0][0].hist(data_a["target"].to_numpy() , bins=2, density=True)
    ax[0][0].set_xlabel('Target/Label (0: Non-binder, 1: Binder)')
    ax[0][0].set_title(r'Histogram of targets/labels')
    ax[0][0].xaxis.set_ticks([0.25,0.75])
    ax[0][0].set_xticklabels([0,1])
    # Annotate the bars.
    for bar in patches: #iterate over the bars
        n_data_bin = (bar.get_height()*ndata)/2
        ax[0][0].annotate(format(n_data_bin, '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
    ############LABELS CORRECTED #############
    freq, bins, patches = ax[0][1].hist(data_a["target_corrected"].to_numpy() , bins=2, density=True)
    ax[0][1].set_xlabel('Target/Label (0: Non-binder, 1: Binder)')
    ax[0][1].set_title(r'Histogram of re-assigned targets/labels')
    ax[0][1].xaxis.set_ticks([0.25,0.75])
    ax[0][1].set_xticklabels([0,1])
    # Annotate the bars.
    for bar in patches: #iterate over the bars
        n_data_bin = (bar.get_height()*ndata)/2
        ax[0][1].annotate(format(n_data_bin, '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
    #######immunodominance scoreS
    ax[1][0].hist(data_a["immunodominance_score_scaled"].to_numpy() , num_bins, density=True)
    ax[1][0].set_xlabel('Minmax scaled immunodominance score \n  (N_+ / Subjects)')
    ax[1][0].set_title(r'Histogram of immunodominance scores')
    ##########RANK###################
    ax[1][1].hist(data_a["Rnk_EL"].to_numpy(), num_bins, density=True)
    ax[1][1].set_xlabel("Binding rank estimated by NetMHCpan-4.1")
    ax[1][1].set_title(r'Histogram of Rnk_EL scores')
    plt.ylabel("Counts")
    fig.tight_layout()
    plt.savefig("{}/{}/Viruses_histograms".format(storage_folder,args.dataset_name), dpi=300)
    plt.clf()
    data_info = process_data(data_a,args,storage_folder,script_dir,analysis_mode,filters_dict)
    return data_info

def select_filters(args):
    filters_dict = {"filter_kmers":[False,9,args.sequence_type], #Icore_non_anchor #Highlight: Remmeber to use 8!!
                    "group_alleles":[True],
                    "filter_alleles":[False], #if True keeps the most common allele
                    "filter_ntested":[False,10],
                    "filter_lowconfidence":[False],
                    "corrected_immunodominance_score":[False,10]}

    allele_selection = {True:"same_allele",False:"diff_allele"}
    len_selection = {True:"same_len",False:"diff_len"}
    if filters_dict["filter_kmers"][0]:
        analysis_mode = "{}/{}/{}mers".format(allele_selection[filters_dict["filter_alleles"][0]],len_selection[filters_dict["filter_kmers"][0]],filters_dict["filter_kmers"][1])
    else:
        analysis_mode = "{}/{}".format(allele_selection[filters_dict["filter_alleles"][0]],len_selection[filters_dict["filter_kmers"][0]])

    return filters_dict,analysis_mode

def group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file):
    """Filters, groups and prepares the files from the viral_dataset*() functions"""
    if filters_dict["filter_ntested"][0]:
        # Highlight: Filter the points with low subject count and only keep if all "negative"
        threshold = filters_dict["filter_ntested"][1]
        nprefilter = data.shape[0]
        data = data[(data["Assay_number_of_subjects_tested"] > threshold)]
        nfiltered = data.shape[0]
        dataset_info_file.write("Filter 1: Icores with number of subjects lower than {}. Drops {} data points, remaining {} \n".format(threshold,nprefilter - nfiltered, nfiltered))

    data["immunodominance_score"] = data["Assay_number_of_subjects_responded"] / data["Assay_number_of_subjects_tested"]

    data = data.fillna({"immunodominance_score":0})

    if filters_dict["corrected_immunodominance_score"][0]:
        treshold = filters_dict["corrected_immunodominance_score"][1]
        data.loc[(data["Assay_number_of_subjects_tested"] < treshold) | (data["Assay_number_of_subjects_responded"] == 0), "immunodominance_score"] = 0.01

    # Highlight: Scale-standarize values . This is done here for visualization purposes, it is done afterwards separately for train, eval and test
    data = VegvisirUtils.minmax_scale(data, column_name="immunodominance_score", suffix="_scaled")

    if filters_dict["filter_kmers"][0]:
        #Highlight: Grab only k-mers
        use_column = filters_dict["filter_kmers"][2]
        kmer_size = filters_dict["filter_kmers"][1]
        nprefilter = data.shape[0]
        data = data[data[use_column].apply(lambda x: len(x) == kmer_size)]
        nfiltered = data.shape[0]
        dataset_info_file.write("Filter 2: {} whose length is different than 9. Drops {} data points, remaining {} \n".format(use_column,kmer_size,nprefilter-nfiltered,nfiltered))

    #Highlight: Strict target reassignment
    data.loc[data["immunodominance_score_scaled"] <= 0.,"target_corrected"] = 0 #["target"] = 0. #Strict target reassignment
    #print(data_a.sort_values(by="immunodominance_score", ascending=True)[["immunodominance_score","target"]])
    data.loc[data["immunodominance_score_scaled"] > 0.,"target_corrected"] = 1.

    #Highlight: Filter data points with low confidence (!= 0, 1)
    if filters_dict["filter_lowconfidence"][0]:
        nprefilter = data.shape[0]
        data = data[data["immunodominance_score"].isin([0.,1.])]
        nfiltered = data.shape[0]
        dataset_info_file.write("Filter 3: Remove data points with low immunodominance score. Drops {} data points, remaining {} \n".format(nprefilter - nfiltered, nfiltered))

    #Highlight: Annotate which data points have low confidence
    data = set_confidence_score(data)
    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])

    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t")

    VegvisirPlots.plot_data_information(data, filters_dict, storage_folder, args, name_suffix)
    #Highlight: Prep data to run in NNalign
    if args.run_nnalign:
        prepare_nnalign(args,storage_folder,data,[filters_dict["filter_kmers"][2],"target_corrected","partition"])

    return data

def viral_dataset3(dataset_name,script_dir,storage_folder,args,results_dir,update):
    """
    ####################
    #HEADER DESCRIPTIONS#
    ####################
    allele
    Icore: Interaction core. This is the sequence of the binding core including eventual insertions of deletions (derived from the prediction of the likelihood of binding of the peptide to the reported MHC-I with NetMHCpan-4.1).
    Number of Subjects Tested: number of papers where the peptide-MHC was reported to have a positive interaction with the TCR.
    Number of Subjects Responded
    target: target value (1: immunogenic/positive, 0:non-immunogenic/negative).
    training
    Icore_non_anchor: Peptide without the amino acids that are anchored to the MHC
    partition

    return
          :param pandas dataframe: Results pandas dataframe with the following structure:
                  Icore:Interaction peptide core
                  immunodominance_score: Number of + / Number of tested. Except for when the number of tested subjects is lower than 10 and all the subjects where negative, the conficence score is lowered to 0.1
                  immunodominance_score_scaled: Number of + / Number of tested ---> Minmax scaled to 0-1 range (only for visualization purposed, this step is re-done for each partition to avoid data leakage from test to train
                  training: True assign data point to train , else assign to Test (given)
                  partition: Indicates partition assignment within 5-fold cross validation (given)
                  target: Pre-assigned target(given)
                  target_corrected: Corrected target based on the immunodominance score, it is negative (0) only and only if the number of tested subjects is higher than 10 and all of them tested negative
            """
    dataset_info_file = open("{}/dataset_info.txt".format(results_dir), 'a+')
    data = pd.read_csv("{}/{}/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    data.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    data = data.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)
    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = data.value_counts("allele").index[0] #allele with most conserved positions HLA-B0707, the most common allele here is also ok

    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]


    if filters_dict["group_alleles"][0]:
        # Group data by Icore, therefore the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
    else:
        allele_counts_dict = data["allele"].value_counts().to_dict()
        allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys()))))) #TODO: Replace with allele encoding based on sequential information
        data["allele_encoded"] = data["allele"]
        data.replace({"allele_encoded": allele_dict},inplace=True)
    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file)

    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset4(dataset_name,script_dir,storage_folder,args,results_dir,update):
    """
    ####################
    #HEADER DESCRIPTIONS#
    ####################
    allele: MHC allele
    Icore: Interaction core. This is the sequence of the binding core including eventual insertions of deletions (derived from the prediction of the likelihood of binding of the peptide to the reported MHC-I with NetMHCpan-4.1).
    Number of Subjects Tested: number of papers where the peptide-MHC was reported to have a positive interaction with the TCR.
    Number of Subjects Responded
    target: target value (1: immunogenic/positive, 0:non-immunogenic/negative).
    training: True, include in training, False include in test
    Icore_non_anchor: Peptide without the amino acids that are anchored to the MHC
    partition: partition number

    return
          :param pandas dataframe: Results pandas dataframe with the following structure:
                  Icore:Interaction peptide core
                  immunodominance_score: Number of + / Number of tested. Except for when the number of tested subjects is lower than 10 and all the subjects where negative, the conficence score is lowered to 0.1
                  immunodominance_score_scaled: Number of + / Number of tested ---> Minmax scaled to 0-1 range (only for visualization purposed, this step is re-done for each partition to avoid data leakage from test to train
                  training: True assign data point to train , else assign to Test (given)
                  partition: Indicates partition assignment within 5-fold cross validation (given)
                  target: Pre-assigned target(given)
                  target_corrected: Corrected target based on the immunodominance score, it is negative (0) only and only if the number of tested subjects is higher than 10 and all of them tested negative
            """
    dataset_info_file = open("{}/dataset_info.txt".format(results_dir), 'a+')
    data_features = pd.read_csv("{}/{}/dataset_all_features.tsv".format(storage_folder,args.dataset_name),sep="\s+",index_col=0)
    data_partitions = pd.read_csv("{}/viral_dataset3/dataset_target.tsv".format(storage_folder),sep = "\t",index_col=0)
    data_partitions.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    data_partitions = data_partitions[["Icore","Icore_non_anchor","allele","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","partition","target","training"]]
    data_features = data_features[["Icore","allele","Pred_netstab","prot_inst_index","prot_median_iupred_score_long","prot_molar_excoef_cys_cys_bond","prot_p[q3_E]_netsurfp","prot_p[q3_C]_netsurfp","prot_rsa_netsurfp"]]
    features_names = data_features.columns.tolist()[2:]

    data = pd.merge(data_features,data_partitions, on=['Icore',"allele"], how='outer')

    data = data.dropna(subset=["Icore_non_anchor","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training","Pred_netstab"]).reset_index(drop=True)

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    if filters_dict["group_alleles"][0]:
        # Group data by Icore
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        data_b = data.groupby('Icore', as_index=False)[features_names].agg(lambda x: sum(list(x)) / len(list(x)))
        data_c = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
        data = pd.merge(data, data_c, on='Icore', how='outer')
    else:
        allele_counts_dict = data["allele"].value_counts().to_dict()
        allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
        data["allele_encoded"] = data["allele"]
        data.replace({"allele_encoded": allele_dict},inplace=True)
        #features_names.append("allele_encoded")

    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file)

    name_suffix = "__".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
    VegvisirPlots.plot_features_histogram(data,features_names,"{}/{}".format(storage_folder,args.dataset_name),name_suffix)
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict,features_names=features_names)

    return data_info

def viral_dataset5(dataset_name,script_dir,storage_folder,args,results_dir,update):
    """
    Contains "artificial" or fake negative epitopes solely in the test dataset. The artificial negatives can be identified by having a inmmunodominance score of
    HEADER descriptions:
    allele: MHC allele
    Icore: Interaction peptide core
    Number of Subjects Tested
    Number of Subjects Responded
    target
    training
    Icore_non_anchor
    partition
    confidence_score
    immunodominance
    Length
    Rnk_EL
    """
    dataset_info_file = open("{}/dataset_info.txt".format(results_dir), 'a+')
    data_partitions = pd.read_csv("{}/{}/dataset_target_correction_artificial_negatives.tsv".format(storage_folder,args.dataset_name),sep = "\t")

    data_partitions.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition","confidence_score","immunodominance","Length","Rnk_EL"]
    data = data_partitions[["Icore","Icore_non_anchor","allele","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","partition","target","training"]]
    #mask = data["Assay_number_of_subjects_tested"] == 0
    #Highlight: Dealing with the artificial data points
    mask = data["Assay_number_of_subjects_tested"] == 0
    data.loc[mask,"training"] = 0
    data = data.copy() #needed to avoid funky column re-assignation warning errors
    data.loc[:,'training'] = data.loc[:,'training'].replace({1: True, 0: False})
    data = data.dropna(subset=["Icore_non_anchor","training"]).reset_index(drop=True)

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    if filters_dict["group_alleles"][0]:
        # Group data by Icore only, therefore the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
    else:
        allele_counts_dict = data["allele"].value_counts().to_dict()
        allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
        data["allele_encoded"] = data["allele"]
        data.replace({"allele_encoded": allele_dict},inplace=True)

    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file)

    mask2 = data["Assay_number_of_subjects_tested"] == 0

    warnings.warn("Setting low confidence score to the artificial negatives in the test dataset")
    data.loc[mask2,"confidence_score"] = 0.6
    data.loc[mask2,"immunodominance_score"] = np.nan

    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)



    return data_info

def viral_dataset6(dataset_name,script_dir,storage_folder,args,results_dir,update):
    """
    Collects IEDB data and creates artificially generated epitopes. The artificial epitopes are designed by randomly chopping viral proteins onto peptides and then checking binding to MHC with NetMHC-pan

    ####################
    #HEADER DESCRIPTIONS#
    ####################
    allele
    Icore: Interaction core. This is the sequence of the binding core including eventual insertions of deletions (derived from the prediction of the likelihood of binding of the peptide to the reported MHC-I with NetMHCpan-4.1).
    Number of Subjects Tested: number of papers where the peptide-MHC was reported to have a positive interaction with the TCR.
    Number of Subjects Responded
    target: target value (1: immunogenic/positive, 0:non-immunogenic/negative).
    training
    Icore_non_anchor: Peptide without the amino acids that are anchored to the MHC
    partition

    return
          :param pandas dataframe: Results pandas dataframe with the following structure:
                  Icore:Interaction peptide core
                  allele: MHC allele
                  immunodominance_score: Number of + / Number of tested. Except for when the number of tested subjects is lower than 10 and all the subjects where negative, the conficence score is lowered to 0.1
                  immunodominance_score_scaled: Number of + / Number of tested ---> Minmax scaled to 0-1 range (only for visualization purposed, this step is re-done for each partition to avoid data leakage from test to train
                  training: True assign data point to train , else assign to Test (given)
                  partition: Indicates partition assignment within 5-fold cross validation (given)
                  target: Pre-assigned target (given)
                  target_corrected: Corrected target based on the immunodominance score, it is negative (0) only and only if the number of tested subjects is higher than 10 and all of them tested negative
    """

    dataset_info_file = open("{}/dataset_info.txt".format(results_dir), 'a+')
    data_labelled = pd.read_csv("{}/{}/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    data_labelled.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    data_labelled = data_labelled.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)
    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    data_artificial = pd.read_csv("{}/{}/dataset_artificial_peptides_from_proteins_partitioned_hla.tsv".format(storage_folder,args.dataset_name),sep = "\s+") #Highlight: The sequences from the labelled dataset have been filtered for some reason
    data_artificial.columns = ["Icore", "target", "partition", "source","allele"]
    data_artificial = data_artificial[(data_artificial["source"] == "artificial")]
    data_artificial = data_artificial.sample(n=5000)

    data = data_labelled.merge(data_artificial, on=['Icore', 'allele'], how='outer',suffixes=('_labelled', '_artificial'))

    data = data.drop(["target_artificial","partition_labelled"],axis=1)
    data = data.rename(columns={"target_labelled": "target","partition_artificial":"partition"})
    data.loc[(data["source"] == "artificial"), "target"] = 2

    #Filter peptides/Icores with unknown aminoacids "X"
    data = data[~data['Icore'].str.contains("X")]

    # anchors_per_allele = pd.read_csv("{}/anchor_info_content/anchors_per_allele_average.txt".format(storage_folder),sep="\s+",header=0)
    # anchors_per_allele=anchors_per_allele[anchors_per_allele[["allele"]].isin(data_allele_types).any(axis=1)]
    # allele_anchors_dict = dict(zip(anchors_per_allele["allele"],anchors_per_allele["anchors"]))
    allele_counts = data.value_counts("allele")
    most_common_allele = allele_counts.index[0] #allele with most conserved positions HLA-B0707, the most common allele here is also ok
    #data_allele_types = allele_counts.index.tolist() #alleles present in this dataset
    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        print("Grouping alleles")
        # Group data by Icore, therefore, the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
           # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
    else:
        allele_counts_dict = data["allele"].value_counts().to_dict()
        allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
        data["allele_encoded"] = data["allele"]
        data.replace({"allele_encoded": allele_dict},inplace=True)

    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file)
    data.loc[(data["target"] == 2), "target_corrected"] = 2
    data.loc[(data["target"] == 2), "confidence_score"] = 0
    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset7(dataset_name,script_dir,storage_folder,args,results_dir,update):
    """
    ####################
    #HEADER DESCRIPTIONS#
    ####################
    allele
    Icore: Interaction core. This is the sequence of the binding core including eventual insertions of deletions (derived from the prediction of the likelihood of binding of the peptide to the reported MHC-I with NetMHCpan-4.1).
    Number of Subjects Tested: number of papers where the peptide-MHC was reported to have a positive interaction with the TCR.
    Number of Subjects Responded
    target: target value (1: immunogenic/positive, 0:non-immunogenic/negative).
    training
    Icore_non_anchor: Peptide without the amino acids that are anchored to the MHC
    partition
    Of: The starting position of the Core within the Peptide (if > 0, the method predicts a N-terminal protrusion) (derived from the prediction with NetMHCpan-4.1).
    Gp: Position of the deletion, if any (derived from the prediction with NetMHCpan-4.1).
    Gl: Length of the deletion, if any (derived from the prediction with NetMHCpan-4.1).
    Ip: Position of the insertion, if any (derived from the prediction with NetMHCpan-4.1).
    Il: Length of the insertion, if any (derived from the prediction with NetMHCpan-4.1).

    return
          :param pandas dataframe: Results pandas dataframe with the following structure:
                  Icore:Interaction peptide core
                  immunodominance_score: Number of + / Number of tested. Except for when the number of tested subjects is lower than 10 and all the subjects where negative, the conficence score is lowered to 0.1
                  immunodominance_score_scaled: Number of + / Number of tested ---> Minmax scaled to 0-1 range (only for visualization purposed, this step is re-done for each partition to avoid data leakage from test to train
                  training: True assign data point to train , else assign to Test (given)
                  partition: Indicates partition assignment within 5-fold cross validation (given)
                  target: Pre-assigned target(given)
                  target_corrected: Corrected target based on the immunodominance score, it is negative (0) only and only if the number of tested subjects is higher than 10 and all of them tested negative
            """
    dataset_info_file = open("{}/dataset_info.txt".format(results_dir), 'a+')
    new_partitions = pd.read_csv("{}/{}/Viruses_db_partitions_notest.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    #new_partitions.columns = ["Icore","allele","Core","Of","Gp","Gl","Ip","Il","Rnk_EL","org_id","uniprot_id","target","start_prot","Icore_non_anchor","partition"]


    data = pd.read_csv("{}/{}/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    data.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]

    data = data.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)

    #Highlight: Replace the training and partition columns for the new ones

    data = data.merge(new_partitions, on=['Icore', 'allele'], how='left',suffixes=('_old', '_new'))
    data = data.loc[:, ~data.columns.str.endswith('_old') | (data.columns == 'partition_old')  ] #remove all columns ending with _old
    data = data.rename(columns={"Icore_non_anchor_new": "Icore_non_anchor", "target_new": "target","partition_new":"partition"})
    #TODO: Rename columns with old vs new partition

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = data.value_counts("allele").index[0] #allele with most conserved positions HLA-B0707, the most common allele here is also ok

    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]


    if filters_dict["group_alleles"][0]:
        # Group data by Icore, therefore the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition","partition_old", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
    else:
        allele_counts_dict = data["allele"].value_counts().to_dict()
        allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys()))))) #TODO: Replace with allele encoding based on sequential information
        data["allele_encoded"] = data["allele"]
        data.replace({"allele_encoded": allele_dict},inplace=True)



    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file)

    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info


def data_class_division(array,array_mask,idx,labels,confidence_scores):
    """

    :param array: epitopes_array_int, apitopes_array_blosum, epitopes_array_blosum_norm_group, epitopes_array_aa_group
    :param idx: training or test idx
    :return:
    """
    #TODO: not indexing by train or test

    labels_ = labels[idx]
    confidence_scores_ = confidence_scores[idx]


    array_ = array[idx]
    mask_ = array_mask[idx]


    positives_arr = array_[labels_ == 1]
    positives_arr_mask = mask_[labels_ == 1]
    negatives_arr = array_[labels_ == 0]
    negatives_arr_mask = mask_[labels_ == 0]
    high_conf_negatives_arr = array_[(confidence_scores_ > 0.6)&(labels_ == 0)]
    high_conf_negatives_arr_mask = mask_[(confidence_scores_ > 0.6)&(labels_ == 0)]


    data_subdivision = DatasetDivision(all = array_,
                                       all_mask = mask_,
                                       positives=positives_arr,
                                       positives_mask=positives_arr_mask,
                                       negatives=negatives_arr,
                                       negatives_mask=negatives_arr_mask,
                                       high_confidence_negatives=high_conf_negatives_arr,
                                       high_confidence_negatives_mask=high_conf_negatives_arr_mask)
    return data_subdivision
def build_exploration_folders(args,storage_folder,filters_dict):

    for mode in ["All","Train", "Test"]:
        VegvisirUtils.folders("all","{}/{}/similarities/{}/{}/diff_allele/same_len/{}mers/neighbours1/".format(storage_folder,args.dataset_name,args.sequence_type, mode,filters_dict["filter_kmers"][1]))
        VegvisirUtils.folders("positives","{}/{}/similarities/{}/{}/diff_allele/same_len/{}mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode,filters_dict["filter_kmers"][1]),overwrite=False)
        VegvisirUtils.folders("negatives","{}/{}/similarities/{}/{}/diff_allele/same_len/{}mers/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode,filters_dict["filter_kmers"][1]),overwrite=False)
        VegvisirUtils.folders("highconfnegatives","{}/{}/similarities/{}/{}/diff_allele/same_len/{}mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode,filters_dict["filter_kmers"][1]),overwrite=False)


        VegvisirUtils.folders("all","{}/{}/similarities/{}/{}/diff_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode))
        VegvisirUtils.folders("positives","{}/{}/similarities/{}/{}/diff_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
        VegvisirUtils.folders("negatives","{}/{}/similarities/{}/{}/diff_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
        VegvisirUtils.folders("highconfnegatives","{}/{}/similarities/{}/{}/diff_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode),overwrite=False)


        VegvisirUtils.folders("all","{}/{}/similarities/{}/{}/same_allele/same_len/{}mers/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode,filters_dict["filter_kmers"][1]))
        VegvisirUtils.folders("positives","{}/{}/similarities/{}/{}/same_allele/same_len/{}mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode,filters_dict["filter_kmers"][1]),overwrite=False)
        VegvisirUtils.folders("negatives","{}/{}/similarities/{}/{}/same_allele/same_len/{}mers/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode,filters_dict["filter_kmers"][1]),overwrite=False)
        VegvisirUtils.folders("highconfnegatives","{}/{}/similarities/{}/{}/same_allele/same_len/{}mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode,filters_dict["filter_kmers"][1]),overwrite=False)

        VegvisirUtils.folders("all","{}/{}/similarities/{}/{}/same_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode))
        VegvisirUtils.folders("positives","{}/{}/similarities/{}/{}/same_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
        VegvisirUtils.folders("negatives","{}/{}/similarities/{}/{}/same_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
        VegvisirUtils.folders("highconfnegatives","{}/{}/similarities/{}/{}/same_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)

def sample_datapoints_mi(a,b):

    longest, shortest = [(0, 1) if a.shape[0] >b.shape[0] else (1, 0)][0]
    dict_counts = {0: a.shape[0],1: b.shape[0]}
    idx_longest = np.arange(dict_counts[longest])
    idx_sample = np.array(random.sample(range(dict_counts[longest]), dict_counts[shortest]))
    idx_sample = np.sort(idx_sample,axis=0)
    idx_sample = (idx_longest[..., None] == idx_sample).any(-1)
    return idx_sample,dict_counts[longest]
def data_exploration(data,epitopes_array_blosum,epitopes_array_int,epitopes_array_mask,aa_dict,aa_list,blosum_norm,seq_max_len,storage_folder,args,corrected_aa_types,analysis_mode,filters_dict):

    if not os.path.exists("{}/{}/similarities/{}".format(storage_folder,args.dataset_name,args.sequence_type)):
        build_exploration_folders(args, storage_folder,filters_dict)
        #/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset3/similarities
    else:
        print("Folder structure existing")
    plot_mi,plot_frequencies,plot_cosine_similarity = False,False,False

    #Highlight: Encode amino acid by chemical group
    aa_groups_colors_dict, aa_groups_dict, groups_names_colors_dict,aa_by_groups_dict = VegvisirUtils.aminoacids_groups(aa_dict)
    aa_groups = len(groups_names_colors_dict.keys())
    epitopes_array_int_group = np.vectorize(aa_groups_dict.get)(epitopes_array_int)
    #Highlight: Encode amino acid by blosum norm group
    blosum_norm_sorted_idx = np.argsort(blosum_norm)
    blosum_norm_sorted = blosum_norm[blosum_norm_sorted_idx]
    amino_acids_sorted = np.array(aa_list)[blosum_norm_sorted_idx]
    blosum_norm_sorted = np.concatenate([amino_acids_sorted[:,None],blosum_norm_sorted[:,None]],axis=1)
    blosum_norm_groups = np.array_split(blosum_norm_sorted,6)
    blosum_norm_groups_dict = dict.fromkeys(aa_list)
    for group_idx,group_array in enumerate(blosum_norm_groups):
        aa_group = group_array[:,0]
        for aa in aa_group:
            blosum_norm_groups_dict[aa] = group_idx
    epitopes_array_blosum_norm_group = np.vectorize(blosum_norm_groups_dict.get)(epitopes_array_int)

    ksize = 3
    labels_arr = np.array(data[["target_corrected"]].values.tolist()).squeeze()
    training = data[["training"]].values.tolist()
    training = np.array(training).squeeze(-1)
    confidence_scores = np.array(data["confidence_score"].values.tolist())

    epitopes_array_int_division_train = data_class_division(epitopes_array_int,
                                                                         epitopes_array_mask, training, labels_arr,
                                                                         confidence_scores)
    epitopes_array_int_division_test = data_class_division(epitopes_array_int, epitopes_array_mask,
                                                                        np.invert(training), labels_arr,
                                                                        confidence_scores)
    epitopes_array_blosum_division_train = data_class_division(epitopes_array_blosum,
                                                           epitopes_array_mask, training, labels_arr,
                                                           confidence_scores)
    epitopes_array_blosum_division_test = data_class_division(epitopes_array_blosum, epitopes_array_mask,
                                                          np.invert(training), labels_arr,
                                                          confidence_scores)

    epitopes_array_blosum_norm_group_divison_train = data_class_division(epitopes_array_blosum_norm_group,epitopes_array_mask,training,labels_arr,confidence_scores)
    epitopes_array_blosum_norm_group_divison_test = data_class_division(epitopes_array_blosum_norm_group,epitopes_array_mask,np.invert(training),labels_arr,confidence_scores)

    epitopes_array_int_group_divison_train = data_class_division(epitopes_array_int_group,epitopes_array_mask,training,labels_arr,confidence_scores)
    epitopes_array_int_group_divison_test = data_class_division(epitopes_array_int_group,epitopes_array_mask,np.invert(training),labels_arr,confidence_scores)


    if plot_mi:
        warnings.warn("Calculating Mutual information: If the dataset is unbalanced , then the longest dataset will be subsampled to match the smaller dataset")
        #Highlight: Mutual informaton calculated using raw amino acids

        train_idx_select,train_longest_array = sample_datapoints_mi(epitopes_array_int_division_train.positives,epitopes_array_int_division_train.negatives)
        test_idx_select,test_longest_array = sample_datapoints_mi(epitopes_array_int_division_test.positives,epitopes_array_int_division_test.negatives)
        #Highlight: Train
        VegvisirMI.calculate_mi(epitopes_array_int_division_train.all,epitopes_array_blosum_norm_group_divison_train.all_mask,
                                aa_groups,seq_max_len,"TrainAll_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode))

        if epitopes_array_int_division_train.negatives.shape[0] == train_longest_array:
            VegvisirMI.calculate_mi(epitopes_array_int_division_train.positives,epitopes_array_blosum_norm_group_divison_train.positives_mask,
                                    aa_groups,seq_max_len,"TrainPositives_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_int_division_train.negatives[train_idx_select],epitopes_array_blosum_norm_group_divison_train.negatives_mask,
                                    aa_groups,seq_max_len,"TrainNegatives_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))
        else:
            VegvisirMI.calculate_mi(epitopes_array_int_division_train.positives[train_idx_select],epitopes_array_blosum_norm_group_divison_train.positives_mask,
                                    aa_groups,seq_max_len,"TrainPositives_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_int_division_train.negatives,epitopes_array_blosum_norm_group_divison_train.negatives_mask,
                                    aa_groups,seq_max_len,"TrainNegatives_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))

        VegvisirMI.calculate_mi(epitopes_array_int_division_train.high_confidence_negatives,epitopes_array_blosum_norm_group_divison_train.high_confidence_negatives_mask,
                                aa_groups,seq_max_len,"TrainHighlyConfidentNegatives_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode))

        #Highlight: Test

        VegvisirMI.calculate_mi(epitopes_array_int_division_test.all,epitopes_array_blosum_norm_group_divison_test.all_mask,
                                aa_groups,seq_max_len,"TestAll_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/all".format(args.sequence_type,analysis_mode))

        if epitopes_array_int_division_test.negatives.shape[0] == test_longest_array:

            VegvisirMI.calculate_mi(epitopes_array_int_division_test.positives,epitopes_array_blosum_norm_group_divison_test.positives_mask,
                                    aa_groups,seq_max_len,"TestPositives_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_int_division_test.negatives[test_idx_select],epitopes_array_blosum_norm_group_divison_test.negatives_mask,
                                    aa_groups,seq_max_len,"TestNegatives_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))
        else:
            VegvisirMI.calculate_mi(epitopes_array_int_division_test.positives[test_idx_select],
                                    epitopes_array_blosum_norm_group_divison_test.positives_mask,
                                    aa_groups, seq_max_len, "TestPositives_raw_aa", storage_folder, args.dataset_name,
                                    "similarities/{}/Test/{}/neighbours1/positives".format(args.sequence_type,
                                                                                           analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_int_division_test.negatives,
                                    epitopes_array_blosum_norm_group_divison_test.negatives_mask,
                                    aa_groups, seq_max_len, "TestNegatives_raw_aa", storage_folder, args.dataset_name,
                                    "similarities/{}/Test/{}/neighbours1/negatives".format(args.sequence_type,
                                                                                           analysis_mode))

        VegvisirMI.calculate_mi(epitopes_array_int_division_test.high_confidence_negatives,epitopes_array_blosum_norm_group_divison_test.high_confidence_negatives_mask,
                                aa_groups,seq_max_len,"TestHighlyConfidentNegatives_raw_aa",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode))

        #Highlight: Mutual informaton calculated using custom blosum norm amino acid subdivision

        #Highlight: Train
        VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_train.all,epitopes_array_blosum_norm_group_divison_train.all_mask,
                                aa_groups,seq_max_len,"TrainAll_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode))

        if epitopes_array_blosum_norm_group_divison_train.negatives.shape[0] == train_longest_array:

            VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_train.positives,epitopes_array_blosum_norm_group_divison_train.positives_mask,
                                    aa_groups,seq_max_len,"TrainPositives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_train.negatives[train_idx_select],epitopes_array_blosum_norm_group_divison_train.negatives_mask,
                                    aa_groups,seq_max_len,"TrainNegatives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))
        else:
            VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_train.positives[train_idx_select],epitopes_array_blosum_norm_group_divison_train.positives_mask,
                                    aa_groups,seq_max_len,"TrainPositives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_train.negatives[train_idx_select],epitopes_array_blosum_norm_group_divison_train.negatives_mask,
                                    aa_groups,seq_max_len,"TrainNegatives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))


        VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_train.high_confidence_negatives,epitopes_array_blosum_norm_group_divison_train.high_confidence_negatives_mask,
                                aa_groups,seq_max_len,"TrainHighlyConfidentNegatives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode))

        #Highlight: Test
        VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_test.all,epitopes_array_blosum_norm_group_divison_test.all_mask,
                                aa_groups,seq_max_len,"TestAll_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/all".format(args.sequence_type,analysis_mode))

        if epitopes_array_int_division_test.negatives.shape[0] == test_longest_array:
            VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_test.positives,epitopes_array_blosum_norm_group_divison_test.positives_mask,
                                    aa_groups,seq_max_len,"TestPositives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_test.negatives[test_idx_select],epitopes_array_blosum_norm_group_divison_test.negatives_mask,
                                    aa_groups,seq_max_len,"TestNegatives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))

        else:
            VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_test.positives[test_idx_select],epitopes_array_blosum_norm_group_divison_test.positives_mask,
                                    aa_groups,seq_max_len,"TestPositives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_test.negatives,epitopes_array_blosum_norm_group_divison_test.negatives_mask,
                                    aa_groups,seq_max_len,"TestNegatives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))



        VegvisirMI.calculate_mi(epitopes_array_blosum_norm_group_divison_test.high_confidence_negatives,epitopes_array_blosum_norm_group_divison_test.high_confidence_negatives_mask,
                                aa_groups,seq_max_len,"TestHighlyConfidentNegatives_blosum_norm_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode))

        #Highlight: Mutual Information using the classical amino acids subdivisions
        #Highlight: Train
        VegvisirMI.calculate_mi(epitopes_array_int_group_divison_train.all,epitopes_array_int_group_divison_train.all_mask,
                                aa_groups,seq_max_len,"TrainAll_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode))

        if epitopes_array_int_group_divison_train.negatives.shape[0] == train_longest_array:
            VegvisirMI.calculate_mi(epitopes_array_int_group_divison_train.positives,epitopes_array_int_group_divison_train.positives_mask,
                                    aa_groups,seq_max_len,"TrainPositives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_int_group_divison_train.negatives[train_idx_select],epitopes_array_int_group_divison_train.negatives_mask,
                                    aa_groups,seq_max_len,"TrainNegatives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))
        else:
            VegvisirMI.calculate_mi(epitopes_array_int_group_divison_train.positives[train_idx_select],epitopes_array_int_group_divison_train.positives_mask,
                                    aa_groups,seq_max_len,"TrainPositives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_int_group_divison_train.negatives,epitopes_array_int_group_divison_train.negatives_mask,
                                    aa_groups,seq_max_len,"TrainNegatives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))

        VegvisirMI.calculate_mi(epitopes_array_int_group_divison_train.high_confidence_negatives,epitopes_array_int_group_divison_train.high_confidence_negatives_mask,
                                aa_groups,seq_max_len,"TrainHighlyConfidentNegatives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Train/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode))
        #Highlight: Test
        VegvisirMI.calculate_mi(epitopes_array_int_group_divison_test.all,epitopes_array_int_group_divison_test.all_mask,
                                aa_groups,seq_max_len,"TestAll_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/all".format(args.sequence_type,analysis_mode))

        if epitopes_array_int_group_divison_test.negatives.shape[0] == test_longest_array:
            VegvisirMI.calculate_mi(epitopes_array_int_group_divison_test.positives,epitopes_array_int_group_divison_test.positives_mask,
                                    aa_groups,seq_max_len,"TestPositives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_int_group_divison_test.negatives[test_idx_select],epitopes_array_int_group_divison_test.negatives_mask,
                                    aa_groups,seq_max_len,"TestNegatives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))
        else:
            VegvisirMI.calculate_mi(epitopes_array_int_group_divison_test.positives[test_idx_select],epitopes_array_int_group_divison_test.positives_mask,
                                    aa_groups,seq_max_len,"TestPositives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))
            VegvisirMI.calculate_mi(epitopes_array_int_group_divison_test.negatives,epitopes_array_int_group_divison_test.negatives_mask,
                                    aa_groups,seq_max_len,"TestNegatives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))



        VegvisirMI.calculate_mi(epitopes_array_int_group_divison_test.high_confidence_negatives,epitopes_array_int_group_divison_test.high_confidence_negatives_mask,
                                aa_groups,seq_max_len,"TestHighlyConfidentNegatives_int_group_division",storage_folder,args.dataset_name,"similarities/{}/Test/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode))



        # identifiers = data.index.values.tolist()  # TODO: reset index in process data function?
        # VegvisirMI.calculate_mutual_information(positives_arr.tolist(),identifiers,seq_max_len,"TrainPositives",storage_folder,args.dataset_name)
        # VegvisirMI.calculate_mutual_information(negatives_arr.tolist(),identifiers,seq_max_len,"TrainNegatives",storage_folder,args.dataset_name)
        # VegvisirMI.calculate_mutual_information(high_conf_negatives_arr.tolist(),identifiers,seq_max_len,"TrainHighlyConfidentNegatives",storage_folder,args.dataset_name)
    if plot_frequencies:#Highlight: remember to use "int"!!!!!!!!
        VegvisirPlots.plot_aa_frequencies(epitopes_array_int_division_test.all,corrected_aa_types,aa_dict,seq_max_len,storage_folder,args,"similarities/{}/Test/{}/neighbours1/all".format(args.sequence_type,analysis_mode),"TestAll")
        VegvisirPlots.plot_aa_frequencies(epitopes_array_int_division_test.positives,corrected_aa_types,aa_dict,seq_max_len,storage_folder,args,"similarities/{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode),"TestPositives")
        VegvisirPlots.plot_aa_frequencies(epitopes_array_int_division_test.negatives,corrected_aa_types,aa_dict,seq_max_len,storage_folder,args,"similarities/{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode),"TestNegatives")
        VegvisirPlots.plot_aa_frequencies(epitopes_array_int_division_test.high_confidence_negatives,corrected_aa_types,aa_dict,seq_max_len,storage_folder,args,"similarities/{}/Test/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode),"TestHighConfidenceNegatives")

        VegvisirPlots.plot_aa_frequencies(epitopes_array_int_division_train.all,corrected_aa_types,aa_dict,seq_max_len,storage_folder,args,"similarities/{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode),"TrainAll")
        VegvisirPlots.plot_aa_frequencies(epitopes_array_int_division_train.positives,corrected_aa_types,aa_dict,seq_max_len,storage_folder,args,"similarities/{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode),"TrainPositives")
        VegvisirPlots.plot_aa_frequencies(epitopes_array_int_division_train.negatives,corrected_aa_types,aa_dict,seq_max_len,storage_folder,args,"similarities/{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode),"TrainNegatives")
        VegvisirPlots.plot_aa_frequencies(epitopes_array_int_division_train.high_confidence_negatives,corrected_aa_types,aa_dict,seq_max_len,storage_folder,args,"similarities/{}/Train/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode),"TrainHighConfidenceNegatives")
    if plot_cosine_similarity:
        print("Calculating  epitopes similarity matrices (this might take a while, 10 minutes for 10000 sequences) ....")
        all_sim_results =VegvisirSimilarities.calculate_similarity_matrix_parallel(epitopes_array_blosum,
                                                                      seq_max_len,
                                                                      epitopes_array_mask,
                                                                      storage_folder, args,
                                                                     "{}/All/{}/neighbours1/all".format(args.sequence_type,analysis_mode),
                                                                      ksize=ksize)
        
        #Highlight: Train dataset
        train_all_sim_results =VegvisirSimilarities.calculate_similarity_matrix_parallel(epitopes_array_blosum_division_train.all,
                                                                      seq_max_len,
                                                                      epitopes_array_blosum_division_train.all_mask,
                                                                      storage_folder, args,
                                                                     "{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode),
                                                                     ksize=ksize)
        train_positives_sim_results =VegvisirSimilarities.calculate_similarity_matrix_parallel(epitopes_array_blosum_division_train.positives,
                                                                      seq_max_len,
                                                                      epitopes_array_blosum_division_train.positives_mask,
                                                                      storage_folder, args,
                                                                     "{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode),
                                                                     ksize=ksize)

        train_negatives_sim_results=VegvisirSimilarities.calculate_similarity_matrix_parallel(epitopes_array_blosum_division_train.negatives,
                                                                      seq_max_len,
                                                                      epitopes_array_blosum_division_train.negatives_mask,
                                                                      storage_folder, args,
                                                                     "{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode),
                                                                      ksize=ksize)
        train_high_conf_sim_results=VegvisirSimilarities.calculate_similarity_matrix_parallel(epitopes_array_blosum_division_train.high_confidence_negatives,
                                                                      seq_max_len,
                                                                      epitopes_array_blosum_division_train.high_confidence_negatives_mask,
                                                                      storage_folder, args,
                                                                      "{}/Train/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode),
                                                                      ksize=ksize)


        train_sim_results = {"all":train_all_sim_results,
                             "positives":train_positives_sim_results,
                             "negatives":train_negatives_sim_results,
                             "high_conf_negatives":train_high_conf_sim_results}

        #Highlight: Test dataset
        test_all_sim_results = VegvisirSimilarities.calculate_similarity_matrix_parallel(
            epitopes_array_blosum_division_test.all,
            seq_max_len,
            epitopes_array_blosum_division_test.all_mask,
            storage_folder, args,
            "{}/Test/{}/neighbours1/all".format(args.sequence_type, analysis_mode),
            ksize=ksize)

        test_positives_sim_results = VegvisirSimilarities.calculate_similarity_matrix_parallel(
            epitopes_array_blosum_division_test.positives,
            seq_max_len,
            epitopes_array_blosum_division_test.positives_mask,
            storage_folder, args,
            "{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode),
            ksize=ksize)

        test_negatives_sim_results = VegvisirSimilarities.calculate_similarity_matrix_parallel(
            epitopes_array_blosum_division_test.negatives,
            seq_max_len,
            epitopes_array_blosum_division_test.negatives_mask,
            storage_folder, args,
            "{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode),
            ksize=ksize)
        test_high_conf_negatives_sim_results = VegvisirSimilarities.calculate_similarity_matrix_parallel(
            epitopes_array_blosum_division_test.high_confidence_negatives,
            seq_max_len,
            epitopes_array_blosum_division_test.high_confidence_negatives_mask,
            storage_folder, args,
            "{}/Test/{}/neighbours1/highconfnegatives".format(args.sequence_type,analysis_mode),
            ksize=ksize)

        test_sim_results = { "all":test_all_sim_results,
                             "positives": test_positives_sim_results,
                             "negatives": test_negatives_sim_results,
                             "high_conf_negatives": test_high_conf_negatives_sim_results}

    else:
        print("Loading pre-calculated epitopes similarity/weights matrices (warning big matrices) located at {}".format("{}/{}/similarities/".format(storage_folder,args.dataset_name)))
        
        train_all_sim_results = SimilarityResults(positional_weights=np.load("{}/{}/similarities/{}/positional_weights.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                               percent_identity_mean=np.load("{}/{}/similarities/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                               cosine_similarity_mean=np.load("{}/{}/similarities/{}/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                               kmers_pid_similarity=np.load("{}/{}/similarities/{}/kmers_pid_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                               kmers_cosine_similarity=np.load("{}/{}/similarities/{}/kmers_cosine_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode))))

        train_positives_sim_results = SimilarityResults(positional_weights=np.load("{}/{}/similarities/{}/positional_weights.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))),
                                               percent_identity_mean=np.load("{}/{}/similarities/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))),
                                               cosine_similarity_mean=np.load("{}/{}/similarities/{}/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))),
                                               kmers_pid_similarity=np.load("{}/{}/similarities/{}/kmers_pid_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))),
                                               kmers_cosine_similarity=np.load("{}/{}/similarities/{}/kmers_cosine_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))))
        train_negatives_sim_results = SimilarityResults(positional_weights=np.load("{}/{}/similarities/{}/positional_weights.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))),
                                               percent_identity_mean=np.load("{}/{}/similarities/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))),
                                               cosine_similarity_mean=np.load("{}/{}/similarities/{}/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))),
                                               kmers_pid_similarity=np.load("{}/{}/similarities/{}/kmers_pid_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))),
                                               kmers_cosine_similarity=np.load("{}/{}/similarities/{}/kmers_cosine_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))))
        
        try:
            train_high_conf_negatives_sim_results = SimilarityResults(positional_weights=np.load("{}/{}/similarities/{}/positional_weights.npy".format(storage_folder,args.dataset_name,"Train/{}/neighbours1/highconfnegatives".format(analysis_mode))),
                                                   percent_identity_mean=np.load("{}/{}/similarities/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name,"Train/{}/neighbours1/highconfnegatives".format(analysis_mode))),
                                                   cosine_similarity_mean=np.load("{}/{}/similarities/{}/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name,"Train/{}/neighbours1/highconfnegatives".format(analysis_mode))),
                                                   kmers_pid_similarity=np.load("{}/{}/similarities/{}/kmers_pid_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"Train/{}/neighbours1/highconfnegatives".format(analysis_mode))),
                                                   kmers_cosine_similarity=np.load("{}/{}/similarities/{}/kmers_cosine_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"Train/{}/neighbours1/highconfnegatives".format(analysis_mode))))
        except:
            train_high_conf_negatives_sim_results = None

        train_sim_results = {
            "positives":train_positives_sim_results,
            "negatives": train_negatives_sim_results,
            "high_conf_negatives": train_high_conf_negatives_sim_results,
        }
        
        
        #Highlight: Test
        test_all_sim_results = SimilarityResults(positional_weights=np.load("{}/{}/similarities/{}/positional_weights.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                                        percent_identity_mean=np.load("{}/{}/similarities/{}/percent_identity_mean.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                                        cosine_similarity_mean=np.load("{}/{}/similarities/{}/cosine_similarity_mean.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                                        kmers_pid_similarity=np.load("{}/{}/similarities/{}/kmers_pid_similarity_3ksize.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                                        kmers_cosine_similarity=np.load("{}/{}/similarities/{}/kmers_cosine_similarity_3ksize.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/all".format(args.sequence_type,analysis_mode))))

        test_positives_sim_results = SimilarityResults(positional_weights=np.load("{}/{}/similarities/{}/positional_weights.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))),
                                                        percent_identity_mean=np.load("{}/{}/similarities/{}/percent_identity_mean.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))),
                                                        cosine_similarity_mean=np.load("{}/{}/similarities/{}/cosine_similarity_mean.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))),
                                                        kmers_pid_similarity=np.load("{}/{}/similarities/{}/kmers_pid_similarity_3ksize.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))),
                                                        kmers_cosine_similarity=np.load("{}/{}/similarities/{}/kmers_cosine_similarity_3ksize.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode))))
        test_negatives_sim_results = SimilarityResults(positional_weights=np.load("{}/{}/similarities/{}/positional_weights.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))),
                                                        percent_identity_mean=np.load("{}/{}/similarities/{}/percent_identity_mean.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))),
                                                        cosine_similarity_mean=np.load("{}/{}/similarities/{}/cosine_similarity_mean.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))),
                                                        kmers_pid_similarity=np.load("{}/{}/similarities/{}/kmers_pid_similarity_3ksize.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))),
                                                        kmers_cosine_similarity=np.load("{}/{}/similarities/{}/kmers_cosine_similarity_3ksize.npy".format(storage_folder, args.dataset_name,"{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode))))

        try:
            test_high_conf_negatives_sim_results = SimilarityResults(positional_weights=np.load("{}/{}/similarities/{}/positional_weights.npy".format(storage_folder, args.dataset_name,"Test/{}/neighbours1/highconfnegatives".format(analysis_mode))),
                                                                      percent_identity_mean=np.load("{}/{}/similarities/{}/percent_identity_mean.npy".format(storage_folder, args.dataset_name,"Test/{}/neighbours1/highconfnegatives".format(analysis_mode))),
                                                                      cosine_similarity_mean=np.load("{}/{}/similarities/{}/cosine_similarity_mean.npy".format(storage_folder, args.dataset_name,"Test/{}/neighbours1/highconfnegatives".format(analysis_mode))),
                                                                      kmers_pid_similarity=np.load("{}/{}/similarities/{}/kmers_pid_similarity_3ksize.npy".format(storage_folder, args.dataset_name,"Test/{}/neighbours1/highconfnegatives".format(analysis_mode))),
                                                                      kmers_cosine_similarity=np.load("{}/{}/similarities/{}/kmers_cosine_similarity_3ksize.npy".format(storage_folder, args.dataset_name,"Test/{}/neighbours1/highconfnegatives".format(analysis_mode))))
        except:
            test_high_conf_negatives_sim_results = None

        test_sim_results = {
            "all": test_all_sim_results,
            "positives": test_positives_sim_results,
            "negatives": test_negatives_sim_results,
            "high_conf_negatives": test_high_conf_negatives_sim_results,
        }

        all_sim_results= SimilarityResults(positional_weights=np.load("{}/{}/similarities/{}/positional_weights.npy".format(storage_folder,args.dataset_name,"{}/All/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                               percent_identity_mean=np.load("{}/{}/similarities/{}/percent_identity_mean.npy".format(storage_folder,args.dataset_name,"{}/All/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                               cosine_similarity_mean=np.load("{}/{}/similarities/{}/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name,"{}/All/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                               kmers_pid_similarity=np.load("{}/{}/similarities/{}/kmers_pid_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"{}/All/{}/neighbours1/all".format(args.sequence_type,analysis_mode))),
                                               kmers_cosine_similarity=np.load("{}/{}/similarities/{}/kmers_cosine_similarity_3ksize.npy".format(storage_folder,args.dataset_name,"{}/All/{}/neighbours1/all".format(args.sequence_type,analysis_mode))))

    calculate_partitions = False
    if calculate_partitions: #TODO: move elsewhere
        cosine_similarity_mean = all_sim_results.cosine_similarity_mean
        cosine_umap = umap.UMAP(n_components=6).fit_transform(cosine_similarity_mean)
        clustering = DBSCAN(eps=0.3, min_samples=1,metric="euclidean",algorithm="auto",p=3).fit(cosine_umap) #eps 4
        #clustering = hdbscan.HDBSCAN(min_cluster_size=1, gen_min_span_tree=True).fit(cosine_similarity_mean)
        labels = np.unique(clustering.labels_,return_counts=True)

    return all_sim_results
def process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict,features_names=None,plot_blosum=False,plot_umap=False):
    """
    Notes:
      - Mid-padding : https://www.nature.com/articles/s41598-020-71450-8
    :param pandas dataframe data: Contains Icore, immunodominance_score, immunodominance_score_scaled, training , partition and Rnk_EL
    :param args: Commmand line arguments
    :param storage_folder: Data location path
    """
    sequence_column = filters_dict["filter_kmers"][2]
    epitopes_list = data[[sequence_column]].values.tolist()
    epitopes_list = functools.reduce(operator.iconcat, epitopes_list, [])  # flatten list of lists
    seq_max_len = len(max(epitopes_list, key=len))
    epitopes_lens = np.array(list(map(len, epitopes_list)))
    unique_lens = list(set(epitopes_lens))
    corrected_aa_types = len(set().union(*epitopes_list))
    corrected_aa_types = [corrected_aa_types + 1 if len(unique_lens) > 1 else corrected_aa_types][0]
    if len(unique_lens) > 1: # Highlight: Pad the sequences (relevant when they differ in length)
        aa_dict = VegvisirUtils.aminoacid_names_dict(corrected_aa_types , zero_characters=["#"])
        if args.shuffle_sequence:
            warnings.warn("shuffling the sequence sites for testing purposes")
        epitopes_pad_result = VegvisirLoadUtils.SequencePadding(epitopes_list,seq_max_len,args.seq_padding,args.shuffle_sequence).run()
        epitopes_padded, epitopes_padded_mask = zip(*epitopes_pad_result) #unpack list of tuples onto 2 lists
        blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(corrected_aa_types , args.subs_matrix,
                                                                                   zero_characters= ["#"],
                                                                                   include_zero_characters=True)

    else:
        print("All sequences found to have the same length")
        aa_dict = VegvisirUtils.aminoacid_names_dict(corrected_aa_types)
        if args.shuffle_sequence:
            warnings.warn("shuffling the sequence for testing purposes")

        epitopes_pad_result = VegvisirLoadUtils.SequencePadding(epitopes_list, seq_max_len, "no_padding",args.shuffle_sequence).run()
        epitopes_padded, epitopes_padded_mask = zip(*epitopes_pad_result)  # unpack list of tuples onto 2 lists
        blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(corrected_aa_types, args.subs_matrix,
                                                                                   zero_characters=[],
                                                                                   include_zero_characters=False)




    #VegvisirUtils.convert_to_pandas_dataframe(epitopes_padded,data,storage_folder,args,use_test=True)




    epitopes_array = np.array(epitopes_padded)
    if args.seq_padding == "replicated_borders":  # I keep it separately to avoid doing the np vectorized loop twice
        epitopes_array_int = np.vectorize(aa_dict.get)(epitopes_array)
        epitopes_array_mask = np.array(epitopes_padded_mask)
        epitopes_array_int_mask = np.vectorize(aa_dict.get)(epitopes_array_mask)
        epitopes_mask = epitopes_array_int_mask.astype(bool)
    else:
        epitopes_array_int = np.vectorize(aa_dict.get)(epitopes_array)
        if len(unique_lens) > 1: #there are some paddings, that equal 0, therefore we can set them to False
            epitopes_mask = epitopes_array_int.astype(bool)
        else: #there is no padding, therefore number 0 equals an amino acid
            epitopes_mask = np.ones_like(epitopes_array_int).astype(bool)
    if args.subset_data != "no":
        print("WARNING : Using a subset of the data of {}".format(args.subset_data))
        epitopes_array = epitopes_array[:args.subset_data]
        epitopes_mask = epitopes_mask[:args.subset_data]
    blosum_norm = np.linalg.norm(blosum_array[1:, 1:], axis=0)

    aa_list = [val for key, val in aa_dict.items() if val in list(blosum_array[:, 0])]
    blosum_norm_dict = dict(zip(aa_list,blosum_norm.tolist()))
    epitopes_array_blosum_norm = np.vectorize(blosum_norm_dict.get)(epitopes_array_int)
    if plot_blosum:
        VegvisirPlots.plot_blosum_cosine(blosum_array, storage_folder, args)


    epitopes_array_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(epitopes_array_int)
    epitopes_array_onehot_encoding = VegvisirUtils.convert_to_onehot(epitopes_array_int,dimensions=epitopes_array_blosum.shape[2])
    n_data = epitopes_array.shape[0]
    if args.dataset_name != "viral_dataset6":
        all_sim_results = data_exploration(data, epitopes_array_blosum, epitopes_array_int, epitopes_mask, aa_dict, aa_list,
                         blosum_norm, seq_max_len, storage_folder, args, corrected_aa_types,analysis_mode,filters_dict)
        positional_weights = all_sim_results.positional_weights
        positional_weights_mask = (positional_weights[..., None] > 0.6).any(-1)
    else:
        all_sim_results = SimilarityResults(positional_weights=np.ones((n_data,seq_max_len)),
                                               percent_identity_mean=None,
                                               cosine_similarity_mean=None,
                                               kmers_pid_similarity=None,
                                               kmers_cosine_similarity=None)
        positional_weights = np.ones((n_data,seq_max_len))
        positional_weights_mask = np.ones((n_data,seq_max_len)).astype(bool)


    #Highlight: Reattatch partition, identifier, label, immunodominance score
    labels = data[["target_corrected"]].values.tolist()
    identifiers = data.index.values.tolist() #TODO: reset index in process data function?
    partitions = data[["partition"]].values.tolist()
    training = data[["training"]].values.tolist()
    confidence_scores = data["confidence_score"].values.tolist()
    immunodominance_scores = data[["immunodominance_score"]].values.tolist()

     #TODO : make a function?
    if features_names is not None:
        features_scores = data[features_names].to_numpy()
        #for feat in features_scores
        identifiers_labels_array = np.zeros((n_data, 1, seq_max_len + len(features_names)))
        identifiers_labels_array[:, 0, 0] = np.array(labels).squeeze(-1)
        identifiers_labels_array[:, 0, 1] = np.array(identifiers)
        identifiers_labels_array[:, 0, 2] = np.array(partitions).squeeze(-1)
        identifiers_labels_array[:, 0, 3] = np.array(training).squeeze(-1).astype(int)
        identifiers_labels_array[:, 0, 4] = np.array(immunodominance_scores).squeeze(-1)
        identifiers_labels_array[:, 0, 5] = np.array(confidence_scores)

        epitopes_array = np.concatenate([epitopes_array,features_scores],axis=1)
        epitopes_array_int = np.concatenate([epitopes_array_int,features_scores],axis=1)
        epitopes_array_blosum_norm = np.concatenate([epitopes_array_blosum_norm,features_scores],axis=1)

        data_array_raw = np.concatenate([identifiers_labels_array, epitopes_array[:, None]], axis=1)
        data_array_int = np.concatenate([identifiers_labels_array,epitopes_array_int[:, None]], axis=1)
        data_array_blosum_norm = np.concatenate([identifiers_labels_array,epitopes_array_blosum_norm[:, None]], axis=1)

        identifiers_labels_array_blosum = np.zeros((n_data, 1, seq_max_len + len(features_names), epitopes_array_blosum.shape[2]))
        identifiers_labels_array_blosum[:, 0, 0, 0] = np.array(labels).squeeze(-1)
        identifiers_labels_array_blosum[:, 0, 0, 1] = np.array(identifiers)
        identifiers_labels_array_blosum[:, 0, 0, 2] = np.array(partitions).squeeze(-1)
        identifiers_labels_array_blosum[:, 0, 0, 3] = np.array(training).squeeze(-1).astype(int)
        identifiers_labels_array_blosum[:, 0, 0, 4] = np.array(immunodominance_scores).squeeze(-1)
        identifiers_labels_array_blosum[:, 0, 0, 5] = np.array(confidence_scores)


        features_array_blosum = np.zeros((n_data,len(features_names), epitopes_array_blosum.shape[2]))
        features_array_blosum[:,:,0] = features_scores

        epitopes_array_blosum = np.concatenate([epitopes_array_blosum,features_array_blosum],axis=1)
        epitopes_array_onehot_encoding = np.concatenate([epitopes_array_onehot_encoding,features_array_blosum],axis=1)
        data_array_blosum_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_blosum[:, None]],axis=1)
        data_array_onehot_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_onehot_encoding[:, None]], axis=1)

    else:
        identifiers_labels_array = np.zeros((n_data, 1, seq_max_len))
        identifiers_labels_array[:, 0, 0] = np.array(labels).squeeze(-1)
        identifiers_labels_array[:, 0, 1] = np.array(identifiers)
        identifiers_labels_array[:, 0, 2] = np.array(partitions).squeeze(-1)
        identifiers_labels_array[:, 0, 3] = np.array(training).squeeze(-1).astype(int)
        identifiers_labels_array[:, 0, 4] = np.array(immunodominance_scores).squeeze(-1)
        identifiers_labels_array[:, 0, 5] = np.array(confidence_scores)


        data_array_raw = np.concatenate([identifiers_labels_array, epitopes_array[:,None]], axis=1)
        data_array_int = np.concatenate([identifiers_labels_array, epitopes_array_int[:,None]], axis=1)
        data_array_blosum_norm = np.concatenate([identifiers_labels_array, epitopes_array_blosum_norm[:,None]], axis=1)

        identifiers_labels_array_blosum = np.zeros((n_data, 1, seq_max_len, epitopes_array_blosum.shape[2]))
        identifiers_labels_array_blosum[:, 0, 0, 0] = np.array(labels).squeeze(-1)
        identifiers_labels_array_blosum[:, 0, 0, 1] = np.array(identifiers)
        identifiers_labels_array_blosum[:, 0, 0, 2] = np.array(partitions).squeeze(-1)
        identifiers_labels_array_blosum[:, 0, 0, 3] = np.array(training).squeeze(-1).astype(int)
        identifiers_labels_array_blosum[:, 0, 0, 4] = np.array(immunodominance_scores).squeeze(-1)
        identifiers_labels_array_blosum[:, 0, 0, 5] = np.array(confidence_scores)


        data_array_blosum_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_blosum[:,None]], axis=1)
        data_array_onehot_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_onehot_encoding[:,None]], axis=1)

    data_array_blosum_encoding_mask = np.broadcast_to(epitopes_mask[:, None, :, None], (n_data, 2, seq_max_len,corrected_aa_types))  # I do it like this in case the padding is not represented as 0, otherwise just use bool. Note: The first row of the second dimension is a dummy
    #distance_pid_cosine = VegvisirUtils.euclidean_2d_norm(percent_identity_mean,cosine_similarity_mean) #TODO: What to do with this?


    data_info = DatasetInfo(script_dir=script_dir,
                            storage_folder=storage_folder,
                            data_array_raw=data_array_raw,
                            data_array_int=torch.from_numpy(data_array_int),
                            data_array_int_mask=epitopes_mask,
                            data_array_blosum_encoding=torch.from_numpy(data_array_blosum_encoding),
                            data_array_blosum_encoding_mask=torch.from_numpy(data_array_blosum_encoding_mask),
                            data_array_onehot_encoding=torch.from_numpy(data_array_onehot_encoding),
                            data_array_blosum_norm=torch.from_numpy(data_array_blosum_norm),
                            blosum=blosum_array,
                            n_data=n_data,
                            seq_max_len = seq_max_len,
                            max_len=[seq_max_len + len(features_names) if features_names is not None else seq_max_len][0],
                            corrected_aa_types = corrected_aa_types,
                            input_dim=corrected_aa_types,
                            positional_weights=torch.from_numpy(all_sim_results.positional_weights),
                            positional_weights_mask=torch.from_numpy(positional_weights_mask),
                            percent_identity_mean= all_sim_results.percent_identity_mean,
                            cosine_similarity_mean= all_sim_results.cosine_similarity_mean,
                            kmers_pid_similarity=all_sim_results.kmers_pid_similarity,
                            kmers_cosine_similarity=all_sim_results.kmers_cosine_similarity,
                            features_names = features_names)

    if not os.path.exists("{}/{}/umap_data_norm.png".format(storage_folder,args.dataset_name)):
        VegvisirPlots.plot_data_umap(data_array_blosum_norm,data_info.seq_max_len,data_info.max_len,storage_folder,args.dataset_name)
    return data_info

def prepare_nnalign_no_test(args,storage_folder,data,column_names):
    data_train = data[(data["training"] == 1) & (data["partition"] != 4)][column_names]
    data_valid = data[(data["training"] == 1) & (data["partition"] == 4)][column_names]
    data_train = data_train.astype({'partition': 'int'})
    data_valid = data_valid.astype({'partition': 'int'})
    data_train["Icore"].to_csv("{}/{}/viral_seq2logo.tsv".format(storage_folder,args.dataset_name),sep="\t",index=False,header=None)

    data_train.to_csv("{}/{}/viral_nnalign_input_train.tsv".format(storage_folder,args.dataset_name),sep="\t",index=False,header=None)
    data_valid.to_csv("{}/{}/viral_nnalign_input_valid.tsv".format(storage_folder,args.dataset_name), sep="\t",index=False,header=None) #TODO: Header None?
    VegvisirNNalign.run_nnalign(args,storage_folder)
def prepare_nnalign(args,storage_folder,data,column_names,no_test=True):

    if no_test:
        warnings.warn("Creating only training dataset for NNAlign (otherwise please set <no_test> to False")
        column_names.remove("partition")
        data_train = data[column_names]
        #data["Icore"].to_csv("{}/{}/viral_seq2logo.tsv".format(storage_folder, args.dataset_name), sep="\t",index=False, header=None)

        data_train.to_csv("{}/{}/viral_nnalign_input_train.tsv".format(storage_folder, args.dataset_name), sep="\t",
                          index=False, header=None)

    else:
        data_train = data[data["training"] == True][column_names]
        data_valid = data[data["training"] == False][column_names]
        data_train = data_train.astype({'partition': 'int'})
        data_valid.drop("partition",axis=1,inplace=True)
        data_train["Icore"].to_csv("{}/{}/viral_seq2logo.tsv".format(storage_folder,args.dataset_name),sep="\t",index=False,header=None)

        data_train.to_csv("{}/{}/viral_nnalign_input_train.tsv".format(storage_folder,args.dataset_name),sep="\t",index=False,header=None)
        data_valid.to_csv("{}/{}/viral_nnalign_input_valid.tsv".format(storage_folder,args.dataset_name), sep="\t",index=False,header=None) #TODO: Header None?

    exit()
    VegvisirNNalign.run_nnalign(args,storage_folder)
def set_confidence_score(data):

    nmax_tested = data["Assay_number_of_subjects_tested"].max()
    # data.loc[(data["Assay_number_of_subjects_tested"] < 50) & (data["Assay_number_of_subjects_responded"] == 0), "confidence_score"] = data.loc[(data["Assay_number_of_subjects_tested"] < 50) & (data["Assay_number_of_subjects_responded"] == 0),"Assay_number_of_subjects_tested"]/nmax_tested
    # data.loc[(data["Assay_number_of_subjects_tested"] <= 50) & (data["Assay_number_of_subjects_responded"] > 0), "confidence_score"] = 1
    # data.loc[data["Assay_number_of_subjects_tested"] > 50, "confidence_score"] = 1
    data.loc[(data["Assay_number_of_subjects_tested"] <= 50) & (data["Assay_number_of_subjects_responded"] == 0), "confidence_score"] = 0.5
    data.loc[(data["Assay_number_of_subjects_tested"] <= 50) & (data["Assay_number_of_subjects_responded"] > 0), "confidence_score"] = 1
    data.loc[data["Assay_number_of_subjects_tested"] > 50, "confidence_score"] = 1
    data.loc[(data["Assay_number_of_subjects_tested"] <= 10) & (data["Assay_number_of_subjects_responded"] == 0), "confidence_score"] = 0.1
    data.loc[(data["Assay_number_of_subjects_tested"] <= 10) & (data["Assay_number_of_subjects_responded"] > 0), "confidence_score"] = 1
    data.loc[(data["Assay_number_of_subjects_tested"] <= 20) & (data["Assay_number_of_subjects_responded"] == 0), "confidence_score"] = 0.35
    data.loc[(data["Assay_number_of_subjects_tested"] <= 20) & (data["Assay_number_of_subjects_responded"] > 0), "confidence_score"] = 1
    nan_rows = data[data["confidence_score"].isna()]
    #print(nan_rows[["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"]])
    return data

