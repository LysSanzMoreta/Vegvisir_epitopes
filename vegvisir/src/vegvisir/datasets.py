"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import itertools
import json
import os
import time,datetime
import warnings

import dill
import pandas as pd
import operator,functools
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict,namedtuple
import seaborn as sns
import dataframe_image as dfi
import torch
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches
import vegvisir.nnalign as VegvisirNNalign
import vegvisir.utils as VegvisirUtils
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.plots as VegvisirPlots
plt.style.use('ggplot')
DatasetInfo = namedtuple("DatasetInfo",["script_dir","storage_folder","data_array_raw","data_array_int","data_array_int_mask",
                                        "data_array_blosum_encoding","data_array_blosum_encoding_mask","data_array_onehot_encoding","data_array_blosum_norm","blosum",
                                        "n_data","seq_max_len","max_len","corrected_aa_types","input_dim","percent_identity_mean","cosine_similarity_mean","kmers_pid_similarity","kmers_cosine_similarity","features_names"])
def available_datasets():
    """Prints the available datasets"""
    datasets = {0:"viral_dataset",
                1:"viral_dataset2",
                2:"viral_dataset3",
                3:"viral_dataset4",
                4:"viral_dataset5"}
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
                 "viral_dataset5":viral_dataset5}
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
    data_info = process_data(data_a,args,storage_folder,script_dir,"Icore")
    return data_info

def select_filters():
    filters_dict = {"filter_kmers":[False,9,"Icore_non_anchor"], #Icore_non_anchor
                    "group_alleles":[True],
                    "filter_ntested":[False,10],
                    "filter_lowconfidence":[False],
                    "corrected_immunodominance_score":[False,10]}
    return filters_dict

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
    name_suffix = "__".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
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
    filters_dict = select_filters()
    json.dump(filters_dict, dataset_info_file, indent=2)

    if filters_dict["group_alleles"][0]:
        # Group data by Icore, therefore the alleles are grouped
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

    data_info = process_data(data,args,storage_folder,script_dir,filters_dict["filter_kmers"][2])

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

    filters_dict = select_filters()
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
    data_info = process_data(data,args,storage_folder,script_dir,sequence_column=filters_dict["filter_kmers"][2],features_names=features_names)

    return data_info

def viral_dataset5(dataset_name,script_dir,storage_folder,args,results_dir,update):
    """
    Contains "artificial" or fake negative epitopes solely in the test dataset
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

    filters_dict = select_filters()
    filters_dict = select_filters()
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
    warnings.warn("Setting low confidence score to the artificial negatives in the test dataset")
    data.loc[mask,"confidence_score"] = 0.6
    data_info = process_data(data,args,storage_folder,script_dir,filters_dict["filter_kmers"][2])

    return data_info

def process_data(data,args,storage_folder,script_dir,sequence_column="Icore",features_names=None,plot_blosum=False,plot_umap=False):
    """
    Notes:
      - Mid-padding : https://www.nature.com/articles/s41598-020-71450-8
    :param pandas dataframe data: Contains Icore, immunodominance_score, immunodominance_score_scaled, training , partition and Rnk_EL
    :param args: Commmand line arguments
    :param storage_folder: Data location path
    """

    epitopes_list = data[[sequence_column]].values.tolist()
    epitopes_list = functools.reduce(operator.iconcat, epitopes_list, [])  # flatten list of lists
    seq_max_len = len(max(epitopes_list, key=len))
    epitopes_lens = np.array(list(map(len, epitopes_list)))
    unique_lens = list(set(epitopes_lens))
    corrected_aa_types = len(set().union(*epitopes_list))
    corrected_aa_types = [corrected_aa_types + 1 if len(unique_lens) > 1 else corrected_aa_types][0]
    if len(unique_lens) > 1: # Highlight: Pad the sequences (relevant when they differ in length)
        aa_dict = VegvisirUtils.aminoacid_names_dict(corrected_aa_types , zero_characters=["#"])
        epitopes_pad_result = VegvisirLoadUtils.SequencePadding(epitopes_list,seq_max_len,args.seq_padding).run()
        epitopes_padded, epitopes_padded_mask = zip(*epitopes_pad_result) #unpack list of tuples onto 2 lists
        blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(corrected_aa_types , args.subs_matrix,
                                                                                   zero_characters= ["#"],
                                                                                   include_zero_characters=True)

    else:
        aa_dict = VegvisirUtils.aminoacid_names_dict(corrected_aa_types)
        epitopes_padded = epitopes_padded_mask = list(map(lambda seq: list(seq),epitopes_list))
        blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(corrected_aa_types, args.subs_matrix,
                                                                                   zero_characters=["#"],
                                                                                   include_zero_characters=False)

    epitopes_array = np.array(epitopes_padded)
    if args.seq_padding == "replicated_borders":  # I keep it separately to avoid doing the np vectorized loop twice
        epitopes_array_int = np.vectorize(aa_dict.get)(epitopes_array)
        epitopes_array_mask = np.array(epitopes_padded_mask)
        epitopes_array_int_mask = np.vectorize(aa_dict.get)(epitopes_array_mask)
        epitopes_mask = epitopes_array_int_mask.astype(bool)
    else:
        epitopes_array_int = np.vectorize(aa_dict.get)(epitopes_array)
        epitopes_mask = epitopes_array_int.astype(bool)
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
    ksize = 3 #TODO: manage in args
    if not os.path.exists("{}/{}/similarities/percent_identity_mean.npy".format(storage_folder,args.dataset_name)):
        print("Epitopes similarity matrices not existing, calculating (approx 2-3 min) ....")
        VegvisirUtils.folders("{}/similarities".format(args.dataset_name), storage_folder)
        percent_identity_mean,cosine_similarity_mean,kmers_pid_similarity,kmers_cosine_similarity = VegvisirUtils.calculate_similarity_matrix(epitopes_array_blosum,seq_max_len,epitopes_mask,ksize=ksize)
        np.save("{}/{}/similarities/percent_identity_mean.npy".format(storage_folder,args.dataset_name), percent_identity_mean)
        np.save("{}/{}/similarities/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name), cosine_similarity_mean)
        np.save("{}/{}/similarities/kmers_pid_similarity_{}ksize.npy".format(storage_folder,args.dataset_name,ksize), kmers_pid_similarity)
        np.save("{}/{}/similarities/kmers_cosine_similarity_{}ksize.npy".format(storage_folder,args.dataset_name,ksize), kmers_cosine_similarity)
    else:
        print("Loading pre-calculated epitopes similarity matrices located at {}".format("{}/{}/similarities/".format(storage_folder,args.dataset_name)))
        percent_identity_mean = np.load("{}/{}/similarities/percent_identity_mean.npy".format(storage_folder,args.dataset_name))
        cosine_similarity_mean = np.load("{}/{}/similarities/cosine_similarity_mean.npy".format(storage_folder,args.dataset_name))
        kmers_pid_similarity = np.load("{}/{}/similarities/kmers_pid_similarity_{}ksize.npy".format(storage_folder, args.dataset_name,ksize))
        kmers_cosine_similarity = np.load("{}/{}/similarities/kmers_cosine_similarity_{}ksize.npy".format(storage_folder, args.dataset_name,ksize))

    if not os.path.exists("{}/{}/similarities/HEATMAP_percent_identity_mean.png".format(storage_folder,args.dataset_name)):
        VegvisirPlots.plot_heatmap(percent_identity_mean, "Percent Identity","{}/{}/similarities/HEATMAP_percent_identity_mean.png".format(storage_folder,args.dataset_name))
        VegvisirPlots.plot_heatmap(cosine_similarity_mean, "Cosine similarity","{}/{}/similarities/HEATMAP_cosine_similarity_mean.png".format(storage_folder,args.dataset_name))
        VegvisirPlots.plot_heatmap(kmers_pid_similarity, "Kmers ({}) percent identity".format(ksize),"{}/{}/similarities/HEATMAP_kmers_pid_similarity_{}ksize.png".format(storage_folder, args.dataset_name,ksize))
        VegvisirPlots.plot_heatmap(kmers_cosine_similarity, "Kmers ({}) cosine similarity".format(ksize),"{}/{}/similarities/HEATMAP_kmers_cosine_similarity_{}ksize.png".format(storage_folder, args.dataset_name,ksize))

    calculate_partitions = False
    if calculate_partitions: #TODO: move elsewhere
        import umap,hdbscan
        cosine_umap = umap.UMAP(n_components=6).fit_transform(cosine_similarity_mean)
        clustering = DBSCAN(eps=0.3, min_samples=1,metric="euclidean",algorithm="auto",p=3).fit(cosine_umap) #eps 4
        #clustering = hdbscan.HDBSCAN(min_cluster_size=1, gen_min_span_tree=True).fit(cosine_similarity_mean)
        labels = np.unique(clustering.labels_,return_counts=True)
        #TODO: Separate the most disimilar sequences onto the test dataset. Select labels with counts lower than 20

    #Highlight: Reattatch partition, identifier, label, immunodominance score
    labels = data[["target_corrected"]].values.tolist()
    identifiers = data.index.values.tolist() #TODO: reset index in process data function?
    partitions = data[["partition"]].values.tolist()
    training = data[["training"]].values.tolist()
    confidence_scores = data["confidence_score"].values.tolist()
    immunodominance_scores = data[["immunodominance_score"]].values.tolist()

    if plot_umap:
        VegvisirPlots.plot_umap1(epitopes_array_blosum_norm, immunodominance_scores, storage_folder, args, "Blosum Norm","UMAP_blosum_norm_{}_immunodominance_score".format(sequence_column))
        VegvisirPlots.plot_umap1(epitopes_array_blosum_norm, labels, storage_folder, args, "Blosum Norm","UMAP_blosum_norm_{}".format(sequence_column))
        VegvisirPlots.plot_umap1(percent_identity_mean,labels,storage_folder,args,"Percent Identity Mean","UMAP_percent_identity_mean_{}".format(sequence_column))
        VegvisirPlots.plot_umap1(cosine_similarity_mean, labels, storage_folder, args, "Cosine similarity Mean","UMAP_cosine_similarity_mean_{}".format(sequence_column))
        VegvisirPlots.plot_umap1(kmers_pid_similarity, labels, storage_folder, args, "Kmers Percent Identity Mean","UMAP_kmers_percent_identity_{}".format(sequence_column))
        VegvisirPlots.plot_umap1(kmers_cosine_similarity, labels, storage_folder, args, "Kmers Cosine similarity Mean","UMAP_kmers_cosine_similarity_{}".format(sequence_column))

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
                            percent_identity_mean=percent_identity_mean,
                            cosine_similarity_mean=cosine_similarity_mean,
                            kmers_pid_similarity=kmers_pid_similarity,
                            kmers_cosine_similarity=kmers_cosine_similarity,
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

def prepare_nnalign(args,storage_folder,data,column_names):
    data_train = data[data["training"] == True][column_names]
    data_valid = data[data["training"] == False][column_names]
    data_train = data_train.astype({'partition': 'int'})
    data_valid.drop("partition",axis=1,inplace=True)
    data_train["Icore"].to_csv("{}/{}/viral_seq2logo.tsv".format(storage_folder,args.dataset_name),sep="\t",index=False,header=None)

    data_train.to_csv("{}/{}/viral_nnalign_input_train.tsv".format(storage_folder,args.dataset_name),sep="\t",index=False,header=None)
    data_valid.to_csv("{}/{}/viral_nnalign_input_valid.tsv".format(storage_folder,args.dataset_name), sep="\t",index=False,header=None) #TODO: Header None?
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

