"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import json
import os,random
import pickle
import warnings
from argparse import Namespace

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict,namedtuple
try:
    import umap
except:
    print("Could not import UMAP because of numpy incompatibility")
    pass
import scipy
import torch
from sklearn.cluster import DBSCAN
import vegvisir.nnalign as VegvisirNNalign
import vegvisir.utils as VegvisirUtils
import vegvisir.similarities as VegvisirSimilarities
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.plots as VegvisirPlots
import vegvisir.mutual_information as VegvisirMI
from collections import Counter
plt.style.use('ggplot')
DatasetInfo = namedtuple("DatasetInfo",["script_dir","storage_folder","data_array_raw","data_array_int","data_array_int_mask",
                                        "data_array_blosum_encoding",
                                        "data_array_blosum_encoding_mask",
                                        "data_array_onehot_encoding",
                                        "data_array_onehot_encoding_mask",
                                        "data_array_blosum_norm","blosum",
                                        "n_data","seq_max_len",
                                        "max_len",
                                        "corrected_aa_types",
                                        "input_dim",
                                        "positional_weights",
                                        "positional_weights_mask",
                                        "percent_identity_mean",
                                        "cosine_similarity_mean",
                                        "kmers_pid_similarity",
                                        "kmers_cosine_similarity",
                                        "features_names",
                                        "blosum_weighted",
                                        "unique_lens",
                                        "immunomodulate_dataset"])
DatasetDivision = namedtuple("DatasetDivision",["all","all_mask","positives","positives_mask","positives_idx","negatives","negatives_mask","negatives_idx","high_confidence_negatives",
                                                "high_confidence_negatives_mask","high_conf_negatives_idx"])
SimilarityResults = namedtuple("SimilarityResults",["positional_weights","percent_identity_mean","cosine_similarity_mean","kmers_pid_similarity","kmers_cosine_similarity"])

def available_datasets():
    """Prints the available datasets"""
    #TODO: Print description
    datasets = {0:"custom_dataset",
                1:"viral_dataset3",
                2:"viral_dataset4",
                3:"viral_dataset5",
                4:"viral_dataset6",
                5:"viral_dataset7",
                6:"viral_dataset8",
                7:"viral_dataset9",
                8:"viral_dataset10",
                9:"viral_dataset11",
                10:"viral_dataset12",
                11:"viral_dataset13",
                12:"viral_dataset14",
                13:"viral_dataset15",
                }
    return datasets

def select_dataset(dataset_name,script_dir,args,results_dir,update=True):
    """Selects from available datasets
    :param dataset_name: dataset of choice
    :param script_dir: Path from where the scriptis being executed
    :param update: If true it will download and update the most recent version of the dataset
    """

    func_dict = {"immunomodulate_dataset":immunomodulate_dataset,
                 "custom_dataset_binders":custom_dataset,
                 "custom_dataset_random":custom_dataset,
                 "custom_dataset_random_icore_non_anchor":custom_dataset,
                 "viral_dataset3":viral_dataset3,
                 "viral_dataset4":viral_dataset4,
                 "viral_dataset5":viral_dataset5,
                 "viral_dataset6": viral_dataset6,
                 "viral_dataset7":viral_dataset7,
                 "viral_dataset8":viral_dataset8,
                 "viral_dataset9":viral_dataset9,
                 "viral_dataset10":viral_dataset10,
                 "viral_dataset11":viral_dataset11,
                 "viral_dataset12":viral_dataset12,
                 "viral_dataset13":viral_dataset13,
                 "viral_dataset14":viral_dataset14,
                 "viral_dataset15":viral_dataset15,
                 }
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) #finds the /data folder of the repository
    if args.learning_type == "semisupervised":
        if args.dataset_name in ["viral_dataset3","viral_dataset7","viral_dataset9"]:
            raise ValueError("Please select viral_dataset6 or viral_dataset8 or viral_dataset10 for semisupervised learning, else select supervised learning")
    if args.dataset_name in ["viral_dataset6","viral_dataset8","viral_dataset10","viral_dataset11"]:
        assert args.learning_type == "semisupervised", "Please select semisupervised learning for dataset {}".format(args.dataset_name)
    if args.dataset_name == "viral_dataset12" and args.learning_type not in ["unsupervised","supervised"]:
        raise ValueError("Please select either unsupervised or supervised learning types, despite this dataset being built upon unobserved data points (they will be randomly assigned a target value")
    if args.dataset_name == "viral_dataset12" and args.test:
        raise ValueError("No testing is available for this dataset, only validation. Please set args.validate == True and args.test == False")
    #if args.dataset_name == "viral_dataset14" and args.predefined_partitions:
    #    raise ValueError("Please select args.predefined_partitions == False")

    dataset_load_fx = lambda f,current_path,storage_folder,args,results_dir,corrected_parameters: lambda current_path,storage_folder,args,results_dir,corrected_parameters: f(current_path,storage_folder,args,results_dir,corrected_parameters)
    data_load_function = dataset_load_fx(func_dict[dataset_name],script_dir,storage_folder,args,results_dir,None)
    dataset = data_load_function(script_dir,storage_folder,args,results_dir,None)

    if args.immunomodulate:
        data_immunomodulate_load_function = dataset_load_fx(func_dict["immunomodulate_dataset"],script_dir,storage_folder,args,results_dir,(dataset.corrected_aa_types,dataset.unique_lens))
        dataset_immunomodulate = data_immunomodulate_load_function(script_dir, storage_folder, args, results_dir,(dataset.corrected_aa_types,dataset.unique_lens))
        dataset = dataset._replace(immunomodulate_dataset=dataset_immunomodulate)
    print("Data retrieved")
    return dataset

def select_filters(args):
    if args.filter_kmers:
        if args.sequence_type == "Icore_non_anchor":
            kmer_size = 7
        else:
            kmer_size = 9
    else:
        kmer_size = 9

    filters_dict = {"filter_kmers":[args.filter_kmers,kmer_size,args.sequence_type], #Icore_non_anchor #Highlight: Remmeber to use 8!!
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

def group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,unobserved=False,no_subjects_test=False,plot_histograms=True):
    """Filters, groups and prepares the files from the viral_dataset*() functions
    :param bool unobserved: Indicates the presence of unlabelled/unobserved data , which is missing the number of patients tested
    :param bool no_subjects_test: Boolean to circumvent that some of the labelled data is missing the information on the number of tested people (no_subjects)"""
    if filters_dict["filter_ntested"][0]:
        # Highlight: Filter the points with low subject count and only keep if all "negative"
        threshold = filters_dict["filter_ntested"][1]
        nprefilter = data.shape[0]
        data = data[(data["Assay_number_of_subjects_tested"] > threshold)]
        nfiltered = data.shape[0]
        dataset_info_file.write("Filter 1: Icores with number of subjects lower than {}. Drops {} data points, remaining {} \n".format(threshold,nprefilter - nfiltered, nfiltered))

    data["immunodominance_score"] = data["Assay_number_of_subjects_responded"] / data["Assay_number_of_subjects_tested"]
    #data = data.fillna({"immunodominance_score":0}) #this assumes that if the immunodominance score is nan is because there is not enough information and then assigns it to 0, perhaps no good
    data["immunodominance_score"] = data["immunodominance_score"].fillna(data["target"])

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
        data[[use_column]] = data[[use_column]].fillna('AAAAA')  # Replace nan values when there is some sequence missing (some icore_non_anchor are missing)
        data = data[data[use_column].apply(lambda x: len(x) == kmer_size)]
        nfiltered = data.shape[0]
        dataset_info_file.write("Filter 2: {} whose length is different than 9. Drops {} data points, remaining {} \n".format(use_column,kmer_size,nprefilter-nfiltered,nfiltered))
    #Highlight: Strict target reassignment if immunodominance score is available
    if unobserved:
        data.loc[(data["immunodominance_score_scaled"] <= 0.) & (data["target_corrected"] != 2), "target_corrected"] = 0  # ["target"] = 0. #Strict target reassignment
        # print(data_a.sort_values(by="immunodominance_score", ascending=True)[["immunodominance_score","target"]])
        data.loc[(data["immunodominance_score_scaled"] > 0.) & (data["target_corrected"] != 2), "target_corrected"] = 1.
        if no_subjects_test:
            data.loc[(data["training"] == False), "target_corrected"] = data["target"]  # No re-assignment for the test data

    elif no_subjects_test:
        data.loc[(data["training"] == True) & (data["immunodominance_score_scaled"] <= 0.), "target_corrected"] = 0  # ["target"] = 0. #Strict target reassignment
        # print(data_a.sort_values(by="immunodominance_score", ascending=True)[["immunodominance_score","target"]])
        data.loc[(data["training"] == True) & (data["immunodominance_score_scaled"] > 0.), "target_corrected"] = 1.
        data.loc[(data["training"] == False),"target_corrected"] = data["target"] # No re-assignment for the test data
    else:
        data.loc[data["immunodominance_score_scaled"] <= 0., "target_corrected"] = 0  # ["target"] = 0. #Strict target reassignment
        # print(data_a.sort_values(by="immunodominance_score", ascending=True)[["immunodominance_score","target"]])
        data.loc[data["immunodominance_score_scaled"] > 0., "target_corrected"] = 1.

    #Highlight: Filter data points with low confidence (!= 0, 1)
    if filters_dict["filter_lowconfidence"][0]:
        nprefilter = data.shape[0]
        data = data[data["immunodominance_score"].isin([0.,1.])]
        nfiltered = data.shape[0]
        dataset_info_file.write("Filter 3: Remove data points with low immunodominance score. Drops {} data points, remaining {} \n".format(nprefilter - nfiltered, nfiltered))

    #Highlight: Annotate which data points have low confidence
    data = set_confidence_score(data)


    if plot_histograms:
        name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
        VegvisirPlots.plot_data_information(data, filters_dict, storage_folder, args, name_suffix)

    #Highlight: Prep data to run in NNalign
    if args.run_nnalign:
        prepare_nnalign(args,storage_folder,data,[filters_dict["filter_kmers"][2],"target_corrected","partition"])

    return data.copy()

def check_overlap(data):
    train = data[data["training"] == True]
    test = data[data["training"] == False]
    overlap = pd.merge(train,test,on = ["Icore"],how="inner")
    assert overlap.size == 0, "Test data points included in the train dataset"

def save_alleles(data,storage_folder,args):
    alleles_counts = data.value_counts("allele")
    most_common_allele = alleles_counts.index[0] #allele with most conserved positions HLA-B0707, the most common allele here is also ok
    alleles_list = alleles_counts.index.tolist()
    alleles_list = "".join(list(map(lambda seq: seq[:7] + ":" + seq[7:] + ",",alleles_list)))
    alleles_file = open("{}/{}/alleles_list.txt".format(storage_folder,args.dataset_name),"w+")
    n=20*11
    splitted_alleles = [alleles_list[i:i + n] for i in range(0, len(alleles_list), n)]
    for segment in splitted_alleles:
        alleles_file.write("{}\n".format(segment))
    return most_common_allele

def immunomodulate_dataset(script_dir,storage_folder,args,results_dir,corrected_parameters):
    """Builds a Vegvisir dataset that can be integrated with the model's pipeline
    NOTES:
        Immunomodulated sequences paper: https://doi.org/10.1074/jbc.M503060200
    """

    if args.immunomodulate_path is not None and os.path.exists(args.immunomodulate_path):
            immunomodulate_data = pd.read_csv(args.immunomodulate_path,names=["Icore"])
            immunomodulate_data = immunomodulate_data.drop_duplicates(subset=['Icore'])
            assert immunomodulate_data.shape[0] <= 200, "It does not make sense to modify so many peptides, please set the number of unique peptides below 200"
            immunomodulate_data["Icore_non_anchor"] = immunomodulate_data["Icore"]
            immunomodulate_data["training"] = False
            immunomodulate_data["target_corrected"] = 1 #fakely assigning to 1 (otherwise I have to correct more things downstairs)
            immunomodulate_data["partition"] = 80
            immunomodulate_data["immunodominance_score"] = 0
            immunomodulate_data["immunodominance_score_scaled"] = 0
            immunomodulate_data["confidence_score"] = 0
            immunomodulate_data["org_name"] = 0
            immunomodulate_data["allele"] = "HLA-A0101"
            filters_dict, analysis_mode = select_filters(args)
            data_info = process_data(immunomodulate_data, args, storage_folder, script_dir, analysis_mode, filters_dict,corrected_parameters=corrected_parameters)

            return data_info



    else:
        raise ValueError("You have selected args.immunomodulate == True, however args.immunomodulate_path has not recieved any valid path."
                         "\n Switch off args.immunomodulate or input a valid path")

def custom_dataset(script_dir,storage_folder,args,results_dir,corrected_parameters=None): #TODO: Finish
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

    if args.train_path is not None:
        print("Loading your train sequences")
        if os.path.exists(args.train_path):
            train_data = pd.read_csv("{}".format(args.train_path),sep="\t")
            #train_data = train_data[["Icore", "target_corrected"]] #TODO:target
            if "Icore" not in train_data.columns:
                train_data = train_data.rename(columns={"Icore_non_anchor":"Icore"})

            if "partition" in train_data.columns:
                train_data = train_data[["Icore","target_corrected","partition"]]
            else:
                train_data = train_data[["Icore","target_corrected"]]
                train_data["partition"] = np.random.choice(np.arange(5),size=train_data.shape[0],replace=True)
                unique_partitions = train_data["partition"].value_counts()

            #train_data = train_data.replace(columns={"target":"target_corrected"})
            train_data["training"] = True #np.random.choice([True,False],train_data.shape[0],p=[0.8,0.2]) #For random training & testing
            if "target_corrected" in train_data.columns:
                pass
            else:
                warnings.warn("You did not provide a <target_corrected> column TRAIN dataset (args.train_path), therefore I am setting them to random values")
                train_data["target_corrected"] = np.random.choice([0,1],train_data.shape[0])
            train_data["immunodominance_score"] = 0
            train_data["confidence_score"] = 0
            train_data["org_name"] = 0
            if "allele_encoded" in train_data.columns:
                pass
            else:
                train_data["allele_encoded"] = np.random.choice([0,1],train_data.shape[0])
            name_suffix = "_custom_dataset"
            #allele_counts_dict = train_data["allele"].value_counts().to_dict()
            #allele_dict = dict(zip(allele_counts_dict.keys(), list(range(len(allele_counts_dict.keys())))))
            #allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))), allele_counts_dict.keys()))
            #json.dump(allele_dict_reversed,open('{}/{}/alleles_dict.txt'.format(storage_folder, args.dataset_name), 'w'), indent=2)
            filters_dict,analysis_mode = select_filters(args)


    if args.test_path is not None:
        print("Loading your test sequences")
        if os.path.exists(args.test_path):
            test_data = pd.read_csv("{}".format(args.test_path),sep="\t")
            if "Icore" not in test_data.columns:
                test_data = test_data.rename(columns={"Icore_non_anchor":"Icore"})
            else:
                test_data = test_data[["Icore"]] #TODO: If incore non anchor not in columns make it equal to icore
            test_data = test_data.dropna(axis=1)
            test_data = test_data.drop_duplicates(subset=['Icore'])
            test_data["training"] = False
            if "target_corrected" in test_data.columns:
                pass
            else:
                warnings.warn("You did not provide a <target_corrected> column for your test dataset (args.test_path), therefore I am setting them to random values")
                test_data["target_corrected"] = np.random.choice([0,1],test_data.shape[0])
            test_data["partition"] = None
            test_data["immunodominance_score"] = 0
            test_data["confidence_score"] = 0
            test_data["org_name"] = 0
            if "allele_encoded" in test_data.columns:
                pass
            else:
                test_data["allele_encoded"] = np.random.choice([0, 1], test_data.shape[0])


    if args.train_path is not None and args.test_path is not None:

            data = pd.concat([train_data,test_data],axis=0).reset_index()

    elif args.train_path is not None:
        print("You did not provide a test dataset, therefore predictions/training will be made only in your train dataset. Setting args.test to False")
        args_dict = vars(args)
        args_dict["test"] = False
        args_dict["validate"] = False
        args = Namespace(**args_dict)
        data = train_data
    else:
        print("You did not provide a train dataset, therefore predictions/training will be made only in your test dataset. Setting args.train to False")
        args_dict = vars(args)
        args_dict["test"] = False
        args_dict["validate"] = False
        args = Namespace(**args_dict)
        data = test_data


    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset3(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    data = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    data.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    data = data.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]
    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = save_alleles(data,storage_folder,args)


    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        # Group data by Icore, therefore the alleles are grouped

        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count)) # messy when same number of counts
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training"]].nth(0) #returns first hit
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])#selects mode and returns first hit in case of equal counts
        data_b = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor","allele","partition", "target", "training"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor

        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
        data_species = data_species.groupby('Icore', as_index=False)[["allele", "org_name"]].agg(lambda x: list(x)[0])

    #else:
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys()))))) #TODO: Replace with allele encoding based on sequential information
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)
    data_species["allele_encoded"] = data_species["allele"]
    data_species.replace({"allele_encoded": allele_dict},inplace=True)


    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file)

    data = pd.merge(data,data_species, on=['Icore'], how='left')

    unique_values = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values))),unique_values))
    org_name_dict_reverse = dict(zip(unique_values,list(range(len(unique_values)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name":org_name_dict_reverse})

    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)

    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset4(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    data_features = pd.read_csv("{}/common_files/dataset_all_features.tsv".format(storage_folder),sep="\s+",index_col=0)
    data_partitions = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder),sep = "\t",index_col=0)
    data_partitions.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    data_partitions = data_partitions[["Icore","Icore_non_anchor","allele","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","partition","target","training"]]
    data_features = data_features[["Icore","allele","Pred_netstab","prot_inst_index","prot_median_iupred_score_long","prot_molar_excoef_cys_cys_bond","prot_p[q3_E]_netsurfp","prot_p[q3_C]_netsurfp","prot_rsa_netsurfp"]]
    features_names = data_features.columns.tolist()[2:]
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]
    data = pd.merge(data_features,data_partitions, on=['Icore',"allele"], how='outer')
    data = data.dropna(subset=["Icore_non_anchor","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training","Pred_netstab"]).reset_index(drop=True)

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = save_alleles(data,storage_folder,args)

    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        # Group data by Icore
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        data_b = data.groupby('Icore', as_index=False)[features_names].agg(lambda x: sum(list(x)) / len(list(x)))
        #data_c = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        #data_c = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        data_c = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor","allele","partition", "target", "training"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor

        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
        data = pd.merge(data, data_c, on='Icore', how='outer')
        data_species = data_species.groupby('Icore', as_index=False)[["allele", "org_name"]].agg(lambda x: list(x)[0])

    #else:
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)
    #features_names.append("allele_encoded")

    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file)
    data = pd.merge(data,data_species, on=['Icore'], how='left')

    unique_values = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values))),unique_values))
    org_name_dict_reverse = dict(zip(unique_values,list(range(len(unique_values)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name":org_name_dict_reverse})

    name_suffix = "__".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)


    VegvisirPlots.plot_features_histogram(data,features_names,"{}/{}".format(storage_folder,args.dataset_name),name_suffix)
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict,features_names=features_names)

    return data_info

def viral_dataset5(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    data_partitions = pd.read_csv("{}/common_files/dataset_target_correction_artificial_negatives.tsv".format(storage_folder,args.dataset_name),sep = "\t")

    data_partitions.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition","confidence_score","immunodominance","Length","Rnk_EL"]
    data = data_partitions[["Icore","Icore_non_anchor","allele","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","partition","target","training"]]
    #mask = data["Assay_number_of_subjects_tested"] == 0
    #Highlight: Dealing with the artificial data points
    mask = data["Assay_number_of_subjects_tested"] == 0
    data.loc[mask,"training"] = 0
    data = data.copy() #needed to avoid funky column re-assignation warning errors
    data.loc[:,'training'] = data.loc[:,'training'].replace({1: True, 0: False})
    data = data.dropna(subset=["Icore_non_anchor","training"]).reset_index(drop=True)
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]
    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = save_alleles(data,storage_folder,args)


    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]



    if filters_dict["group_alleles"][0]:
        # Group data by Icore only, therefore the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        data_b = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor","allele","partition", "target", "training"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor

        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
        data_species = data_species.groupby('Icore', as_index=False)[["allele", "org_name"]].agg(lambda x: list(x)[0])

    #else:
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)

    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file)

    mask2 = data["Assay_number_of_subjects_tested"] == 0

    warnings.warn("Setting low confidence score to the artificial negatives in the test dataset")
    data.loc[mask2,"confidence_score"] = 0.6
    data.loc[mask2,"immunodominance_score"] = np.nan
    data = pd.merge(data,data_species, on=['Icore'], how='left')

    unique_values = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values))),unique_values))
    org_name_dict_reverse = dict(zip(unique_values,list(range(len(unique_values)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name":org_name_dict_reverse})

    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)

    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)



    return data_info

def viral_dataset6(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    data_observed = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    data_observed.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    data_observed = data_observed.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)
    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    data_unobserved_anchors = pd.read_csv("{}/common_files/dataset_artificial_peptides_from_proteins_partitioned_hla_anchors.tsv".format(storage_folder,args.dataset_name),sep = "\t") #Highlight: The sequences from the labelled dataset have been filtered for some reason
    data_unobserved_anchors = data_unobserved_anchors[["Icore","allele","Icore_non_anchor"]]
    data_unobserved = pd.read_csv("{}/common_files/dataset_artificial_peptides_from_proteins_partitioned_hla.tsv".format(storage_folder,args.dataset_name),sep = "\s+") #Highlight: The sequences from the labelled dataset have been filtered for some reason
    data_unobserved.columns = ["Icore", "target", "partition", "source","allele"]
    data_unobserved_partition = data_unobserved[["Icore","partition","source"]]
    data_unobserved = data_unobserved[(data_unobserved["source"] == "artificial")]
    #data_unobserved = data_unobserved.sample(n=args.num_unobserved,replace=False)
    unique_partitions = data_unobserved["partition"].value_counts()
    if args.num_unobserved < data_unobserved.shape[0]:
        if args.num_unobserved != 0:
            data_unobserved_list = []
            for partition in unique_partitions.keys():
                data_unobserved_i = data_unobserved.loc[(data_unobserved["partition"] == partition)]
                data_unobserved_list.append(data_unobserved_i.sample(n=int((args.num_unobserved / len(unique_partitions.keys()))),replace=False))
            data_unobserved = pd.concat(data_unobserved_list, axis=0)
        else:
            print("Not using unobserved data points")
            data_unobserved = pd.DataFrame(columns=data_unobserved.columns)
    data = data_observed.merge(data_unobserved, on=['Icore', 'allele'], how='outer',suffixes=('_observed', '_unobserved'))
    data = data.drop(["target_unobserved","partition_observed","partition_unobserved"],axis=1)
    data = data.merge(data_unobserved_partition,on=["Icore"],how="left",suffixes=('_observed', '_unobserved'))
    data = data.drop(["source_observed"],axis=1)
    data = data.rename(columns={"target_observed": "target","source_unobserved":"source"})
    data.loc[(data["source"] == "artificial"), "target"] = 2

    #Highlight: Incorporate the Icore-non_anchors
    data = data.merge(data_unobserved_anchors,on=["Icore","allele"],how="left")
    data["Icore_non_anchor_x"] = data["Icore_non_anchor_x"].fillna(data["Icore_non_anchor_y"])
    data = data.drop(["Icore_non_anchor_y"],axis=1)
    data = data.rename(columns={"Icore_non_anchor_x":"Icore_non_anchor"})
    #Filter peptides/Icores with unknown aminoacids "X"
    data = data[~data['Icore'].str.contains("X")]

    # anchors_per_allele = pd.read_csv("{}/anchor_info_content/anchors_per_allele_average.txt".format(storage_folder),sep="\s+",header=0)
    # anchors_per_allele=anchors_per_allele[anchors_per_allele[["allele"]].isin(data_allele_types).any(axis=1)]
    # allele_anchors_dict = dict(zip(anchors_per_allele["allele"],anchors_per_allele["anchors"]))
    most_common_allele = save_alleles(data,storage_folder,args)

    #data_allele_types = allele_counts.index.tolist() #alleles present in this dataset
    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        print("Grouping alleles")
        # Group data by Icore, therefore, the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        data_b = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor","allele","partition", "target", "training"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor
        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
        data_species = data_species.groupby('Icore', as_index=False)[["allele", "org_name"]].agg(lambda x: list(x)[0])

    #else:
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)

    data["training"] = True
    data.loc[(data["target"] == 2), "target_corrected"] = 2
    data.loc[(data["target"] == 2), "confidence_score"] = 0
    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,unobserved=True)

    data.loc[(data["target"] == 2), "target_corrected"] = 2
    data.loc[(data["target"] == 2), "confidence_score"] = 0
    data = pd.merge(data,data_species, on=['Icore'], how='left')

    unique_values = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values))), unique_values))
    org_name_dict_reverse = dict(zip(unique_values, list(range(len(unique_values)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name": org_name_dict_reverse})

    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)

    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset7(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    new_partitions = pd.read_csv("{}/common_files/Viruses_db_partitions_notest.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    #new_partitions.columns = ["Icore","allele","Core","Of","Gp","Gl","Ip","Il","Rnk_EL","org_id","uniprot_id","target","start_prot","Icore_non_anchor","partition"]


    data = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    data.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]

    data = data.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)

    #Highlight: Replace the training and partition columns for the new ones

    data = data.merge(new_partitions, on=['Icore', 'allele'], how='left',suffixes=('_old', '_new'))
    data = data.loc[:, ~data.columns.str.endswith('_old') | (data.columns == 'partition_old')  ] #remove all columns ending with _old
    data = data.rename(columns={"Icore_non_anchor_new": "Icore_non_anchor", "target_new": "target","partition_new":"partition"})

    #Highlight: add species
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = save_alleles(data,storage_folder,args)


    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]


    if filters_dict["group_alleles"][0]:
        # Group data by Icore, therefore the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition","partition_old", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        data_b = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor","allele","partition","target","training"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor

        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
        data_species = data_species.groupby('Icore', as_index=False)[["allele", "org_name"]].agg(lambda x: list(x)[0])

    #else:
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys()))))) #TODO: Replace with allele encoding based on sequential information
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)


    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file)

    data = pd.merge(data,data_species, on=['Icore'], how='left',suffixes=("_a","_b"))
    unique_values = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values))), unique_values))
    org_name_dict_reverse = dict(zip(unique_values, list(range(len(unique_values)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name": org_name_dict_reverse})

    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)

    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset8(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    data_observed = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    data_observed.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    data_observed = data_observed.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)
    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    data_unobserved_anchors = pd.read_csv("{}/common_files/dataset_artificial_peptides_from_proteins_partitioned_hla_anchors.tsv".format(storage_folder,args.dataset_name),sep = "\t") #Highlight: The sequences from the labelled dataset have been filtered for some reason
    data_unobserved_anchors = data_unobserved_anchors[["Icore","allele","Icore_non_anchor"]]
    data_unobserved = pd.read_csv("{}/common_files/dataset_artificial_peptides_from_proteins_partitioned_hla.tsv".format(storage_folder,args.dataset_name),sep = "\s+") #Highlight: The sequences from the labelled dataset have been filtered for some reason
    data_unobserved.columns = ["Icore", "target", "partition", "source","allele"]
    data_unobserved_partition = data_unobserved[["Icore","partition","source"]]
    data_unobserved = data_unobserved[(data_unobserved["source"] == "artificial")]
    unique_partitions = data_unobserved["partition"].value_counts()
    if args.num_unobserved < data_unobserved.shape[0]:
        if args.num_unobserved != 0:
            data_unobserved_list = []
            for partition in unique_partitions.keys():
                data_unobserved_i = data_unobserved.loc[(data_unobserved["partition"] == partition)]
                data_unobserved_list.append(data_unobserved_i.sample(n=int((args.num_unobserved/len(unique_partitions.keys()))),replace=False))
            data_unobserved = pd.concat(data_unobserved_list,axis=0)
        else:
            assert args.num_unobserved != 0, "Please, if num-unobserved== 0 use the supervised method instead with viral_dataset3"
            data_unobserved = pd.DataFrame(columns=data_unobserved.columns)

    #data_unobserved = data_unobserved.sample(n=args.num_unobserved,replace=False)
    data = data_observed.merge(data_unobserved, on=['Icore', 'allele'], how='outer',suffixes=('_observed', '_unobserved'))
    data = data.drop(["target_unobserved","partition_observed","partition_unobserved"],axis=1)
    data = data.merge(data_unobserved_partition,on=["Icore"],how="left",suffixes=('_observed', '_unobserved'))
    data = data.drop(["source_observed"],axis=1)
    data = data.rename(columns={"target_observed": "target","source_unobserved":"source"})
    data.loc[(data["source"] == "artificial"), "target"] = 2
    data.loc[(data["source"] != "dataset_test"), "training"] = True
    data.loc[(data["source"] == "dataset_test"), "training"] = False

    #Highlight: Incorporate the Icore-non_anchors
    data = data.merge(data_unobserved_anchors,on=["Icore","allele"],how="left")
    data["Icore_non_anchor_x"] = data["Icore_non_anchor_x"].fillna(data["Icore_non_anchor_y"])
    data = data.drop(["Icore_non_anchor_y"],axis=1)
    data = data.rename(columns={"Icore_non_anchor_x":"Icore_non_anchor"})

    #Filter peptides/Icores with unknown aminoacids "X"
    data = data[~data['Icore'].str.contains("X")]

    # anchors_per_allele = pd.read_csv("{}/anchor_info_content/anchors_per_allele_average.txt".format(storage_folder),sep="\s+",header=0)
    # anchors_per_allele=anchors_per_allele[anchors_per_allele[["allele"]].isin(data_allele_types).any(axis=1)]
    # allele_anchors_dict = dict(zip(anchors_per_allele["allele"],anchors_per_allele["anchors"]))
    most_common_allele = save_alleles(data,storage_folder,args)


    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        print("Grouping alleles")
        # Group data by Icore, therefore, the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        data_b = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor","allele","partition", "target", "training"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor

        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')

        # data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #remove nans and retun first occurrence
        # data_b  = data_b[data_b['Icore_non_anchor'].notna()]
        # data_c = data.groupby('Icore', as_index=False)[["Icore","partition", "target", "training","org_name"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #return first occurence
        #
        # # Reattach info on training
        # data = pd.merge(data_a, data_b, on='Icore', how='right')
        # data = pd.merge(data,data_c,on="Icore",how="left")



        data_species = data_species.groupby('Icore', as_index=False)[[ "org_name"]].agg(lambda x: list(x)[0])

    #else:
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)


    data.loc[(data["target"] == 2), "target_corrected"] = 2
    data.loc[(data["target"] == 2), "confidence_score"] = 0

    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,unobserved=True)
    data.loc[(data["target"] == 2), "target_corrected"] = 2
    data.loc[(data["target"] == 2), "confidence_score"] = 0
    data = pd.merge(data,data_species, on=['Icore'], how='left')

    unique_values = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values))),unique_values))
    org_name_dict_reverse = dict(zip(unique_values,list(range(len(unique_values)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name":org_name_dict_reverse})

    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)

    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset9(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    org_name:
    org_family:
    org_genus:
    kingdom:
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
    new_partitions = pd.read_csv("{}/common_files/Viruses_db_partitions_notest.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    #new_partitions.columns = ["Icore","allele","Core","Of","Gp","Gl","Ip","Il","Rnk_EL","org_id","uniprot_id","target","start_prot","Icore_non_anchor","partition"]
    data = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    data.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]

    data = data.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)

    #Highlight: Replace the training and partition columns for the new ones

    data = data.merge(new_partitions, on=['Icore', 'allele'], how='left',suffixes=('_old', '_new'))
    data = data.loc[:, ~data.columns.str.endswith('_old')] #remove all columns ending with _old
    data = data.rename(columns={"Icore_non_anchor_new": "Icore_non_anchor", "target_new": "target","partition_new":"partition"})

    #Highlight: add species information
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    #Highlight: Add new test dataset

    #/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/NEW_pMHC_test.csv
    new_test_dataset = pd.read_csv("{}/common_files/NEW_pMHC_test.csv".format(storage_folder,args.dataset_name),sep = ",")
    new_test_dataset_anchors = pd.read_csv("{}/common_files/new_test_nonanchor.csv".format(storage_folder,args.dataset_name),sep = ",")
    new_test_dataset_anchors = new_test_dataset_anchors[["Icore","Icore_non_anchor"]]
    new_test_dataset_immunogenicity = pd.read_csv("{}/common_files/new_test_nonanchor_immunodominance.csv".format(storage_folder,args.dataset_name),sep=",")
    new_test_dataset_immunogenicity = new_test_dataset_immunogenicity[["Icore","allele","subjects_tested","subjects_responded"]] #"Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"
    new_test_dataset_immunogenicity.columns = ["Icore","alelle","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"]
    new_test_dataset = pd.merge(new_test_dataset,new_test_dataset_anchors,on=["Icore"],how="left")
    new_test_dataset = pd.merge(new_test_dataset,new_test_dataset_immunogenicity,on=["Icore"],how="left")


    test_mode_dict = {0:"test_virus",
                      1:"test_bacteria",
                      2:"test_cancer"}

    new_test_dataset["training"] = False
    #new_test_dataset["target_corrected"]  = new_test_dataset["target"]
    test_mode = test_mode_dict[0]
    if test_mode=="test_virus":
        #new_test_dataset = new_test_dataset[(new_test_dataset['org_name'].str.contains("virus")) | (new_test_dataset['org_name'].str.contains("SARS-CoV2"))]
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Viruses"))]
    elif test_mode == "test_bacteria":
        warnings.warn("Using epitopes from bacteria as test")
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Bacteria"))]
    elif test_mode == "test_cancer":
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Eukaryota"))]

    data["training"] = True #Highlight: At this point, everything is training (also the old test)
    data = pd.merge(data,new_test_dataset, on=['Icore',"allele","training","target"], how='outer',suffixes=('_a', '_b')) #merege the new test dataset

    data["Icore_non_anchor"] = data["Icore_non_anchor_a"].fillna(data["Icore_non_anchor_b"])
    data["Assay_number_of_subjects_responded"] = data["Assay_number_of_subjects_responded_a"].fillna(data["Assay_number_of_subjects_responded_b"])
    data["Assay_number_of_subjects_tested"] = data["Assay_number_of_subjects_tested_a"].fillna(data["Assay_number_of_subjects_tested_b"])

    data = data.drop(["Icore_non_anchor_a", "Icore_non_anchor_b"], axis=1)
    data = data.drop(["Assay_number_of_subjects_tested_a", "Assay_number_of_subjects_tested_b"], axis=1)
    data = data.drop(["Assay_number_of_subjects_responded_a", "Assay_number_of_subjects_responded_b"], axis=1)
    data = data.drop("kingdom",axis=1)

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = save_alleles(data,storage_folder,args)

    if filters_dict["filter_alleles"][0]:#Highlight: pick only the data corresponding to the most frequent allele
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        # Group data by Icore, therefore the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))

        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training","org_name"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training","org_name"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        data_b = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor","allele"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor

        data_b  = data_b[data_b['Icore_non_anchor'].notna()]
        data_c = data.groupby('Icore', as_index=False)[["Icore","partition", "target", "training","org_name"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #return first occurence
        data = pd.merge(data_a, data_b, on='Icore', how='right')

        data = pd.merge(data,data_c,on="Icore",how="left")
        data_species = data_species.groupby('Icore', as_index=False)[["org_name"]].agg(lambda x: list(x)[0])


    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)


    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,no_subjects_test=False,plot_histograms=False)


    if filters_dict["group_alleles"][0]:
        data = pd.merge(data,data_species, on=['Icore'], how='left',suffixes=('_a', '_b'))
    else:
        data = pd.merge(data,data_species, on=['Icore',"allele"], how='left',suffixes=('_a', '_b'))

    data["org_name"] = data["org_name_a"].fillna(data["org_name_b"])
    data.drop(["org_name_b","org_name_a"],axis=1,inplace=True)
    data.loc[(data["training"] == False), "confidence_score"] = 0

    unique_values_species = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values_species))), unique_values_species))
    org_name_dict_reverse = dict(zip(unique_values_species, list(range(len(unique_values_species)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name": org_name_dict_reverse})
    # nan_rows = data[data["confidence_score"].isna()]
    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])

    VegvisirPlots.plot_data_information_reduced(data, filters_dict, storage_folder, args, name_suffix)

    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)

    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)
    return data_info

def viral_dataset10(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    data_observed = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    data_observed.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    data_observed = data_observed.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)
    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    data_unobserved_anchors = pd.read_csv("{}/common_files/dataset_artificial_peptides_from_proteins_partitioned_hla_anchors.tsv".format(storage_folder,args.dataset_name),sep = "\t") #Highlight: The sequences from the labelled dataset have been filtered for some reason
    data_unobserved_anchors = data_unobserved_anchors[["Icore","allele","Icore_non_anchor"]]
    data_unobserved = pd.read_csv("{}/common_files/dataset_artificial_peptides_from_proteins_partitioned_hla.tsv".format(storage_folder,args.dataset_name),sep = "\s+") #Highlight: The sequences from the labelled dataset have been filtered for some reason
    data_unobserved.columns = ["Icore", "target", "partition", "source","allele"]
    data_unobserved_partition = data_unobserved[["Icore", "partition", "source"]] #this is in the right order, do not change
    data_unobserved = data_unobserved[(data_unobserved["source"] == "artificial")]
    #data_unobserved = data_unobserved.sample(n=args.num_unobserved,replace=False)
    unique_partitions = data_unobserved["partition"].value_counts()
    if args.num_unobserved < data_unobserved.shape[0]:
        if args.num_unobserved != 0:
            data_unobserved_list = []
            for partition in unique_partitions.keys():
                data_unobserved_i = data_unobserved.loc[(data_unobserved["partition"] == partition)]
                data_unobserved_list.append(
                    data_unobserved_i.sample(n=int((args.num_unobserved / len(unique_partitions.keys()))),replace=False))
            data_unobserved = pd.concat(data_unobserved_list, axis=0)
        else:
            print("Not using unobserved data points")
            data_unobserved = pd.DataFrame(columns=data_unobserved.columns)
    data = data_observed.merge(data_unobserved, on=['Icore', 'allele'], how='outer',suffixes=('_observed', '_unobserved'))
    data = data.drop(["target_unobserved","partition_observed","partition_unobserved"],axis=1)
    data = data.merge(data_unobserved_partition,on=["Icore"],how="left",suffixes=('_observed', '_unobserved'))
    data = data.drop(["source_observed"],axis=1)
    data = data.rename(columns={"target_observed": "target","source_unobserved":"source"})
    data.loc[(data["source"] == "artificial"), "target"] = 2
    #Highlight: Incorporate the Icore-non_anchors
    data = data.merge(data_unobserved_anchors,on=["Icore","allele"],how="left")
    data["Icore_non_anchor_x"] = data["Icore_non_anchor_x"].fillna(data["Icore_non_anchor_y"])
    data = data.drop(["Icore_non_anchor_y"],axis=1)
    data = data.rename(columns={"Icore_non_anchor_x":"Icore_non_anchor"})

    #Filter peptides/Icores with unknown aminoacids "X"
    data = data[~data['Icore'].str.contains("X")]

    # anchors_per_allele = pd.read_csv("{}/anchor_info_content/anchors_per_allele_average.txt".format(storage_folder),sep="\s+",header=0)
    # anchors_per_allele=anchors_per_allele[anchors_per_allele[["allele"]].isin(data_allele_types).any(axis=1)]
    # allele_anchors_dict = dict(zip(anchors_per_allele["allele"],anchors_per_allele["anchors"]))
    most_common_allele = save_alleles(data,storage_folder,args)

    #data_allele_types = allele_counts.index.tolist() #alleles present in this dataset


    #Highlight: Add new test dataset

    #/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/NEW_pMHC_test.csv
    new_test_dataset = pd.read_csv("{}/common_files/NEW_pMHC_test.csv".format(storage_folder,args.dataset_name),sep = ",")
    new_test_dataset_anchors = pd.read_csv("{}/common_files/new_test_nonanchor.csv".format(storage_folder,args.dataset_name),sep = ",")
    new_test_dataset_anchors = new_test_dataset_anchors[["Icore","Icore_non_anchor"]]
    new_test_dataset_immunogenicity = pd.read_csv("{}/common_files/new_test_nonanchor_immunodominance.csv".format(storage_folder,args.dataset_name),sep=",")
    new_test_dataset_immunogenicity = new_test_dataset_immunogenicity[["Icore","allele","subjects_tested","subjects_responded"]] #"Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"
    new_test_dataset_immunogenicity.columns = ["Icore","alelle","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"]
    new_test_dataset = pd.merge(new_test_dataset,new_test_dataset_anchors,on=["Icore"],how="left")
    new_test_dataset = pd.merge(new_test_dataset,new_test_dataset_immunogenicity,on=["Icore"],how="left")

    #print(new_test_dataset[["Assay_number_of_subjects_tested"]].value_counts())


    test_mode_dict = {0:"test_virus",
                      1:"test_bacteria",
                      2:"test_cancer"}

    new_test_dataset["training"] = False
    #new_test_dataset["target_corrected"]  = new_test_dataset["target"]
    test_mode = test_mode_dict[0]

    if test_mode=="test_virus":
        #new_test_dataset = new_test_dataset[(new_test_dataset['org_name'].str.contains("virus")) | (new_test_dataset['org_name'].str.contains("SARS-CoV2"))]
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Viruses"))]
    elif test_mode == "test_bacteria":
        warnings.warn("Using epitopes from bacteria as test")
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Bacteria"))]
    elif test_mode == "test_cancer":
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Eukaryota"))]
    data["training"] = True #Highlight: This time we do not keep track of the training data
    data = pd.merge(data,new_test_dataset, on=['Icore',"allele","training","target"], how='outer',suffixes=('_a', '_b'))
    data["Icore_non_anchor"] = data["Icore_non_anchor_a"].fillna(data["Icore_non_anchor_b"])
    data["Assay_number_of_subjects_responded"] = data["Assay_number_of_subjects_responded_a"].fillna(data["Assay_number_of_subjects_responded_b"])
    data["Assay_number_of_subjects_tested"] = data["Assay_number_of_subjects_tested_a"].fillna(data["Assay_number_of_subjects_tested_b"])

    data = data.drop(["Icore_non_anchor_a","Icore_non_anchor_b"],axis=1)
    data = data.drop(["Assay_number_of_subjects_tested_a","Assay_number_of_subjects_tested_b"],axis=1)
    data = data.drop(["Assay_number_of_subjects_responded_a","Assay_number_of_subjects_responded_b"],axis=1)

    data = data.drop("kingdom",axis=1)
    if filters_dict["filter_alleles"][0]:
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        print("Grouping alleles")
        # Group data by Icore, therefore, the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training","org_name"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","allele","partition", "target", "training","org_name"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])

        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
        data_species = data_species.groupby('Icore', as_index=False)[["org_name"]].agg(lambda x: list(x)[0])

    #else:
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)




    data.loc[(data["target"] == 2), "target_corrected"] = 2
    data.loc[(data["target"] == 2), "confidence_score"] = 0
    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,unobserved=True,no_subjects_test=False)


    # test_data =  data[data["training"] == False]
    # print(test_data.shape)
    # print(test_data["target"].tolist())
    # print(test_data["target_corrected"].tolist())
    # #nan_rows = test_data[test_data["Assay_number_of_subjects_tested"].isna()]
    # #print(nan_rows)
    # exit()

    data.loc[(data["target"] == 2), "target_corrected"] = 2
    data.loc[(data["target"] == 2), "confidence_score"] = 0
    data = pd.merge(data, data_species, on=['Icore'], how='left', suffixes=('_a', '_b'))
    data["org_name"] = data["org_name_a"].fillna(data["org_name_b"])
    data.drop(["org_name_b","org_name_a"],axis=1,inplace=True)




    data.loc[(data["training"] == False), "confidence_score"] = 0
    unique_values = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values))), unique_values))
    org_name_dict_reverse = dict(zip(unique_values, list(range(len(unique_values)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name": org_name_dict_reverse})



    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t")

    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset11(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    data_observed = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    data_observed.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    data_observed = data_observed.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)

    #Highlight: Extract the test dataset with the original labels
    test_data = data_observed[data_observed["training"] == False]
    #Highlight: Change the label of the test dataset to unobserved (2), to do we also have to change "Assay_number_of_subjects_tested" and "Assay_number_of_subjects_responded" to 1 and 2 respectively, to obtain target_corrected = 2
    data_observed.loc[(data_observed["training"] == False), "target"] = 2
    data_observed.loc[(data_observed["training"] == False), "Assay_number_of_subjects_tested"] = 1
    data_observed.loc[(data_observed["training"] == False), "Assay_number_of_subjects_responded"] = 2

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    data_unobserved_anchors = pd.read_csv("{}/common_files/dataset_artificial_peptides_from_proteins_partitioned_hla_anchors.tsv".format(storage_folder,args.dataset_name),sep = "\t")
    data_unobserved_anchors = data_unobserved_anchors[["Icore","allele","Icore_non_anchor"]]

    data_unobserved = pd.read_csv("{}/common_files/dataset_artificial_peptides_from_proteins_partitioned_hla.tsv".format(storage_folder,args.dataset_name),sep = "\s+") #Highlight: The sequences from the labelled dataset have been filtered for some reason
    data_unobserved.columns = ["Icore", "target", "partition", "source","allele"]
    data_unobserved_partition = data_unobserved[["Icore","partition","source"]]
    data_unobserved = data_unobserved[(data_unobserved["source"] == "artificial")]
    #data_unobserved = data_unobserved.sample(n=args.num_unobserved,replace=False)
    unique_partitions = data_unobserved["partition"].value_counts()
    if args.num_unobserved < data_unobserved.shape[0]:
        if args.num_unobserved != 0:
            data_unobserved_list = []
            for partition in unique_partitions.keys():
                data_unobserved_i = data_unobserved.loc[(data_unobserved["partition"] == partition)]
                data_unobserved_list.append(data_unobserved_i.sample(n=int((args.num_unobserved/len(unique_partitions.keys()))),replace=False))
            data_unobserved = pd.concat(data_unobserved_list,axis=0)
        else:
            print("Not using unobserved data points")
            data_unobserved = pd.DataFrame(columns=data_unobserved.columns)

    data = data_observed.merge(data_unobserved, on=['Icore', 'allele'], how='outer',suffixes=('_observed', '_unobserved'))

    data = data.drop(["target_unobserved","partition_observed","partition_unobserved"],axis=1)
    data = data.merge(data_unobserved_partition,on=["Icore"],how="left",suffixes=('_observed', '_unobserved'))
    data = data.drop(["source_observed"],axis=1)
    data = data.rename(columns={"target_observed": "target","source_unobserved":"source"})
    data.loc[(data["source"] == "artificial"), "target"] = 2


    #Highlight: Incorporate the Icore-non_anchors
    data = data.merge(data_unobserved_anchors,on=["Icore","allele"],how="left")
    data["Icore_non_anchor_x"] = data["Icore_non_anchor_x"].fillna(data["Icore_non_anchor_y"])
    data = data.drop(["Icore_non_anchor_y"],axis=1)
    data = data.rename(columns={"Icore_non_anchor_x":"Icore_non_anchor"})

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
        test_data = test_data[test_data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        print("Grouping alleles")
        # Group data by Icore, therefore, the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
        data_species = data_species.groupby('Icore', as_index=False)[["allele", "org_name"]].agg(lambda x: list(x)[0])

        # Group data by Icore, therefore, the alleles are grouped
        test_data_a = test_data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))
        #test_data_b = test_data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        test_data_b = test_data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])

        # Reattach info on training
        test_data = pd.merge(test_data_a, test_data_b, on='Icore', how='outer')

    #else:
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)

    test_data["allele_encoded"] = test_data["allele"]
    test_data.replace({"allele_encoded": allele_dict},inplace=True)

    data["training"] = True
    data.loc[(data["target"] == 2), "target_corrected"] = 2
    data.loc[(data["target"] == 2), "confidence_score"] = 0
    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,unobserved=True,plot_histograms=False)

    data.loc[(data["target"] == 2), "target_corrected"] = 2
    data.loc[(data["target"] == 2), "confidence_score"] = 0
    data = pd.merge(data,data_species, on=['Icore'], how='left')

    unique_values = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values))), unique_values))
    org_name_dict_reverse = dict(zip(unique_values, list(range(len(unique_values)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name": org_name_dict_reverse})

    #Highlight: Reattaching the test data
    test_data = group_and_filter(test_data,args,storage_folder,filters_dict,dataset_info_file,unobserved=False,plot_histograms=False)
    data = pd.concat([data,test_data],axis=0).reset_index(drop=True) #very important to reset the index otherwise it messes up later things

    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])
    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t")
    VegvisirPlots.plot_data_information(data, filters_dict, storage_folder, args, name_suffix)
    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset12(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    # data_observed = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    # data_observed.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]
    # data_observed = data_observed.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)
    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    data_unobserved = pd.read_csv("{}/common_files/dataset_artificial_peptides_from_proteins_partitioned_hla_anchors.tsv".format(storage_folder,args.dataset_name),sep = "\t") #Highlight: The sequences from the labelled dataset have been filtered for some reason
    data_unobserved = data_unobserved[(data_unobserved["source"] == "artificial")]
    data_unobserved = data_unobserved[["Icore","target","partition","allele","Icore_non_anchor"]]
    #data_unobserved = data_unobserved.sample(n=args.num_unobserved,replace=False)
    unique_partitions = data_unobserved["partition"].value_counts()
    if args.num_unobserved < data_unobserved.shape[0]:
        if args.num_unobserved != 0:
            data_unobserved_list = []
            for partition in unique_partitions.keys():
                data_unobserved_i = data_unobserved.loc[(data_unobserved["partition"] == partition)]
                data_unobserved_list.append(data_unobserved_i.sample(n=int((args.num_unobserved / len(unique_partitions.keys()))),replace=False))
            data_unobserved = pd.concat(data_unobserved_list, axis=0)
        else:
            raise ValueError("Please select args.num_unobserved > 0, at least 5000 datapoints, since this dataset is built solely in unobserved/unlabelled data points")
            data_unobserved = pd.DataFrame(columns=data_unobserved.columns)

    data = data_unobserved #I know this is creating an unnecesary copy of the data, it is here for readibility
    data["target"] = 2 #need to keep it "unobserved" for the grouping function
    data["confidence_score"] = 0
    data["Assay_number_of_subjects_responded"] = 0 #this gives immunodominance score of 0
    data["Assay_number_of_subjects_tested"] = 1
    #Highlight: Incorporate the Icore-non_anchors
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
        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])

        # Reattach info on training
        data = pd.merge(data_a, data_b, on='Icore', how='outer')
        data_species = data_species.groupby('Icore', as_index=False)[["org_name"]].agg(lambda x: list(x)[0])

    #else:
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys())))))
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)

    data["training"] = True #in this dataset wwe only do validation, not testing
    data["target_corrected"] = 2 #need to keep it "unobserved" for the grouping function

    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,unobserved=True,plot_histograms=False)

    a = np.random.choice(np.array([0,1]),p=[0.75,0.25],size=(data.shape[0]))


    data["target"] = np.random.choice(np.array([0,1]),p=[0.75,0.25],size=(data.shape[0])) #randomly assigning targets
    data["target_corrected"] = data["target"] #need to re-assign
    data["confidence_score"] = 0
    data["org_name"] = "unobserved"


    unique_values = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values))), unique_values))
    org_name_dict_reverse = dict(zip(unique_values, list(range(len(unique_values)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name": org_name_dict_reverse})

    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key, val in filters_dict.items()])
    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder, args.dataset_name, name_suffix),sep="\t")
    VegvisirPlots.plot_data_information(data, filters_dict, storage_folder, args, name_suffix)

    #print(data[data["confidence_score"] > 0.7]["target_corrected"].value_counts())
    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)

    return data_info

def viral_dataset13(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    org_name:
    org_family:
    org_genus:
    kingdom:
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
    new_partitions = pd.read_csv("{}/common_files/Viruses_db_partitions_notest.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    #new_partitions.columns = ["Icore","allele","Core","Of","Gp","Gl","Ip","Il","Rnk_EL","org_id","uniprot_id","target","start_prot","Icore_non_anchor","partition"]
    data = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    data.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]

    data = data.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)

    #Highlight: Replace the training and partition columns for the new ones

    data = data.merge(new_partitions, on=['Icore', 'allele'], how='left',suffixes=('_old', '_new'))
    data = data.loc[:, ~data.columns.str.endswith('_old')] #remove all columns ending with _old
    data = data.rename(columns={"Icore_non_anchor_new": "Icore_non_anchor", "target_new": "target","partition_new":"partition"})

    #Highlight: add species information
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    #Highlight: Add new test dataset

    #/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/NEW_pMHC_test.csv
    new_test_dataset = pd.read_csv("{}/common_files/test_more_peptides.csv".format(storage_folder,args.dataset_name),sep = ",")
    new_test_dataset_immunogenicity = pd.read_csv("{}/common_files/new_test_nonanchor_immunodominance.csv".format(storage_folder,args.dataset_name),sep=",")
    new_test_dataset_immunogenicity = new_test_dataset_immunogenicity[["Icore","allele","subjects_tested","subjects_responded"]] #"Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"
    new_test_dataset_immunogenicity.columns = ["Icore","alelle","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"]
    new_test_dataset = pd.merge(new_test_dataset,new_test_dataset_immunogenicity,on=["Icore"],how="left")



    test_mode_dict = {0:"test_virus",
                      1:"test_bacteria",
                      2:"test_cancer"}

    new_test_dataset["training"] = False
    #new_test_dataset["target_corrected"]  = new_test_dataset["target"]
    test_mode = test_mode_dict[0]
    if test_mode=="test_virus":
        #new_test_dataset = new_test_dataset[(new_test_dataset['org_name'].str.contains("virus")) | (new_test_dataset['org_name'].str.contains("SARS-CoV2"))]
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Viruses"))]
    elif test_mode == "test_bacteria":
        warnings.warn("Using epitopes from bacteria as test")
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Bacteria"))]
    elif test_mode == "test_cancer":
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Eukaryota"))]

    data["training"] = True #Highlight: At this point, everything is training (also the old test)
    data = pd.merge(data,new_test_dataset, on=['Icore',"allele","training","target"], how='outer',suffixes=('_a', '_b')) #merege the new test dataset

    data["Icore_non_anchor"] = data["Icore_non_anchor_a"].fillna(data["Icore_non_anchor_b"])
    data["Assay_number_of_subjects_responded"] = data["Assay_number_of_subjects_responded_a"].fillna(data["Assay_number_of_subjects_responded_b"])
    data["Assay_number_of_subjects_tested"] = data["Assay_number_of_subjects_tested_a"].fillna(data["Assay_number_of_subjects_tested_b"])

    data = data.drop(["Icore_non_anchor_a", "Icore_non_anchor_b"], axis=1)
    data = data.drop(["Assay_number_of_subjects_tested_a", "Assay_number_of_subjects_tested_b"], axis=1)
    data = data.drop(["Assay_number_of_subjects_responded_a", "Assay_number_of_subjects_responded_b"], axis=1)
    data = data.drop("kingdom",axis=1)

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = save_alleles(data,storage_folder,args)

    if filters_dict["filter_alleles"][0]:#Highlight: pick only the data corresponding to the most frequent allele
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        # Group data by Icore, therefore the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))

        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training","org_name"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training","org_name"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        data_b = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor

        data_b  = data_b[data_b['Icore_non_anchor'].notna()]
        data_c = data.groupby('Icore', as_index=False)[["Icore","partition", "target", "training","org_name"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #return first occurence
        data = pd.merge(data_a, data_b, on='Icore', how='right')

        data = pd.merge(data,data_c,on="Icore",how="left")
        data_species = data_species.groupby('Icore', as_index=False)[["org_name"]].agg(lambda x: list(x)[0])


    #else: #Highlight: Do not group the alleles
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys()))))) #TODO: Replace with allele encoding based on sequential information
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)


    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,no_subjects_test=False,plot_histograms=False)



    if filters_dict["group_alleles"][0]:
        data = pd.merge(data,data_species, on=['Icore'], how='left',suffixes=('_a', '_b'))
    else:
        data = pd.merge(data,data_species, on=['Icore',"allele"], how='left',suffixes=('_a', '_b'))

    data["org_name"] = data["org_name_a"].fillna(data["org_name_b"])
    data.drop(["org_name_b","org_name_a"],axis=1,inplace=True)
    data.loc[(data["training"] == False), "confidence_score"] = 0

    unique_values_species = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values_species))), unique_values_species))
    org_name_dict_reverse = dict(zip(unique_values_species, list(range(len(unique_values_species)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name": org_name_dict_reverse})
    # nan_rows = data[data["confidence_score"].isna()]
    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])

    VegvisirPlots.plot_data_information_reduced(data, filters_dict, storage_folder, args, name_suffix)

    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)

    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)
    return data_info

def viral_dataset14(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    org_name:
    org_family:
    org_genus:
    kingdom:
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
    new_partitions = pd.read_csv("{}/common_files/Viruses_db_partitions_notest.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    #new_partitions.columns = ["Icore","allele","Core","Of","Gp","Gl","Ip","Il","Rnk_EL","org_id","uniprot_id","target","start_prot","Icore_non_anchor","partition"]
    data = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    data.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]

    data = data.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)

    #Highlight: Replace the training and partition columns for the new ones

    data = data.merge(new_partitions, on=['Icore', 'allele'], how='left',suffixes=('_old', '_new'))
    data = data.loc[:, ~data.columns.str.endswith('_old')] #remove all columns ending with _old
    data = data.rename(columns={"Icore_non_anchor_new": "Icore_non_anchor", "target_new": "target","partition_new":"partition"})

    #Highlight: add species information
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    #Highlight: Add new test dataset

    #/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/NEW_pMHC_test.csv
    new_test_dataset = pd.read_csv("{}/common_files/NEW_pMHC_test.csv".format(storage_folder,args.dataset_name),sep = ",")
    new_test_dataset_anchors = pd.read_csv("{}/common_files/new_test_nonanchor.csv".format(storage_folder,args.dataset_name),sep = ",")
    new_test_dataset_anchors = new_test_dataset_anchors[["Icore","Icore_non_anchor"]]
    new_test_dataset_immunogenicity = pd.read_csv("{}/common_files/new_test_nonanchor_immunodominance.csv".format(storage_folder,args.dataset_name),sep=",")
    new_test_dataset_immunogenicity = new_test_dataset_immunogenicity[["Icore","allele","subjects_tested","subjects_responded"]] #"Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"
    new_test_dataset_immunogenicity.columns = ["Icore","alelle","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"]
    new_test_dataset = pd.merge(new_test_dataset,new_test_dataset_anchors,on=["Icore"],how="left")
    new_test_dataset = pd.merge(new_test_dataset,new_test_dataset_immunogenicity,on=["Icore"],how="left")

    test_mode_dict = {0:"test_virus",
                      1:"test_bacteria",
                      2:"test_cancer"}

    new_test_dataset["training"] = False
    #new_test_dataset["target_corrected"]  = new_test_dataset["target"]
    test_mode = test_mode_dict[0]
    if test_mode=="test_virus":
        #new_test_dataset = new_test_dataset[(new_test_dataset['org_name'].str.contains("virus")) | (new_test_dataset['org_name'].str.contains("SARS-CoV2"))]
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Viruses"))]
    elif test_mode == "test_bacteria":
        warnings.warn("Using epitopes from bacteria as test")
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Bacteria"))]
    elif test_mode == "test_cancer":
        new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Eukaryota"))]

    data["training"] = True #Highlight: At this point, everything is training (also the old old test)
    data = pd.merge(data,new_test_dataset, on=['Icore',"allele","training","target"], how='outer',suffixes=('_a', '_b')) #merege the new test dataset

    data = data[data["allele"].isin(["HLA-A2402","HLA-A2301","HLA-A2407"])]

    data["Icore_non_anchor"] = data["Icore_non_anchor_a"].fillna(data["Icore_non_anchor_b"])
    data["Assay_number_of_subjects_responded"] = data["Assay_number_of_subjects_responded_a"].fillna(data["Assay_number_of_subjects_responded_b"])
    data["Assay_number_of_subjects_tested"] = data["Assay_number_of_subjects_tested_a"].fillna(data["Assay_number_of_subjects_tested_b"])

    data = data.drop(["Icore_non_anchor_a", "Icore_non_anchor_b"], axis=1)
    data = data.drop(["Assay_number_of_subjects_tested_a", "Assay_number_of_subjects_tested_b"], axis=1)
    data = data.drop(["Assay_number_of_subjects_responded_a", "Assay_number_of_subjects_responded_b"], axis=1)
    data = data.drop("kingdom",axis=1)

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = save_alleles(data,storage_folder,args)

    if filters_dict["filter_alleles"][0]:#Highlight: pick only the data corresponding to the most frequent allele
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        # Group data by Icore, therefore the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))

        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training","org_name"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training","org_name"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        data_b = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor

        data_b  = data_b[data_b['Icore_non_anchor'].notna()]
        data_c = data.groupby('Icore', as_index=False)[["Icore","partition", "target","allele", "training","org_name"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #return first occurence
        data = pd.merge(data_a, data_b, on='Icore', how='right')
        data = pd.merge(data,data_c,on="Icore",how="left")
        data_species = data_species.groupby('Icore', as_index=False)[["org_name"]].agg(lambda x: list(x)[0])


    #else: #Highlight: Do not group the alleles
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys()))))) #TODO: Replace with allele encoding based on sequential information
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)


    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,no_subjects_test=False,plot_histograms=False)



    if filters_dict["group_alleles"][0]:
        data = pd.merge(data,data_species, on=['Icore'], how='left',suffixes=('_a', '_b'))
    else:
        data = pd.merge(data,data_species, on=['Icore',"allele"], how='left',suffixes=('_a', '_b'))

    data["org_name"] = data["org_name_a"].fillna(data["org_name_b"])
    data.drop(["org_name_b","org_name_a"],axis=1,inplace=True)
    data.loc[(data["training"] == False), "confidence_score"] = 0

    unique_values_species = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values_species))), unique_values_species))
    org_name_dict_reverse = dict(zip(unique_values_species, list(range(len(unique_values_species)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name": org_name_dict_reverse})
    # nan_rows = data[data["confidence_score"].isna()]
    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])

    VegvisirPlots.plot_data_information_reduced(data, filters_dict, storage_folder, args, name_suffix)

    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)



    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)



    return data_info

def viral_dataset15(script_dir,storage_folder,args,results_dir,corrected_parameters=None):
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
    org_name:
    org_family:
    org_genus:
    kingdom:
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

    #new_partitions.columns = ["Icore","allele","Core","Of","Gp","Gl","Ip","Il","Rnk_EL","org_id","uniprot_id","target","start_prot","Icore_non_anchor","partition"]
    data = pd.read_csv("{}/common_files/dataset_target.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)

    data.columns = ["allele","Icore","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","target","training","Icore_non_anchor","partition"]

    data = data.dropna(subset=["Assay_number_of_subjects_tested","Assay_number_of_subjects_responded","training"]).reset_index(drop=True)

    #Highlight: add species information
    data_species = pd.read_csv("{}/common_files/dataset_species.tsv".format(storage_folder),sep="\t")
    data_species = data_species.dropna(axis=1)
    data_species = data_species[["Icore","allele","org_name"]]

    #Highlight: Add new test dataset

    #/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/viral_dataset9/NEW_pMHC_test.csv
    new_test_dataset = pd.read_csv("{}/common_files/NEW_pMHC_test.csv".format(storage_folder,args.dataset_name),sep = ",")
    new_test_dataset_immunogenicity = pd.read_csv("{}/common_files/new_test_nonanchor_immunodominance.csv".format(storage_folder,args.dataset_name),sep=",")
    new_test_dataset_immunogenicity = new_test_dataset_immunogenicity[["Icore","allele","subjects_tested","subjects_responded"]] #"Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"
    new_test_dataset_immunogenicity.columns = ["Icore","alelle","Assay_number_of_subjects_tested","Assay_number_of_subjects_responded"]
    new_test_dataset = pd.merge(new_test_dataset,new_test_dataset_immunogenicity,on=["Icore"],how="left")
    #
    # test_mode_dict = {0:"test_virus",
    #                   1:"test_bacteria",
    #                   2:"test_cancer"}
    #
    # new_test_dataset["training"] = False
    # #new_test_dataset["target_corrected"]  = new_test_dataset["target"]
    # test_mode = test_mode_dict[0]
    # if test_mode=="test_virus":
    #     #new_test_dataset = new_test_dataset[(new_test_dataset['org_name'].str.contains("virus")) | (new_test_dataset['org_name'].str.contains("SARS-CoV2"))]
    #     new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Viruses"))]
    # elif test_mode == "test_bacteria":
    #     warnings.warn("Using epitopes from bacteria as test")
    #     new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Bacteria"))]
    # elif test_mode == "test_cancer":
    #     new_test_dataset = new_test_dataset[(new_test_dataset['kingdom'].str.contains("Eukaryota"))]

    data["training"] = True #Highlight: At this point, everything is training (also the old test)
    new_partitions = pd.read_csv("{}/common_files/trainvalidtest_new_partitions.tsv".format(storage_folder,args.dataset_name),sep = "\t",index_col=0)
    data = data.merge(new_partitions, on=['Icore'], how='left',suffixes=('_old', '_new'))
    data = data.merge(new_test_dataset, on=['Icore'], how='left',suffixes=('_old', '_new'))

    #Highlight: Replace the training and partition columns for the new ones

    data = data.loc[:, ~data.columns.str.endswith('_old')] #remove all columns ending with _old
    data = data.rename(columns={"Icore_non_anchor_new": "Icore_non_anchor", "target_new": "target","partition_new":"partition"})


    exit()


    data["Icore_non_anchor"] = data["Icore_non_anchor_a"].fillna(data["Icore_non_anchor_b"])
    data["Assay_number_of_subjects_responded"] = data["Assay_number_of_subjects_responded_a"].fillna(data["Assay_number_of_subjects_responded_b"])
    data["Assay_number_of_subjects_tested"] = data["Assay_number_of_subjects_tested_a"].fillna(data["Assay_number_of_subjects_tested_b"])

    data = data.drop(["Icore_non_anchor_a", "Icore_non_anchor_b"], axis=1)
    data = data.drop(["Assay_number_of_subjects_tested_a", "Assay_number_of_subjects_tested_b"], axis=1)
    data = data.drop(["Assay_number_of_subjects_responded_a", "Assay_number_of_subjects_responded_b"], axis=1)
    data = data.drop("kingdom",axis=1)

    filters_dict,analysis_mode = select_filters(args)
    json.dump(filters_dict, dataset_info_file, indent=2)

    most_common_allele = save_alleles(data,storage_folder,args)

    if filters_dict["filter_alleles"][0]:#Highlight: pick only the data corresponding to the most frequent allele
        data = data[data["allele"] == most_common_allele]

    if filters_dict["group_alleles"][0]:
        # Group data by Icore, therefore the alleles are grouped
        data_a = data.groupby('Icore', as_index=False)[["Assay_number_of_subjects_tested", "Assay_number_of_subjects_responded"]].agg(lambda x: sum(list(x)))

        #data_b = data.groupby('Icore', as_index=False)[["Icore_non_anchor","partition", "target", "training","org_name"]].agg(lambda x: max(set(list(x)), key=list(x).count))
        #data_b = data.groupby('Icore', as_index=False)[["Icore","Icore_non_anchor","partition", "target", "training","org_name"]].agg(lambda x: scipy.stats.mode(x,keepdims=True)[0][0])
        data_b = data.groupby('Icore', as_index=False)[['Icore',"Icore_non_anchor"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #exclude nans and return most common Icore non anchor

        data_b  = data_b[data_b['Icore_non_anchor'].notna()]
        data_c = data.groupby('Icore', as_index=False)[["Icore","partition", "target","allele", "training","org_name"]].agg(lambda srs: Counter(list(srs)).most_common(1)[0][0]) #return first occurence
        data = pd.merge(data_a, data_b, on='Icore', how='right')
        data = pd.merge(data,data_c,on="Icore",how="left")
        data_species = data_species.groupby('Icore', as_index=False)[["org_name"]].agg(lambda x: list(x)[0])


    #else: #Highlight: Do not group the alleles
    allele_counts_dict = data["allele"].value_counts().to_dict()
    allele_dict = dict(zip(allele_counts_dict.keys(),list(range(len(allele_counts_dict.keys()))))) #TODO: Replace with allele encoding based on sequential information
    allele_dict_reversed = dict(zip(list(range(len(allele_counts_dict.keys()))),allele_counts_dict.keys()))
    json.dump(allele_dict_reversed, open('{}/{}/alleles_dict.txt'.format(storage_folder,args.dataset_name), 'w'), indent=2)
    data["allele_encoded"] = data["allele"]
    data.replace({"allele_encoded": allele_dict},inplace=True)


    data = group_and_filter(data,args,storage_folder,filters_dict,dataset_info_file,no_subjects_test=False,plot_histograms=False)



    if filters_dict["group_alleles"][0]:
        data = pd.merge(data,data_species, on=['Icore'], how='left',suffixes=('_a', '_b'))
    else:
        data = pd.merge(data,data_species, on=['Icore',"allele"], how='left',suffixes=('_a', '_b'))

    data["org_name"] = data["org_name_a"].fillna(data["org_name_b"])
    data.drop(["org_name_b","org_name_a"],axis=1,inplace=True)
    data.loc[(data["training"] == False), "confidence_score"] = 0

    unique_values_species = pd.unique(data["org_name"])
    org_name_dict = dict(zip(list(range(len(unique_values_species))), unique_values_species))
    org_name_dict_reverse = dict(zip(unique_values_species, list(range(len(unique_values_species)))))
    pickle.dump(org_name_dict,open('{}/{}/org_name_dict.pkl'.format(storage_folder,args.dataset_name), 'wb'))
    data = data.replace({"org_name": org_name_dict_reverse})
    # nan_rows = data[data["confidence_score"].isna()]
    name_suffix = "_".join([key + "_" + "_".join([str(i) for i in val]) for key,val in filters_dict.items()])

    VegvisirPlots.plot_data_information_reduced(data, filters_dict, storage_folder, args, name_suffix)

    data.to_csv("{}/{}/dataset_target_corrected_{}.tsv".format(storage_folder,args.dataset_name,name_suffix),sep="\t",index=False)



    data_info = process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict)



    return data_info

def data_class_division(array,array_mask,idx,labels,confidence_scores):
    """
    Divide the dataset onto data points from positive, negative or high confident negatives
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
    high_conf_negatives_idx = (confidence_scores_ > 0.6)&(labels_ == 0)



    data_subdivision = DatasetDivision(all = array_,
                                       all_mask = mask_,
                                       positives=positives_arr,
                                       positives_mask=positives_arr_mask,
                                       positives_idx = labels_ == 1,
                                       negatives=negatives_arr,
                                       negatives_mask=negatives_arr_mask,
                                       negatives_idx=labels_ == 0,
                                       high_confidence_negatives=high_conf_negatives_arr,
                                       high_confidence_negatives_mask=high_conf_negatives_arr_mask,
                                       high_conf_negatives_idx=high_conf_negatives_idx)
    return data_subdivision

def build_exploration_folders(args,storage_folder,filters_dict):

    for mode in ["All","Train", "Test"]:
        if args.sequence_type == "Icore":
            VegvisirUtils.folders("all","{}/{}/similarities/{}/{}/diff_allele/same_len/9mers/neighbours1/".format(storage_folder,args.dataset_name,args.sequence_type, mode))
            VegvisirUtils.folders("positives","{}/{}/similarities/{}/{}/diff_allele/same_len/9mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
            VegvisirUtils.folders("negatives","{}/{}/similarities/{}/{}/diff_allele/same_len/9mers/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode),overwrite=False)
            VegvisirUtils.folders("highconfnegatives","{}/{}/similarities/{}/{}/diff_allele/same_len/9mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
        else:
            VegvisirUtils.folders("all","{}/{}/similarities/{}/{}/diff_allele/same_len/8mers/neighbours1/".format(storage_folder,args.dataset_name,args.sequence_type, mode))
            VegvisirUtils.folders("positives","{}/{}/similarities/{}/{}/diff_allele/same_len/8mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
            VegvisirUtils.folders("negatives","{}/{}/similarities/{}/{}/diff_allele/same_len/8mers/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode),overwrite=False)
            VegvisirUtils.folders("highconfnegatives","{}/{}/similarities/{}/{}/diff_allele/same_len/8mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)


        VegvisirUtils.folders("all","{}/{}/similarities/{}/{}/diff_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode))
        VegvisirUtils.folders("positives","{}/{}/similarities/{}/{}/diff_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
        VegvisirUtils.folders("negatives","{}/{}/similarities/{}/{}/diff_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
        VegvisirUtils.folders("highconfnegatives","{}/{}/similarities/{}/{}/diff_allele/diff_len/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode),overwrite=False)


        if args.sequence_type == "Icore":
            VegvisirUtils.folders("all", "{}/{}/similarities/{}/{}/same_allele/same_len/9mers/neighbours1".format(storage_folder, args.dataset_name, args.sequence_type, mode))
            VegvisirUtils.folders("positives","{}/{}/similarities/{}/{}/same_allele/same_len/9mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
            VegvisirUtils.folders("negatives","{}/{}/similarities/{}/{}/same_allele/same_len/9mers/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode),overwrite=False)
            VegvisirUtils.folders("highconfnegatives","{}/{}/similarities/{}/{}/same_allele/same_len/9mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
        else:
            VegvisirUtils.folders("all", "{}/{}/similarities/{}/{}/same_allele/same_len/8mers/neighbours1".format(storage_folder, args.dataset_name, args.sequence_type, mode))
            VegvisirUtils.folders("positives","{}/{}/similarities/{}/{}/same_allele/same_len/8mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)
            VegvisirUtils.folders("negatives","{}/{}/similarities/{}/{}/same_allele/same_len/8mers/neighbours1".format(storage_folder,args.dataset_name, args.sequence_type,mode),overwrite=False)
            VegvisirUtils.folders("highconfnegatives","{}/{}/similarities/{}/{}/same_allele/same_len/8mers/neighbours1".format(storage_folder,args.dataset_name,args.sequence_type, mode),overwrite=False)

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

def data_volumetrics(seq_max_len,epitopes_list,data,epitopes_array_mask,storage_folder,args,filters_dict,analysis_mode,plot_volumetrics=False,plot_covariance=False):
    if not os.path.exists("{}/{}/similarities/{}".format(storage_folder, args.dataset_name, args.sequence_type)):
        build_exploration_folders(args, storage_folder, filters_dict)
    else:
        print("Folder structure existing")
    if plot_volumetrics or plot_covariance:
        labels_arr = np.array(data[["target_corrected"]].values.tolist()).squeeze()
        training = data[["training"]].values.tolist()
        training = np.array(training).squeeze(-1)
        confidence_scores = np.array(data["confidence_score"].values.tolist())
        immunodominance_scores = np.array(data["immunodominance_score"].values.tolist())
        epitopes_array_raw = np.array(epitopes_list)

        epitopes_array_raw_division_train = data_class_division(epitopes_array_raw,
                                                                             epitopes_array_mask, training, labels_arr,
                                                                             confidence_scores)
        epitopes_array_raw_division_test = data_class_division(epitopes_array_raw, epitopes_array_mask,
                                                                            np.invert(training), labels_arr,
                                                                            confidence_scores)
    if plot_volumetrics:
        print("Plotting volumetrics analysis")


        #Highlight: Train
        volumetrics_dict_all_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_train.all.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_all_train,seq_max_len,immunodominance_scores[training],storage_folder,args,subfolders,tag="_immunodominance_scores")
        VegvisirPlots.plot_volumetrics(volumetrics_dict_all_train,seq_max_len,labels_arr[training],storage_folder,args,subfolders,tag="_labels")
        volumetrics_dict_positives_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_train.positives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Train/{}/neighbours1/positives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_positives_train, seq_max_len,None,storage_folder, args, subfolders)
        volumetrics_dict_negatives_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_train.negatives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Train/{}/neighbours1/negatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_negatives_train, seq_max_len,None, storage_folder, args, subfolders)
        volumetrics_dict_high_confidence_negatives_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_train.high_confidence_negatives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Train/{}/neighbours1/highconfnegatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_high_confidence_negatives_train, seq_max_len, None,storage_folder, args, subfolders)

        #Highlight: Test
        volumetrics_dict_all_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_test.all.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Test/{}/neighbours1/all".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_all_test, seq_max_len, immunodominance_scores[np.invert(training)],storage_folder, args, subfolders,tag="_immunodominance_scores")
        VegvisirPlots.plot_volumetrics(volumetrics_dict_all_test, seq_max_len, labels_arr[np.invert(training)],storage_folder, args, subfolders,tag="_labels")
        volumetrics_dict_positives_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_test.positives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Test/{}/neighbours1/positives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_positives_test, seq_max_len, None,storage_folder, args, subfolders)
        volumetrics_dict_negatives_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_test.negatives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Test/{}/neighbours1/negatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_negatives_test, seq_max_len,None, storage_folder, args, subfolders)
        volumetrics_dict_high_confidence_negatives_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_test.high_confidence_negatives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Test/{}/neighbours1/highconfnegatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_high_confidence_negatives_test, seq_max_len,None, storage_folder, args, subfolders)

    if plot_covariance:
        print("Plotting features covariance analysis")
        if args.dataset_name == "viral_dataset9":
            use_precomputed_features = False
        else:
            use_precomputed_features = True
        #Highlight: All
        subfolders = "{}/All/{}/neighbours1/all".format(args.sequence_type,analysis_mode)
        feature_embedded_peptides =  VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw.tolist(),storage_folder).aminoacid_embedding()
        feature_embedded_peptides_df = pd.DataFrame({"{}".format(args.sequence_type):epitopes_array_raw.tolist(), "Embedding":feature_embedded_peptides, "target_corrected":labels_arr})
        feature_embedded_peptides_df.to_csv("{}/{}/similarities/{}/EMBEDDED_epitopes.tsv".format(storage_folder,args.dataset_name,subfolders),sep="\t")

        features_dict_all = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw.tolist(),storage_folder).features_summary()
        VegvisirPlots.plot_features_covariance(epitopes_array_raw.tolist(),features_dict_all,seq_max_len,immunodominance_scores,storage_folder,args,subfolders,tag="_immunodominance_scores",use_precomputed_features=use_precomputed_features)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw.tolist(),features_dict_all,seq_max_len,labels_arr,storage_folder,args,subfolders,tag="_binary_labels",use_precomputed_features=use_precomputed_features)
        #Highlight: Train
        features_dict_all_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_train.all.tolist(),storage_folder).features_summary()
        subfolders = "{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_train.all.tolist(),features_dict_all_train,seq_max_len,immunodominance_scores[training],storage_folder,args,subfolders,tag="_immunodominance_scores",use_precomputed_features=use_precomputed_features)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_train.all.tolist(),features_dict_all_train,seq_max_len,labels_arr[training],storage_folder,args,subfolders,tag="_binary_labels",use_precomputed_features=use_precomputed_features)
        features_dict_positives_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_train.positives.tolist(),storage_folder).features_summary()
        subfolders = "{}/Train/{}/neighbours1/positives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_train.positives.tolist(),features_dict_positives_train, seq_max_len,immunodominance_scores[training][epitopes_array_raw_division_train.positives_idx],storage_folder, args, subfolders,tag="_immunodominance_scores",use_precomputed_features=use_precomputed_features)
        features_dict_negatives_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_train.negatives.tolist(),storage_folder).features_summary()
        subfolders = "{}/Train/{}/neighbours1/negatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_train.negatives.tolist(),features_dict_negatives_train, seq_max_len,immunodominance_scores[training][epitopes_array_raw_division_train.negatives_idx], storage_folder, args, subfolders,tag="_immunodominance_scores",use_precomputed_features=use_precomputed_features)
        features_dict_high_confidence_negatives_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_train.high_confidence_negatives.tolist(),storage_folder).features_summary()
        subfolders = "{}/Train/{}/neighbours1/highconfnegatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_train.high_confidence_negatives.tolist(),features_dict_high_confidence_negatives_train, seq_max_len, immunodominance_scores[training][epitopes_array_raw_division_train.high_conf_negatives_idx],storage_folder, args, subfolders,tag="_immunodominance_scores",use_precomputed_features=use_precomputed_features)
        #Highlight: Test
        features_dict_all_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_test.all.tolist(),storage_folder).features_summary()
        subfolders = "{}/Test/{}/neighbours1/all".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_test.all.tolist(),features_dict_all_test, seq_max_len, immunodominance_scores[np.invert(training)],storage_folder, args, subfolders,tag="_immunodominance_scores",use_precomputed_features=use_precomputed_features)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_test.all.tolist(),features_dict_all_test, seq_max_len, labels_arr[np.invert(training)],storage_folder, args, subfolders,tag="_binary_labels",use_precomputed_features=use_precomputed_features)
        features_dict_positives_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_test.positives.tolist(),storage_folder).features_summary()
        subfolders = "{}/Test/{}/neighbours1/positives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_test.positives.tolist(),features_dict_positives_test, seq_max_len,immunodominance_scores[np.invert(training)][epitopes_array_raw_division_test.positives_idx],storage_folder, args, subfolders,tag="_immunodominance_scores",use_precomputed_features=use_precomputed_features)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_test.positives.tolist(),features_dict_positives_test, seq_max_len,labels_arr[np.invert(training)][epitopes_array_raw_division_test.positives_idx],storage_folder, args, subfolders,tag="_binary_labels",use_precomputed_features=use_precomputed_features)
        features_dict_negatives_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_test.negatives.tolist(),storage_folder).features_summary()
        subfolders = "{}/Test/{}/neighbours1/negatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_test.negatives.tolist(),features_dict_negatives_test, seq_max_len,immunodominance_scores[np.invert(training)][epitopes_array_raw_division_test.negatives_idx], storage_folder, args, subfolders,tag="_immunodominance_scores",use_precomputed_features=use_precomputed_features)
        features_dict_high_confidence_negatives_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,epitopes_array_raw_division_test.high_confidence_negatives.tolist(),storage_folder).features_summary()
        subfolders = "{}/Test/{}/neighbours1/highconfnegatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_features_covariance(epitopes_array_raw_division_test.high_confidence_negatives.tolist(),features_dict_high_confidence_negatives_test,
                                               seq_max_len,immunodominance_scores[np.invert(training)][epitopes_array_raw_division_test.high_conf_negatives_idx],
                                               storage_folder, args, subfolders,tag="_immunodominance_scores",use_precomputed_features=use_precomputed_features)

def vector_analysis(seq_max_len,data,data_blosum,epitopes_array_mask,storage_folder,args,filters_dict,analysis_mode,analyse = True):
    if not os.path.exists("{}/{}/similarities/{}".format(storage_folder, args.dataset_name, args.sequence_type)):
        build_exploration_folders(args, storage_folder, filters_dict)
    else:
        print("Folder structure existing")

    labels_arr = np.array(data[["target_corrected"]].values.tolist()).squeeze()
    training = data[["training"]].values.tolist()
    training = np.array(training).squeeze(-1)
    confidence_scores = np.array(data["confidence_score"].values.tolist())
    immunodominance_scores = np.array(data["immunodominance_score"].values.tolist())

    data_blosum_division_train = data_class_division(data_blosum,
                                                                         epitopes_array_mask, training, labels_arr,
                                                                         confidence_scores)
    data_blosum_division_test = data_class_division(data_blosum, epitopes_array_mask,
                                                                        np.invert(training), labels_arr,
                                                                        confidence_scores)
    if analyse:
        print("Plotting vectorial analysis")
        #Highlight: Train
        volumetrics_dict_all_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,data_blosum_division_train.all.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_all_train,seq_max_len,immunodominance_scores[training],storage_folder,args,subfolders,tag="_immunodominance_scores")
        VegvisirPlots.plot_volumetrics(volumetrics_dict_all_train,seq_max_len,labels_arr[training],storage_folder,args,subfolders,tag="_labels")
        volumetrics_dict_positives_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,data_blosum_division_train.positives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Train/{}/neighbours1/positives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_positives_train, seq_max_len,None,storage_folder, args, subfolders)
        volumetrics_dict_negatives_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,data_blosum_division_train.negatives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Train/{}/neighbours1/negatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_negatives_train, seq_max_len,None, storage_folder, args, subfolders)
        volumetrics_dict_high_confidence_negatives_train = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,data_blosum_division_train.high_confidence_negatives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Train/{}/neighbours1/highconfnegatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_high_confidence_negatives_train, seq_max_len, None,storage_folder, args, subfolders)

        #Highlight: Test
        volumetrics_dict_all_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,data_blosum_division_test.all.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Test/{}/neighbours1/all".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_all_test, seq_max_len, immunodominance_scores[np.invert(training)],storage_folder, args, subfolders,tag="_immunodominance_scores")
        VegvisirPlots.plot_volumetrics(volumetrics_dict_all_test, seq_max_len, labels_arr[np.invert(training)],storage_folder, args, subfolders,tag="_labels")
        volumetrics_dict_positives_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,data_blosum_division_test.positives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Test/{}/neighbours1/positives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_positives_test, seq_max_len, None,storage_folder, args, subfolders)
        volumetrics_dict_negatives_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,data_blosum_division_test.negatives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Test/{}/neighbours1/negatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_negatives_test, seq_max_len,None, storage_folder, args, subfolders)
        volumetrics_dict_high_confidence_negatives_test = VegvisirUtils.CalculatePeptideFeatures(seq_max_len,data_blosum_division_test.high_confidence_negatives.tolist(),storage_folder).volumetrics_summary()
        subfolders = "{}/Test/{}/neighbours1/highconfnegatives".format(args.sequence_type, analysis_mode)
        VegvisirPlots.plot_volumetrics(volumetrics_dict_high_confidence_negatives_test, seq_max_len,None, storage_folder, args, subfolders)

def data_exploration(data,epitopes_array_blosum,epitopes_array_int,epitopes_array_mask,aa_dict,aa_list,blosum_norm,seq_max_len,storage_folder,args,corrected_aa_types,analysis_mode,filters_dict):

    if not os.path.exists("{}/{}/similarities/{}".format(storage_folder,args.dataset_name,args.sequence_type)):
        build_exploration_folders(args, storage_folder,filters_dict)
    else:
        print("Folder structure existing")

    plot_mi,plot_frequencies,plot_cosine_similarity = False,False,False
    #Highlight: Encode amino acid raw

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

    epitopes_array_blosum_division_all = data_class_division(epitopes_array_blosum,
                                                           epitopes_array_mask, np.ones(epitopes_array_blosum.shape[0]).astype(bool), labels_arr,
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
        print("Calculating  epitopes similarity matrices (this might take a while, 15 minutes for 10000 sequences) ....")

        # all_sim_results =VegvisirSimilarities.calculate_similarities_parallel(epitopes_array_blosum,
        #                                                               seq_max_len,
        #                                                               epitopes_array_mask,
        #                                                               storage_folder, args,
        #                                                              "{}/All/{}/neighbours1/all".format(args.sequence_type,analysis_mode),
        #                                                               ksize=ksize)

        negatives_sim_results =VegvisirSimilarities.calculate_similarities_parallel(epitopes_array_blosum_division_all.negatives,
                                                                      seq_max_len,
                                                                      epitopes_array_blosum_division_all.negatives_mask,
                                                                      storage_folder, args,
                                                                     "{}/All/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode),
                                                                      ksize=ksize)
        positives_sim_results = VegvisirSimilarities.calculate_similarities_parallel(
                                                                        epitopes_array_blosum_division_all.positives,
                                                                        seq_max_len,
                                                                        epitopes_array_blosum_division_all.positives_mask,
                                                                        storage_folder, args,
                                                                        "{}/All/{}/neighbours1/positives".format(args.sequence_type, analysis_mode),
                                                                        ksize=ksize)
        #Highlight: Train dataset
        #
        # train_idx_select, train_longest_array = sample_datapoints_mi(epitopes_array_int_division_train.positives,
        #                                                              epitopes_array_int_division_train.negatives)

        train_all_sim_results =VegvisirSimilarities.calculate_similarities_parallel(epitopes_array_blosum_division_train.all,
                                                                      seq_max_len,
                                                                      epitopes_array_blosum_division_train.all_mask,
                                                                      storage_folder, args,
                                                                     "{}/Train/{}/neighbours1/all".format(args.sequence_type,analysis_mode),
                                                                     ksize=ksize)
        train_positives_sim_results = VegvisirSimilarities.calculate_similarities_parallel(
            epitopes_array_blosum_division_train.positives,
            seq_max_len,
            epitopes_array_blosum_division_train.positives_mask,
            storage_folder, args,
            "{}/Train/{}/neighbours1/positives".format(args.sequence_type, analysis_mode),
            ksize=ksize)

        #if epitopes_array_int_division_train.negatives.shape[0] == train_longest_array:
        train_negatives_sim_results=VegvisirSimilarities.calculate_similarities_parallel(epitopes_array_blosum_division_train.negatives,
                                                                          seq_max_len,
                                                                          epitopes_array_blosum_division_train.negatives_mask,
                                                                          storage_folder, args,
                                                                         "{}/Train/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode),
                                                                          ksize=ksize)

        train_high_conf_sim_results=VegvisirSimilarities.calculate_similarities_parallel(epitopes_array_blosum_division_train.high_confidence_negatives,
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
        test_all_sim_results = VegvisirSimilarities.calculate_similarities_parallel(
            epitopes_array_blosum_division_test.all,
            seq_max_len,
            epitopes_array_blosum_division_test.all_mask,
            storage_folder, args,
            "{}/Test/{}/neighbours1/all".format(args.sequence_type, analysis_mode),
            ksize=ksize)

        test_positives_sim_results = VegvisirSimilarities.calculate_similarities_parallel(
            epitopes_array_blosum_division_test.positives,
            seq_max_len,
            epitopes_array_blosum_division_test.positives_mask,
            storage_folder, args,
            "{}/Test/{}/neighbours1/positives".format(args.sequence_type,analysis_mode),
            ksize=ksize)

        test_negatives_sim_results = VegvisirSimilarities.calculate_similarities_parallel(
            epitopes_array_blosum_division_test.negatives,
            seq_max_len,
            epitopes_array_blosum_division_test.negatives_mask,
            storage_folder, args,
            "{}/Test/{}/neighbours1/negatives".format(args.sequence_type,analysis_mode),
            ksize=ksize)
        test_high_conf_negatives_sim_results = VegvisirSimilarities.calculate_similarities_parallel(
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

def process_sequences(args,unique_lens,corrected_aa_types,seq_max_len,sequences_list,data,storage_folder="/home/lys/Dropbox/PostDoc/vegvisir/vegvisir/src/vegvisir/data/"):
    """Pads, randomizes or mutates a list of sequences to be converted onto a numpy array"""
    if len(unique_lens) > 1:  # Highlight: Pad the sequences (relevant when they differ in length)
        aa_dict = VegvisirUtils.aminoacid_names_dict(corrected_aa_types, zero_characters=["#"])
        if args.random_sequences:
            warnings.warn("Randomizing the sequence.If you do not wish to randomize the sequence please set args.random_sequences to False")
            sequences_pad_result = VegvisirLoadUtils.SequenceRandomGeneration(sequences_list,seq_max_len,args.seq_padding).run()
        elif args.num_mutations > 0:
            warnings.warn(
                "Performing {} mutaions to your sequence. If you do not wish to mutate your sequence please set n_mutations to 0")
            sequences_pad_result = VegvisirLoadUtils.PointMutations(sequences_list, seq_max_len, args.seq_padding,
                                                  args.num_mutations,args.idx_mutations).run()
        else:
            if args.shuffle_sequence:
                warnings.warn("Shuffling the sequence sites for testing purposes. If you do not wish to randomize the sequence please set args.shuffle_sequences to False")
            sequences_pad_result = VegvisirLoadUtils.SequencePadding(sequences_list, seq_max_len, args.seq_padding,
                                                                    args.shuffle_sequence).run()
        sequences_padded, sequences_padded_mask = zip(*sequences_pad_result)  # unpack list of tuples onto 2 lists
        
        if args.random_sequences or args.num_mutations > 0: #Store the randomized sequences
            VegvisirUtils.convert_to_pandas_dataframe(sequences_padded, data, storage_folder, args, use_test=True)

        blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(corrected_aa_types, args.subs_matrix,
                                                                                   zero_characters=["#"],
                                                                                   include_zero_characters=True)
    else:
        print("All sequences found to have the same length")
        aa_dict = VegvisirUtils.aminoacid_names_dict(corrected_aa_types)
        if args.random_sequences:
            warnings.warn("Randomizing the sequence for model-stress testing purposes. If you do not wish to randomize the sequence please set args.random_sequences to False")
            sequences_pad_result = VegvisirLoadUtils.SequenceRandomGeneration(sequences_list,seq_max_len,"no_padding").run()
        elif args.num_mutations > 0:
            warnings.warn("Performing {} mutations to your sequence. If you do not wish to mutate your sequence please set n_mutations to 0".format(args.num_mutations))
            sequences_pad_result = VegvisirLoadUtils.PointMutations(sequences_list, seq_max_len, "no_padding",args.num_mutations,args.idx_mutations).run()
        else:
            if args.shuffle_sequence:
                warnings.warn("Shuffling the sequence for testing purposes.If you do not wish to randomize the sequence please set args.shuffle_sequences to False")
            sequences_pad_result = VegvisirLoadUtils.SequencePadding(sequences_list, seq_max_len, "no_padding",args.shuffle_sequence).run()
        sequences_padded, sequences_padded_mask = zip(*sequences_pad_result)  # unpack list of tuples onto 2 lists
        blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(corrected_aa_types, args.subs_matrix,
                                                                                   zero_characters=[],
                                                                                   include_zero_characters=False)

    return sequences_padded,sequences_padded_mask,aa_dict,blosum_array,blosum_dict,blosum_array_dict

def process_data(data,args,storage_folder,script_dir,analysis_mode,filters_dict,features_names=None,plot_blosum=False,plot_umap=False,corrected_parameters=None):
    """
    Notes:
      - Mid-padding : https://www.nature.com/articles/s41598-020-71450-8
    :param pandas dataframe data: Contains Icore, immunodominance_score, immunodominance_score_scaled, training , partition and Rnk_EL
    :param args: Commmand line arguments
    :param storage_folder: Data location path
    """

    sequence_column = filters_dict["filter_kmers"][2]

    epitopes_list = data[sequence_column].values.tolist()


    #epitopes_list = functools.reduce(operator.iconcat, epitopes_list, [])  # flatten list of lists

    seq_max_len = len(max(epitopes_list, key=len)) #Highlight: If this gives an error there are some nan values
    epitopes_lens = np.array(list(map(len, epitopes_list)))
    unique_lens = list(set(epitopes_lens))
    if corrected_parameters is not None:
        corrected_aa_types,unique_lens = corrected_parameters
    else:
        corrected_aa_types = len(set().union(*epitopes_list))
        corrected_aa_types = corrected_aa_types + 1 if len(unique_lens) > 1 else corrected_aa_types #correct to include the gap symbol

    epitopes_padded, epitopes_padded_mask, aa_dict, blosum_array, blosum_dict, blosum_array_dict = process_sequences(args,unique_lens,corrected_aa_types,seq_max_len,epitopes_list,data,storage_folder)
    #Highlight: Plotting logos plots
    plot_logos = False
    if plot_logos:
        labels = data["target_corrected"].values.tolist()
        positives = np.array(labels).astype(bool)
        epitopes_positives = np.array(epitopes_padded)[positives].tolist()
        epitopes_negatives = np.array(epitopes_padded)[~positives].tolist()

        #epitopes_negatives
        VegvisirPlots.plot_logos(list(map(lambda seq: "".join(seq),epitopes_padded)),"{}/{}".format(storage_folder,args.dataset_name),"_{}_filter_kmers_{}".format(sequence_column,args.filter_kmers))
        VegvisirPlots.plot_logos(list(map(lambda seq: "".join(seq),epitopes_positives)),"{}/{}".format(storage_folder,args.dataset_name),"_{}_filter_kmers_{}_positives".format(sequence_column,args.filter_kmers))
        VegvisirPlots.plot_logos(list(map(lambda seq: "".join(seq),epitopes_negatives)),"{}/{}".format(storage_folder,args.dataset_name),"_{}_filter_kmers_{}_negatives".format(sequence_column,args.filter_kmers))

    save_intermediate_dataset = False #TODO: Remove eventually
    if save_intermediate_dataset:

        intermediate_dataset = pd.DataFrame({"{}".format(sequence_column):list(map(lambda seq:"".join(seq).replace("#",""),epitopes_padded)),
                                             "target_corrected": data["target_corrected"],
                                             "partitions": data["partition"],
                                             "training":data["training"],
                                             "immunodominance_score":data["immunodominance_score"],
                                             "org_name":data["org_name"],
                                             "confidence_scores": data["confidence_score"],
                                             "allele_encoded": data["allele_encoded"]
                                             })
        prefix1 = "shuffled_" if args.shuffle_sequence else ""
        prefix2 = "random_" if args.random_sequences else ""
        prefix3 = "fixed_length_" if args.filter_kmers else "variable_length_"
        prefix = prefix1 + prefix2 + prefix3  + sequence_column + "_"
        intermediate_dataset_train = intermediate_dataset[intermediate_dataset["training"] == True]
        intermediate_dataset_test = intermediate_dataset[intermediate_dataset["training"] == False]

        intermediate_dataset.to_csv("{}/benchmark_dataset/{}/{}sequences_{}.tsv".format(storage_folder,args.sequence_type,prefix,args.dataset_name),sep="\t",index=False)
        intermediate_dataset_train.to_csv("{}/benchmark_dataset/{}/{}_sequences_{}_TRAIN.tsv".format(storage_folder,args.sequence_type,prefix,args.dataset_name),sep="\t",index=False)
        intermediate_dataset_test.to_csv("{}/benchmark_dataset/{}/{}sequences_{}_TEST.tsv".format(storage_folder,args.sequence_type,prefix,args.dataset_name),sep="\t",index=False)

    epitopes_array_raw = np.array(epitopes_padded)

    if args.seq_padding == "replicated_borders":  # I keep it separately to avoid doing the np vectorized loop twice
        epitopes_array_int = np.vectorize(aa_dict.get)(epitopes_array_raw)
        epitopes_array_mask = np.array(epitopes_padded_mask)
        epitopes_array_int_mask = np.vectorize(aa_dict.get)(epitopes_array_mask)
        epitopes_mask = epitopes_array_int_mask.astype(bool)
    else:
        epitopes_array_int = np.vectorize(aa_dict.get)(epitopes_array_raw)
        if len(unique_lens) > 1: #there are some paddings, that equal 0, therefore we can set them to False
            epitopes_mask = epitopes_array_int.astype(bool)
        else: #there is no padding, therefore number 0 equals an amino acid
            epitopes_mask = np.ones_like(epitopes_array_int).astype(bool)
    if args.subset_data != "no": #TODO:Remove
        print("WARNING : Using a subset of the data of {}".format(args.subset_data))
        epitopes_array_raw = epitopes_array_raw[:args.subset_data]
        epitopes_mask = epitopes_mask[:args.subset_data]

    aa_frequencies = VegvisirUtils.calculate_aa_frequencies(epitopes_array_int,corrected_aa_types)
    blosum_max, blosum_weighted, variable_score = VegvisirUtils.process_blosum(blosum_array, aa_frequencies, seq_max_len,corrected_aa_types)

    n_data = epitopes_array_raw.shape[0]
    blosum_norm = np.linalg.norm(blosum_array[1:, 1:], axis=0)
    aa_list = [val for key, val in aa_dict.items() if val in list(blosum_array[:, 0])]
    blosum_norm_dict = dict(zip(aa_list,blosum_norm.tolist()))
    epitopes_array_blosum_norm = np.vectorize(blosum_norm_dict.get)(epitopes_array_int)
    if plot_blosum:
        VegvisirPlots.plot_blosum_cosine(blosum_array, storage_folder, args)
    epitopes_array_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(epitopes_array_int)
    epitopes_array_onehot_encoding = VegvisirUtils.convert_to_onehot(epitopes_array_int,dimensions=epitopes_array_blosum.shape[2])
    #Highlight: Features correlations
    # if args.learning_type == "unsupervised":
    #     #data_volumetrics(seq_max_len,epitopes_list, data, epitopes_mask, storage_folder, args, filters_dict,analysis_mode,plot_volumetrics=False,plot_covariance=True)
    #     all_sim_results = data_exploration(data, epitopes_array_blosum, epitopes_array_int, epitopes_mask, aa_dict, aa_list,
    #                                        blosum_norm, seq_max_len, storage_folder, args, corrected_aa_types,
    #                                        analysis_mode, filters_dict)
    #     positional_weights = all_sim_results.positional_weights
    #     positional_weights_mask = (positional_weights[..., None] > 0.6).any(-1)


    if args.dataset_name not in  ["viral_dataset6","viral_dataset8","viral_dataset10","viral_dataset11"]:
        try:
            all_sim_results = data_exploration(data, epitopes_array_blosum, epitopes_array_int, epitopes_mask, aa_dict, aa_list,
                             blosum_norm, seq_max_len, storage_folder, args, corrected_aa_types,analysis_mode,filters_dict)
            positional_weights = all_sim_results.positional_weights
            mean_weight= np.mean(positional_weights)
            positional_weights_mask = (positional_weights[..., None] > mean_weight).any(-1)
        except:
            warnings.warn("Created dummy positional weights")
            all_sim_results = SimilarityResults(positional_weights=np.ones((n_data, seq_max_len)),
                                                percent_identity_mean=None,
                                                cosine_similarity_mean=None,
                                                kmers_pid_similarity=None,
                                                kmers_cosine_similarity=None)
            positional_weights = np.ones((n_data, seq_max_len))
            positional_weights_mask = np.ones((n_data, seq_max_len)).astype(bool)
    else:
        warnings.warn("Created dummy positional weights")
        all_sim_results = SimilarityResults(positional_weights=np.ones((n_data,seq_max_len)),
                                               percent_identity_mean=None,
                                               cosine_similarity_mean=None,
                                               kmers_pid_similarity=None,
                                               kmers_cosine_similarity=None)
        positional_weights = np.ones((n_data,seq_max_len))
        positional_weights_mask = np.ones((n_data,seq_max_len)).astype(bool)

    #Highlight: Reattatch partition, identifier, label, immunodominance score
    labels = data["target_corrected"].values.tolist()
    identifiers = data.index.values.tolist() #TODO: reset index in process data function?
    alleles = data["allele_encoded"].values.tolist()

    partitions = data["partition"].values.tolist()
    training = data["training"].values.tolist()
    confidence_scores = data["confidence_score"].values.tolist()
    immunodominance_scores = data["immunodominance_score"].values.tolist()
    org_name = data["org_name"].values.tolist()

    if args.shuffle_labels:
        warnings.warn("Shuffling the data labels (train and test) to create a random dataset for model stress testing")
        np.random.seed(13)
        labels_shuffled = np.array(labels.copy())
        np.random.shuffle(labels_shuffled)
        labels = labels_shuffled

     #TODO : make a function?
    if features_names is not None:
        features_scores = data[features_names].to_numpy()
        #for feat in features_scores
        identifiers_labels_array = np.zeros((n_data, 1, seq_max_len + len(features_names)))
        identifiers_labels_array[:, 0, 0] = np.array(labels)
        identifiers_labels_array[:, 0, 1] = np.array(identifiers)
        identifiers_labels_array[:, 0, 2] = np.array(partitions)
        identifiers_labels_array[:, 0, 3] = np.array(training).astype(int)
        identifiers_labels_array[:, 0, 4] = np.array(immunodominance_scores)
        identifiers_labels_array[:, 0, 5] = np.array(confidence_scores)
        #identifiers_labels_array[:, 0, 6] = np.array(org_name)
        identifiers_labels_array[:, 0, 6] = np.array(alleles)


        epitopes_array = np.concatenate([epitopes_array_raw,features_scores],axis=1)
        epitopes_array_int = np.concatenate([epitopes_array_int,features_scores],axis=1)
        epitopes_array_blosum_norm = np.concatenate([epitopes_array_blosum_norm,features_scores],axis=1)

        data_array_raw = np.concatenate([identifiers_labels_array, epitopes_array[:, None]], axis=1)
        data_array_int = np.concatenate([identifiers_labels_array,epitopes_array_int[:, None]], axis=1)
        data_array_blosum_norm = np.concatenate([identifiers_labels_array,epitopes_array_blosum_norm[:, None]], axis=1)

        identifiers_labels_array_blosum = np.zeros((n_data, 1, seq_max_len + len(features_names), epitopes_array_blosum.shape[2]))
        identifiers_labels_array_blosum[:, 0, 0, 0] = np.array(labels)
        identifiers_labels_array_blosum[:, 0, 0, 1] = np.array(identifiers)
        identifiers_labels_array_blosum[:, 0, 0, 2] = np.array(partitions)
        identifiers_labels_array_blosum[:, 0, 0, 3] = np.array(training).astype(int)
        identifiers_labels_array_blosum[:, 0, 0, 4] = np.array(immunodominance_scores)
        identifiers_labels_array_blosum[:, 0, 0, 5] = np.array(confidence_scores)
        #identifiers_labels_array_blosum[:, 0, 0, 6] = np.array(org_name)
        identifiers_labels_array_blosum[:, 0, 0, 6] = np.array(alleles)



        features_array_blosum = np.zeros((n_data,len(features_names), epitopes_array_blosum.shape[2]))
        features_array_blosum[:,:,0] = features_scores

        epitopes_array_blosum = np.concatenate([epitopes_array_blosum,features_array_blosum],axis=1)
        epitopes_array_onehot_encoding = np.concatenate([epitopes_array_onehot_encoding,features_array_blosum],axis=1)
        data_array_blosum_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_blosum[:, None]],axis=1)
        data_array_onehot_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_onehot_encoding[:, None]], axis=1)

    else:
        identifiers_labels_array = np.zeros((n_data, 1, seq_max_len))
        identifiers_labels_array[:, 0, 0] = np.array(labels)
        identifiers_labels_array[:, 0, 1] = np.array(identifiers)
        identifiers_labels_array[:, 0, 2] = np.array(partitions)
        identifiers_labels_array[:, 0, 3] = np.array(training).astype(int)
        identifiers_labels_array[:, 0, 4] = np.array(immunodominance_scores)
        identifiers_labels_array[:, 0, 5] = np.array(confidence_scores)
        #identifiers_labels_array[:, 0, 6] = np.array(org_name)
        identifiers_labels_array[:, 0, 6] = np.array(alleles)

        data_array_raw = np.concatenate([identifiers_labels_array, epitopes_array_raw[:,None]], axis=1)
        data_array_int = np.concatenate([identifiers_labels_array, epitopes_array_int[:,None]], axis=1)
        data_array_blosum_norm = np.concatenate([identifiers_labels_array, epitopes_array_blosum_norm[:,None]], axis=1)

        identifiers_labels_array_blosum = np.zeros((n_data, 1, seq_max_len, epitopes_array_blosum.shape[2]))
        identifiers_labels_array_blosum[:, 0, 0, 0] = np.array(labels)
        identifiers_labels_array_blosum[:, 0, 0, 1] = np.array(identifiers)
        identifiers_labels_array_blosum[:, 0, 0, 2] = np.array(partitions)
        identifiers_labels_array_blosum[:, 0, 0, 3] = np.array(training).astype(int)
        identifiers_labels_array_blosum[:, 0, 0, 4] = np.array(immunodominance_scores)
        identifiers_labels_array_blosum[:, 0, 0, 5] = np.array(confidence_scores)
        #identifiers_labels_array_blosum[:, 0, 0, 6] = np.array(org_name)
        identifiers_labels_array_blosum[:, 0, 0, 6] = np.array(alleles)


        data_array_blosum_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_blosum[:,None]], axis=1)
        data_array_onehot_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_onehot_encoding[:,None]], axis=1)

    data_array_blosum_encoding_mask = np.broadcast_to(epitopes_mask[:, None, :, None], (n_data, 2, seq_max_len,corrected_aa_types)).copy()  # I do it like this in case the padding is not represented as 0, otherwise just use bool. Note: The first row of the second dimension is a dummy
    #distance_pid_cosine = VegvisirUtils.euclidean_2d_norm(percent_identity_mean,cosine_similarity_mean) #TODO: What to do with this?

    data_info = DatasetInfo(script_dir=script_dir,
                            storage_folder=storage_folder,
                            data_array_raw=data_array_raw,
                            data_array_int=torch.from_numpy(data_array_int),
                            data_array_int_mask=epitopes_mask,
                            data_array_blosum_encoding=torch.from_numpy(data_array_blosum_encoding),
                            data_array_blosum_encoding_mask=torch.from_numpy(data_array_blosum_encoding_mask),
                            data_array_onehot_encoding=torch.from_numpy(data_array_onehot_encoding),
                            data_array_onehot_encoding_mask=torch.from_numpy(data_array_blosum_encoding_mask),
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
                            features_names = features_names,
                            unique_lens=unique_lens,
                            blosum_weighted=blosum_weighted,
                            immunomodulate_dataset=None)

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
    pd.options.mode.chained_assignment = None #supresses some warnings

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

