"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import os
import time,datetime
import dill
import pandas as pd
import operator,functools
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict,namedtuple
import seaborn as sns
import dataframe_image as dfi
import torch
import vegvisir.nnalign as VegvisirNNalign
import vegvisir.utils as VegvisirUtils
import vegvisir.plots as VegvisirPlots
DatasetInfo = namedtuple("DatasetInfo",["script_dir","storage_folder","data_array_raw","data_array_int","data_array_int_mask",
                                        "data_array_blosum_encoding","data_array_blosum_encoding_mask","data_array_onehot_encoding","blosum",
                                        "n_data","max_len","corrected_aa_types","input_dim","percent_identity_mean","cosine_similarity_mean","kmers_pid_similarity","kmers_cosine_similarity"])
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


def viral_dataset2(dataset_name,script_dir,storage_folder,args,update):
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
      :param pandas dataframe: Results pandas dataframe with the following structure
          Icore:Interaction peptide core
          confidence_score: Number of + / Number of tested
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
    data_a["confidence_score"] = data_a["Assay_number_of_subjects_responded"]/data_a["Assay_number_of_subjects_tested"]
    data_a["Rnk_EL"] =data_b["Rnk_EL"]
    data_a.fillna(0,inplace=True)
    #Highlight: Scale-standarize values #TODO: Do separately for train, eval and test
    data_a = VegvisirUtils.minmax_scale(data_a,"confidence_score",suffix="_scaled")
    data_a = VegvisirUtils.minmax_scale(data_a,"Rnk_EL",suffix="_scaled") #Likelihood rank
    data_b.fillna(0, inplace=True)
    # print(data_a["target"].value_counts())
    # print(data_a.sort_values(by="confidence_score",ascending=True)[["confidence_score","target"]])
    data_a.loc[data_a["confidence_score_scaled"] <= 0.,"target_corrected"] = 0 #["target"] = 0. #Strict target reassignment
    #print(data_a.sort_values(by="confidence_score", ascending=True)[["confidence_score","target"]])
    data_a.loc[data_a["confidence_score_scaled"] > 0.,"target_corrected"] = 1.
    # print(data_a["target"].value_counts())
    # print("--------------------")
    # print(data_a["partition"].value_counts())
    # print("--------------------")
    # print(data_a["training"].value_counts())
    ndata = data_a.shape[0]
    fig, ax = plt.subplots(3, figsize=(7, 10))
    num_bins = 50
    ############LABELS #############
    freq, bins, patches = ax[0].hist(data_a["target"].to_numpy() , bins=2, density=True)
    ax[0].set_xlabel('Target/Label (0: Non-binder, 1: Binder)')
    ax[0].set_title(r'Histogram of targets/labels')
    ax[0].xaxis.set_ticks([0.25,0.75])
    ax[0].set_xticklabels([0,1])
    # Annotate the bars.
    for bar in patches: #iterate over the bars
        n_data_bin = (bar.get_height()*ndata)/2
        ax[0].annotate(format(n_data_bin, '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
    #######CONFIDENCE SCORES
    ax[1].hist(data_a["confidence_score_scaled"].to_numpy() , num_bins, density=True)
    ax[1].set_xlabel('Minmax scaled confidence score (N_+ / Subjects)')
    ax[1].set_title(r'Histogram of confidence scores')
    ##########RANK###################
    ax[2].hist(data_a["Rnk_EL"].to_numpy(), num_bins, density=True)
    ax[2].set_xlabel("Binding rank estimated by NetMHCpan-4.1")
    ax[2].set_title(r'Histogram of Rnk_EL scores')
    plt.ylabel("Counts")
    fig.tight_layout()
    plt.savefig("{}/{}/Viruses_histograms".format(storage_folder,args.dataset_name), dpi=300)
    plt.clf()
    data_info = process_data(data_a,args,storage_folder,script_dir)

    return data_info

def process_data(data,args,storage_folder,script_dir,plot_blosum=False):
    """
    :param pandas dataframe data: Contains Icore, confidence_score, confidence_score_scaled, training , partition and Rnk_EL
    :param args: Commmand line arguments
    :param storage_folder: Data location path
    """


    epitopes = data[["Icore"]].values.tolist()
    epitopes = functools.reduce(operator.iconcat, epitopes, [])  # flatten list of lists
    max_len = len(max(epitopes, key=len))
    epitopes_lens = np.array(list(map(len, epitopes)))
    unique_lens = list(set(epitopes_lens))
    corrected_aa_types = [args.aa_types + 1 if len(unique_lens) > 1 and (args.aa_types == 20 or args.aa_types == 24) else args.aa_types][0]
    if len(unique_lens) > 1:
        aa_dict = VegvisirUtils.aminoacid_names_dict(corrected_aa_types , zero_characters=["#"])
        #Pad the sequences (relevant when not all of them are 9-mers)
        epitopes = [list(seq.ljust(max_len, "#")) for seq in epitopes]
        blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(corrected_aa_types , args.subs_matrix,
                                                                                   zero_characters= ["#"],
                                                                                   include_zero_characters=True)

    else:
        aa_dict = VegvisirUtils.aminoacid_names_dict(args.aa_types)
        epitopes = [list(seq) for seq in epitopes]
        blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(corrected_aa_types, args.subs_matrix,
                                                                                   zero_characters=["#"],
                                                                                   include_zero_characters=False)


    # print(epitopes[0])
    # print(epitopes[1])
    # print(epitopes[2])
    # print(epitopes[3])
    epitopes_array = np.array(epitopes)
    if args.subset_data != "no":
        print("WARNING : Using a subset of the data of {}".format(args.subset_data))
        epitopes_array = epitopes_array[:args.subset_data]
    epitopes_array_int = np.vectorize(aa_dict.get)(epitopes_array)
    epitopes_mask = epitopes_array_int.astype(bool)

    if plot_blosum:
        blosum_cosine = VegvisirUtils.cosine_similarity(blosum_array[1:, 1:], blosum_array[1:, 1:])
        aa_dict = VegvisirUtils.aminoacid_names_dict(args.aa_types,zero_characters=["#"])
        aa_list =[key for key,val in aa_dict.items() if val in list(blosum_array[:,0])]
        blosum_cosine_df = pd.DataFrame(blosum_cosine,columns=aa_list,index=aa_list)
        sns.heatmap(blosum_cosine_df.to_numpy(),
                    xticklabels=blosum_cosine_df.columns.values,
                    yticklabels=blosum_cosine_df.columns.values,annot=True,annot_kws={"size": 4},fmt=".2f")
        plt.savefig('{}/{}/blosum_cosine.png'.format(storage_folder,args.dataset_name),dpi=600)

    epitopes_array_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(epitopes_array_int)
    epitopes_array_onehot_encoding = VegvisirUtils.convert_to_onehot(epitopes_array_int,dimensions=epitopes_array_blosum.shape[2])

    n_data = epitopes_array.shape[0]
    ksize = 3 #TODO: manage in args
    if not os.path.exists("{}/{}/similarities/percent_identity_mean.npy".format(storage_folder,args.dataset_name)):
        print("Epitopes similarity matrices not existing, calculating ....")
        percent_identity_mean,cosine_similarity_mean,kmers_pid_similarity,kmers_cosine_similarity = VegvisirUtils.calculate_similarity_matrix(epitopes_array_blosum,max_len,epitopes_mask,ksize=ksize)
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

    #TODO: Reattatch partition, identifier, label, confidence score
    labels = data[["target_corrected"]].values.tolist()
    identifiers = data.index.values.tolist() #TODO: reset index?
    partitions = data[["partition"]].values.tolist()
    training = data[["training"]].values.tolist()

    identifiers_labels_array = np.zeros((n_data,1,max_len))
    identifiers_labels_array[:,0,0] = np.array(labels).squeeze(-1)
    identifiers_labels_array[:,0,1] = np.array(identifiers)
    identifiers_labels_array[:,0,2] = np.array(partitions).squeeze(-1)
    identifiers_labels_array[:,0,3] = np.array(training).squeeze(-1).astype(int)


    data_array_raw = np.concatenate([identifiers_labels_array, epitopes_array[:,None]], axis=1)
    data_array_int = np.concatenate([identifiers_labels_array, epitopes_array_int[:,None]], axis=1)

    identifiers_labels_array_blosum = np.zeros((n_data,1, max_len, epitopes_array_blosum.shape[2]))
    identifiers_labels_array_blosum[:,0,0,0] = np.array(labels).squeeze(-1)
    identifiers_labels_array_blosum[:,0,0,1] = np.array(identifiers)
    identifiers_labels_array_blosum[:,0,0,2] = np.array(partitions).squeeze(-1)
    identifiers_labels_array_blosum[:,0,0,3] = np.array(training).squeeze(-1).astype(int)


    data_array_blosum_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_blosum[:,None]], axis=1)
    data_array_onehot_encoding = np.concatenate([identifiers_labels_array_blosum, epitopes_array_onehot_encoding[:,None]], axis=1)
    data_array_blosum_encoding_mask = epitopes_mask.repeat(data_array_blosum_encoding.shape[1], axis=1).repeat(corrected_aa_types, axis=-1).reshape((n_data, data_array_int.shape[1], max_len,corrected_aa_types))

    #distance_pid_cosine = VegvisirUtils.euclidean_2d_norm(percent_identity_mean,cosine_similarity_mean) #TODO: What to do with this?
    data_info = DatasetInfo(script_dir=script_dir,
                            storage_folder=storage_folder,
                            data_array_raw=data_array_raw,
                            data_array_int=torch.from_numpy(data_array_int),
                            data_array_int_mask=epitopes_mask,
                            data_array_blosum_encoding=torch.from_numpy(data_array_blosum_encoding),
                            data_array_blosum_encoding_mask=torch.from_numpy(data_array_blosum_encoding_mask),
                            data_array_onehot_encoding=torch.from_numpy(data_array_onehot_encoding),
                            blosum=blosum_array,
                            n_data=n_data,
                            max_len=max_len,
                            corrected_aa_types = corrected_aa_types,
                            input_dim=corrected_aa_types, # + 1 if gaps are present
                            percent_identity_mean=percent_identity_mean,
                            cosine_similarity_mean=cosine_similarity_mean,
                            kmers_pid_similarity=kmers_pid_similarity,
                            kmers_cosine_similarity=kmers_cosine_similarity
                            )
    return data_info




