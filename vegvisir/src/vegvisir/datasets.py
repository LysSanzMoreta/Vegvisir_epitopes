import os
import pandas as pd
import operator,functools
import nnalign as VegvisirNNalign
def available_datasets():
    """Prints the available datasets"""
    datasets = {0:"viral"}
    return datasets
def select_dataset(dataset_name,script_dir,args,update=True):
    """Selects from available datasets
    :param dataset_name: dataset of choice
    :param script_dir: Path from where the scriptis being executed
    :param update: If true it will download and update the most recent version of the dataset
    """
    func_dict = {"viral": viral_dataset}
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) #finds the /data folder of the repository

    dataset_load_fx = lambda f,dataset_name,current_path,storage_folder,args,update: lambda dataset_name,current_path,storage_folder,args,update: f(dataset_name,current_path,storage_folder,args,update)
    data_load_function = dataset_load_fx(func_dict[dataset_name],dataset_name,script_dir,storage_folder,args,update)
    dataset = data_load_function(dataset_name,script_dir,storage_folder,args,update)
    print("Data retrieved")

    return dataset

def viral_dataset(dataset_name,current_path,storage_folder,args,update):
    """Loads the viral dataset generated from ...

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
    sequence_column = ["Core","Icore"][0]
    score_column = ["Rnk_EL","target"][1]
    data = pd.read_csv("{}/viral_dataset/Viruses_db_partitions.tsv".format(storage_folder),sep="\t")
    nnalign_input = data[[sequence_column,score_column,"training"]]
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
    nnalign_input_train.to_csv("{}/viral_dataset/viral_nnalign_input_train.tsv".format(storage_folder),sep="\t",index=False)
    nnalign_input_eval.to_csv("{}/viral_dataset/viral_nnalign_input_eval.tsv".format(storage_folder), sep="\t",index=False)

    if args.run_nnalign:
        VegvisirNNalign.run_nnalign(storage_folder)

    exit()



