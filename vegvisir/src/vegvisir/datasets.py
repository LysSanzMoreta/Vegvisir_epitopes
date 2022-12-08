import os
import pandas as pd
import operator,functools
def available_datasets():
    """Prints the available datasets"""
    datasets = {0:"viral"
                }
    return datasets
def select_dataset(dataset_name,script_dir,args,update=True):
    """Selects from available datasets
    :param dataset_name: dataset of choice
    :param script_dir: Path from where the scriptis being executed
    :param update: If true it will download and update the most recent version of the dataset
    The dataset is organized as follows:
    id
    Icore
    allele
    protein_sequence
    Core
    Of
    Gp
    Gl
    Ip
    Il
    Rnk_EL
    org_id
    org_name
    prot_name
    uniprot_id	number_papers_positive	number_papers_negative	target	target_bin_2	start_prot	end_prot	filter_register	training	partition
    """
    func_dict = {"viral": viral_dataset,
                 }
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) #finds the /data folder of the repository

    dataset_load_fx = lambda f,dataset_name,current_path,storage_folder,args,update: lambda dataset_name,current_path,storage_folder,args,update: f(dataset_name,current_path,storage_folder,args,update)
    data_load_function = dataset_load_fx(func_dict[dataset_name],dataset_name,script_dir,storage_folder,args,update)
    dataset = data_load_function(dataset_name,script_dir,storage_folder,args,update)
    print("Data retrieved")

    return dataset

def viral_dataset(dataset_name,current_path,storage_folder,args,update):
    """Loads the viral dataset generated from ..."""
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
    exit()



