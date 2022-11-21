import os
def select_dataset(dataset_name,script_dir,args,update=True):
    """Selects from available datasets
    :param dataset_name: dataset of choice
    :param script_dir: Path from where the scriptis being executed
    :param update: If true it will download and update the most recent version of the dataset
    The dataset is organizzed as follows:
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
    func_dict = {"mcpastcr": Viral_dataset,
                 }
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) #finds the /data folder of the repository

    dataset_load_fx = lambda f,dataset_name,current_path,storage_folder,args,update: lambda dataset_name,current_path,storage_folder,args,update: f(dataset_name,current_path,storage_folder,args,update)
    data_load_function = dataset_load_fx(func_dict[dataset_name],dataset_name,script_dir,storage_folder,args,update)
    dataset = data_load_function(dataset_name,script_dir,storage_folder,args,update)
    print("Data retrieved")

    return dataset

def Viral_dataset():
    """"""



