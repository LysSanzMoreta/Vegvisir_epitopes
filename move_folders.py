import json
import os
import shutil

import pandas as pd


def move_folders():
    source = '/home/tuhingfg/Documents/source'
    destination = '/home/tuhingfg/Documents/destination'

    # gather all files
    allfiles = os.listdir(source)

    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        shutil.move(src_path, dst_path)

def best_params():
    pd.set_option('display.max_columns', None)
    folder_predefined_partitions = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_18_23h18min10s462702ms_3epochs_supervised_Icore_blosum_HPO/HyperparamOptimization_results.tsv"

    folder_random_stratified = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_19_14h56min21s560419ms_3epochs_supervised_Icore_blosum_TESTING_HPO/HyperparamOptimization_results.tsv"


    hpo_results = pd.read_csv(folder_predefined_partitions,sep="\t")

    print(hpo_results.columns)

    #hpo_results_sorted_valid_loss =  hpo_results.sort_values(by=['valid_loss'], ascending=True)
    # print(hpo_results_sorted_valid_loss["valid_loss"])
    # print(hpo_results_sorted_valid_loss["ROC_AUC_valid"])
    hpo_results_sorted_valid_auc =  hpo_results.sort_values(by=['ROC_AUC_valid'], ascending=False)

    print(hpo_results_sorted_valid_auc["valid_loss"])
    print(hpo_results_sorted_valid_auc["ROC_AUC_valid"])

    filter_col = [col for col in hpo_results_sorted_valid_auc if col.startswith('config')]
    config_df = hpo_results_sorted_valid_auc[filter_col]
    config_df.columns = [col.replace("config/","") for col in config_df.columns]

    print(config_df[["lr","beta1","beta2","eps","weight_decay","clip_norm","lrd","z_dim",
                                        "gru_hidden_dim","momentum","batch_size","encoding",
                                        "likelihood_scale","num_epochs","num_samples","hidden_dim"]].head(1))

    best_optimizer_config = config_df[["lr","beta1","beta2","eps","weight_decay","clip_norm","lrd","momentum"]].head(1).to_dict(orient="records")[0]

    print(best_optimizer_config)

    best_general_config = config_df[["batch_size", "encoding","likelihood_scale", "num_epochs", "num_samples", "hidden_dim","z_dim"]].head(1).to_dict(orient="records")[0]

    best_hyper_param_dict = {"optimizer_config":best_optimizer_config,
                             "general_config":best_general_config}


    json.dump(best_hyper_param_dict,open("BEST_hyperparameter_dict.p","w+"),indent=2)


if __name__ == "__main__":
    best_params()
