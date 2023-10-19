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
    folder_predefined_partitions = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_21_16h19min57s991317ms_HPO_supervised_Icore_blosum_DONOTDELETE/HyperparamOptimization_results.tsv"

    folder_random_stratified = "/home/lys/Dropbox/PostDoc/vegvisir/PLOTS_Vegvisir_viral_dataset9_2023_08_19_14h56min21s560419ms_3epochs_supervised_Icore_blosum_TESTING_HPO/HyperparamOptimization_results.tsv"


    hpo_results = pd.read_csv(folder_predefined_partitions,sep="\t")


    #hpo_results_sorted_valid_loss =  hpo_results.sort_values(by=['valid_loss'], ascending=True)
    # print(hpo_results_sorted_valid_loss["valid_loss"])
    # print(hpo_results_sorted_valid_loss["ROC_AUC_valid"])
    hpo_results_sorted_valid_auc =  hpo_results.sort_values(by=['ROC_AUC_valid'], ascending=False).reset_index()

    #print(hpo_results_sorted_valid_auc["valid_loss"])
    #print(hpo_results_sorted_valid_auc["ROC_AUC_valid"])

    filter_col = [col for col in hpo_results_sorted_valid_auc if col.startswith('config')]
    config_df = hpo_results_sorted_valid_auc[filter_col]
    config_df.columns = [col.replace("config/","") for col in config_df.columns]

    #print(config_df[["lr","beta1","beta2","eps","weight_decay","clip_norm","lrd","z_dim","momentum","batch_size","encoding","likelihood_scale","num_epochs","num_samples","hidden_dim"]].head(20))


    onehot_encoding = config_df[config_df["encoding"] == "onehot"].reset_index()
    blosum_encoding = config_df[config_df["encoding"] == "blosum"].reset_index()


    def save_best(config_df,name=""):

        best_optimizer_config = config_df[["lr","beta1","beta2","eps","weight_decay","clip_norm","lrd","momentum"]].head(1).to_dict(orient="records")[0]

        best_general_config = config_df[["batch_size", "encoding","likelihood_scale", "num_epochs", "num_samples", "hidden_dim","z_dim"]].head(1).to_dict(orient="records")[0]

        best_hyper_param_dict = {"optimizer_config":best_optimizer_config,
                                 "general_config":best_general_config}
        json.dump(best_hyper_param_dict,open("BEST_hyperparameter_dict_{}.p".format(name),"w+"),indent=2)


    save_best(onehot_encoding,"onehot")
    save_best(blosum_encoding,"blosum")



def analysis_models():
    """Analyses the results of all possible model combinations (stress testing)"""



    dict_results_likelihood = {"supervised(Icore)":
                            {"vd9-10":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset9_2023_08_03_14h37min35s894838ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_10",
                            "vd9-20":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset9_2023_08_03_16h14min44s785096ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_20",
                            "vd9-30":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset9_2023_08_03_17h48min26s866987ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_30",
                            "vd9-40":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset9_2023_08_03_19h21min17s085055ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_40",
                            "vd3-30": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset3_2023_08_03_21h44min10s036982ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_30",
                            "vd3-40": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset3_2023_08_03_23h04min07s833125ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_40",
                            "vd3-50": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset3_2023_08_04_00h25min53s635200ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_50",
                            "vd3-60":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Supervised/Icore/Likelihood_tuning/PLOTS_Vegvisir_viral_dataset3_2023_08_04_01h47min14s179667ms_60epochs_supervised_Icore_blosum_TESTING_likelihood_60"
                             }}



    #Highlight: Likelihood 100
    dict_results_predefined_partitions_100 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_00h21min39s228581ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_02h02min31s779860ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_03h09min53s385938ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_04h47min05s598852ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_05h53min07s553393ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_07h30min17s042073ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_08h35min44s182897ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_10h14min30s578730ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_11h21min04s589546ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_13h00min11s055366ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_13h32min18s725061ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_15h11min03s424849ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_15h45min21s460981ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_17h22min59s537148ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_19h11min15s401671ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_20h49min36s271884ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}




    dict_results_random_stratified_partitions_100 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_00h21min41s081591ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_01h58min21s438713ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_03h03min37s477755ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_04h37min19s658399ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_05h41min19s365333ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_07h13min19s842305ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_08h17min38s989963ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_11_09h49min01s283379ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_10h53min57s031443ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_12h27min47s743657ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_13h01min00s990819ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_14h34min45s419777ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_15h07min24s308813ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_16h37min00s237374ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_19h11min45s466440ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_100/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_11_20h46min22s152822ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}



    #Highlight: Likelihood 80
    dict_results_predefined_partitions_80 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_16h22min50s455264ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_18h00min16s910930ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_18h59min48s556463ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_20h35min30s186381ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    #"raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_13_22h08min17s858639ms_60epochs_supervised_Icore_blosum_TESTING_lk80",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_21h41min01s940264ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_14_23h20min57s783599ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_00h27min33s253377ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_02h04min06s395090ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_03h11min14s891078ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_04h47min43s218957ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_05h19min21s010814ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_06h56min53s143680ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_07h30min54s481085ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_09h06min53s142223ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_09h40min41s329685ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_11h15min29s130588ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}

    dict_results_random_stratified_partitions_80 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_01h06min29s238603ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_02h40min26s491870ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_03h45min58s786810ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_05h17min55s585799ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_06h21min58s024047ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_07h58min09s427052ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_09h01min57s241008ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_10h31min10s087815ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_11h37min19s196100ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_13h01min31s378988ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_13h35min14s337920ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_15h07min22s959488ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_15h37min52s131820ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_17h15min35s592585ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_17h46min54s694050ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_80/Random_stratified/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_15_19h11min20s045530ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}


    #Highlight: Likelihood 60:

    dict_results_predefined_partitions_60 = {"Icore":{
                                                    "random-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_15_23h23min22s794823ms_60epochs_supervised_Icore_blosum_random_TESTING",
                                                    "random-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_01h10min32s340159ms_60epochs_supervised_Icore_blosum_random_TESTING_9mers",
                                                    "shuffled-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_02h23min08s887505ms_60epochs_supervised_Icore_blosum_shuffled_TESTING",
                                                    "shuffled-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_03h46min01s482802ms_60epochs_supervised_Icore_blosum_shuffled_TESTING_9mers",
                                                    "raw-blosum-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_04h43min52s657889ms_60epochs_supervised_Icore_blosum_TESTING",
                                                    "raw-blosum-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_06h06min06s120049ms_60epochs_supervised_Icore_blosum_TESTING_9mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_07h15min00s364188ms_60epochs_supervised_Icore_onehot_TESTING",
                                                     "raw-onehot-9mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore/PLOTS_Vegvisir_viral_dataset9_2023_08_16_08h42min55s171815ms_60epochs_supervised_Icore_onehot_TESTING_9mers"
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_15h34min52s686599ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING",
                                                     "random-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_17h55min59s277441ms_60epochs_supervised_Icore_non_anchor_blosum_random_TESTING_8mers",
                                                     "shuffled-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_18h36min54s466004ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING",
                                                     "shuffled-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_22h24min27s815811ms_60epochs_supervised_Icore_non_anchor_blosum_shuffled_TESTING_8mers",
                                                     "raw-blosum-variable-length": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_16_23h00min36s001099ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING",
                                                     "raw-blosum-8mers": "/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_17_00h34min23s937002ms_60epochs_supervised_Icore_non_anchor_blosum_TESTING_8mers",
                                                     "raw-onehot-variable-length":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_17_01h10min24s412028ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING",
                                                     "raw-onehot-8mers":"/home/lys/Dropbox/PostDoc/vegvisir/Benchmark/Vegvisir_benchmarking/Likelihood_60/Predefined_partitions/Icore_non_anchor/PLOTS_Vegvisir_viral_dataset9_2023_08_17_02h38min57s577222ms_60epochs_supervised_Icore_non_anchor_onehot_TESTING_8mers"
                                         }}

    dict_results_random_stratified_partitions_60 = {"Icore":{
                                                    "random-blosum-variable-length":"",
                                                    "random-blosum-9mers":"",
                                                    "shuffled-blosum-variable-length":"",
                                                    "shuffled-blosum-9mers":"",
                                                    "raw-blosum-variable-length":"",
                                                    "raw-blosum-9mers":"",
                                                     "raw-onehot-variable-length":"",
                                                     "raw-onehot-9mers":""
                                                   },
                                         "Icore_non_anchor":{
                                                     "random-blosum-variable-length": "",
                                                     "random-blosum-8mers": "",
                                                     "shuffled-blosum-variable-length": "",
                                                     "shuffled-blosum-8mers": "",
                                                     "raw-blosum-variable-length": "",
                                                     "raw-blosum-8mers": "",
                                                     "raw-onehot-variable-length":"",
                                                     "raw-onehot-8mers":""
                                         }}





if __name__ == "__main__":
    best_params()
